"""RoboTwin2 absolute-action SFT adapters for the DexDataset/DM05 pipeline."""

from __future__ import annotations

import random

import cv2
import numpy as np
import torch
from PIL import Image

from dexbotic.data.dataset.dex_dataset import DexDataset
from dexbotic.data.dataset.dm05_data import DM05DataCollator, DM05ImagePreprocess
from dexbotic.data.dataset.transform.action import ActionNorm


class RoboTwin2Dataset(DexDataset):
    """Filter episodes shorter than one second; keep base frame sampling."""

    def _get_index_cache(self, data_path):
        cache = super()._get_index_cache(data_path)
        return {
            **cache,
            "data": {path: n for path, n in cache["data"].items() if n >= 30},
        }


class RoboTwin2ActionNorm(ActionNorm):
    """Reference min/max normalization: no clipping, zero absent semantic dims."""

    def _normalize(self, data, stats):
        lo = np.asarray(stats["min"], dtype=np.float32)
        hi = np.asarray(stats["max"], dtype=np.float32)
        data = np.asarray(data, dtype=np.float32)
        out = ((data - lo) / (hi - lo + 1e-6) * 2.0 - 1.0).astype(np.float32)
        return np.where((lo == 0) & (hi == 0), 0.0, out).astype(np.float32)


class RoboTwin2ImagePreprocess(DM05ImagePreprocess):
    """Match dm05_sim: centered black square padding, OpenCV linear resize."""

    def process_pil(self, image):
        if image is None:
            raise ValueError("RoboTwin2 SFT requires all three camera images")
        array = np.asarray(
            image.convert("RGB") if isinstance(image, Image.Image) else image
        )
        height, width = array.shape[:2]
        size = max(height, width)
        top, left = (size - height) // 2, (size - width) // 2
        array = cv2.copyMakeBorder(
            array,
            top,
            size - height - top,
            left,
            size - width - left,
            cv2.BORDER_CONSTANT,
            value=0,
        )
        return Image.fromarray(
            cv2.resize(array, self.target_size, interpolation=cv2.INTER_LINEAR)
        )


def refine_robotwin2_text(text):
    text = str(text).strip() if text is not None else ""
    if text and text[-1] not in ".!?。！？…":
        text += "."
    return text


class SampleRoboTwin2Prompt:
    """The reference SamplePrompt with grounding and prompt_add_prob disabled."""

    def __call__(self, data, **kwargs):
        meta = kwargs.get("meta_data", {})
        for index in meta.get("fram_indicies", range(len(data["prompt"]))):
            frame = {
                key: value[index] for key, value in data.items() if key != "meta_data"
            }
            prompt = ""
            # Preserve the reference's random stream, including its probability draw.
            if random.random() < 0.0 or frame.get("subtask") is None:
                original = random.choice(
                    [frame["prompt"], *frame.get("rewrite_prompt", [])]
                )
                prompt = f"Task: {refine_robotwin2_text(original)}\n"
            if frame.get("prompt_template"):
                random.random()
            if "subtask" in frame:
                subtask = random.choice(
                    [frame["subtask"], *frame.get("subtask_rewrites", [])]
                )
                prompt += f"Subtask: {subtask}\n"
            elif "subtask_rewrites" in frame:
                prompt += f"Subtask: {random.choice(frame['subtask_rewrites'])}\n"
            data["prompt"][index] = refine_robotwin2_text(prompt)
        return data


def state_bins(values, n_bins=256):
    normalized = (np.clip(np.asarray(values, dtype=np.float32), -1.0, 1.0) + 1.0) / 2.0
    return np.clip(np.floor(normalized * (n_bins - 1)).astype(int), 0, n_bins - 1)


class RoboTwin2DM05DataCollator(DM05DataCollator):
    image_prompts = ("Head", "Left wrist", "Right wrist")

    def _tokenize_sample(self, instance, prompt, pil_views, pil_history_views):
        if pil_history_views:
            raise ValueError("The RoboTwin2 SFT recipe does not use history images")
        content = [
            {
                "type": "text",
                "text": f"Robot: Aloha RoboTwin2\nOverall speed: 0.5\n{prompt}\n",
            }
        ]
        for label, image in zip(self.image_prompts, pil_views, strict=True):
            content[-1]["text"] += f"{label} image: "
            content.extend(
                [{"type": "image", "image": image}, {"type": "text", "text": ""}]
            )
        content[-1]["text"] = (
            "States: " + " ".join(map(str, state_bins(instance["state"][:14]))) + "\n"
        )
        action = instance["action"]
        answer = " ".join(map(str, state_bins(action[action.shape[0] // 2, :14])))
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt"
        )
        if (
            self.max_length is not None
            and inputs["input_ids"].shape[1] > self.max_length
        ):
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            overflow = inputs["input_ids"].shape[1] - self.max_length
            shortened = self.tokenizer.decode(
                prompt_ids[: max(0, len(prompt_ids) - overflow - 16)],
                skip_special_tokens=False,
            ).strip()
            content[0]["text"] = content[0]["text"].replace(prompt, shortened, 1)
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, return_dict=True, return_tensors="pt"
            )
        if (
            self.max_length is not None
            and inputs["input_ids"].shape[1] > self.max_length
        ):
            raise ValueError(
                "RoboTwin2 sample exceeds max_length after shortening the task prompt"
            )
        inputs["labels"] = self._answer_labels(inputs["input_ids"], answer)
        return inputs

    def _answer_labels(self, input_ids, answer):
        """Mask answer, end-of-turn and its newline out of suffix attention."""
        ids = input_ids[0].tolist()
        prefix = self.tokenizer.encode(
            "<start_of_turn>model\n", add_special_tokens=False
        )
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
        eos = self.tokenizer.encode("<end_of_turn>", add_special_tokens=False)[0]
        newline = self.tokenizer.encode("\n", add_special_tokens=False)

        def find(tokens, start):
            for index in range(start, len(ids) - len(tokens) + 1):
                if ids[index : index + len(tokens)] == tokens:
                    return index
            raise ValueError(
                "Cannot locate the RoboTwin2 assistant answer in tokenized input"
            )

        start = find(prefix, 0) + len(prefix)
        turn_end = find([eos], start)
        start = find(answer_ids, start)
        end = start + len(answer_ids)
        if end > turn_end:
            raise ValueError("RoboTwin2 answer crosses the assistant turn boundary")
        if end < len(ids) and ids[end] == eos:
            end += 1
        if newline and end < len(ids) and ids[end] == newline[0]:
            end += 1
        labels = torch.full_like(input_ids, -100)
        labels[:, start:end] = input_ids[:, start:end]
        return labels
