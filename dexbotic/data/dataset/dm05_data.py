"""DexDataset glue for DM05 (Gemma3 processor + flow-matching action head)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from dexbotic.data.dataset.transform.action import ActionNorm
from dexbotic.data.dataset.transform.common import ToTensor
from dexbotic.history import (
    HistoryImageSpec,
    collate_history_image_tensors,
    validate_history_images,
)
from dexbotic.model.dm05.dm05_utils import (
    HISTORY_IMAGE_TOKEN,
    HISTORY_PAD_TOKEN,
    HISTORY_TOKENS_PER_IMAGE,
)


class DM05ActionNorm(ActionNorm):
    """Quantile normalization with clipping and zeroed constant dimensions."""

    def _normalize(self, data, stats):
        lo = np.asarray(stats["min"], dtype=np.float32)
        hi = np.asarray(stats["max"], dtype=np.float32)
        data = np.clip(np.asarray(data, dtype=np.float32), lo, hi)
        out = ((data - lo) / (hi - lo + 1e-6) * 2.0 - 1.0).astype(np.float32)
        return np.where((lo == 0) & (hi == 0), 0.0, out).astype(np.float32)


class DM05ToTensor(ToTensor):
    def __call__(self, data):
        if isinstance(data, dict):
            return {key: self.__call__(value) for key, value in data.items()}
        if isinstance(data, list):
            return [self.__call__(item) for item in data]
        if isinstance(data, str) or data is None:
            return data
        return torch.as_tensor(data)


class DM05ImagePreprocess:
    target_size = (448, 448)

    def process_pil(self, image: Image.Image | None) -> Image.Image:
        if image is None:
            return Image.new("RGB", self.target_size, color=(0, 0, 0))
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        image = image.convert("RGB")
        width, height = image.size
        size = max(width, height)
        if width != height:
            canvas = Image.new("RGB", (size, size), color=(0, 0, 0))
            canvas.paste(image, ((size - width) // 2, (size - height) // 2))
            image = canvas
        resample = getattr(Image, "Resampling", Image).BILINEAR
        return image.resize(self.target_size, resample)

    def __call__(self, image, **kwargs):
        arr = np.array(self.process_pil(image), dtype=np.uint8)
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class DM05Tokenization:
    def __call__(self, conversations, has_image: bool = True, **kwargs):
        prompt = ""
        for turn in conversations:
            if turn.get("from") == "human":
                prompt = turn.get("value", "") or ""
                break
        data = prompt.encode("utf-8")
        if len(data) == 0:
            input_ids = torch.zeros(1, dtype=torch.long)
        else:
            input_ids = torch.tensor(list(data), dtype=torch.long)
        return {"input_ids": input_ids, "labels": torch.zeros(1, dtype=torch.long)}


class DM05DataCollator:
    image_prompts = ("Head", "Left wrist")

    def __init__(
        self,
        processor: AutoProcessor,
        max_length: int | None = None,
        valid_action_dim: int = 7,
        model_action_dim: int = 32,
        chunk_size: int = 10,
        history_enabled: bool = False,
        max_history_images: int = 5,
    ):
        self.processor = processor
        self.tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        self.max_length = max_length
        self.valid_action_dim = valid_action_dim
        self.model_action_dim = model_action_dim
        self.chunk_size = chunk_size
        self.pad_token_id = self.tokenizer.pad_token_id
        self.history_spec = HistoryImageSpec(
            enabled=bool(history_enabled),
            max_images=int(max_history_images),
        )
        self.history_enabled = self.history_spec.enabled
        self.history_token_id = self.tokenizer.convert_tokens_to_ids(
            HISTORY_IMAGE_TOKEN
        )

    def _tokenize_instance(
        self,
        prompt: str,
        pil_views: list[Image.Image],
        pil_history_views: list[Image.Image] | None = None,
    ) -> dict[str, torch.Tensor]:
        text = f"Robot: Franka\nOverall speed: 0.5\nTask: {prompt}.\n"
        user_content = [{"type": "text", "text": text}]
        history_pixel_values = None
        pil_history_views = validate_history_images(
            pil_history_views or [],
            self.history_spec,
            value_name="pil_history_views",
        )
        if self.history_enabled:
            n_valid = len(pil_history_views)
            user_content[-1]["text"] += "History images: "
            user_content[-1]["text"] += (
                HISTORY_PAD_TOKEN
                * (HISTORY_TOKENS_PER_IMAGE * (self.history_spec.max_images - n_valid))
                + ((HISTORY_IMAGE_TOKEN * HISTORY_TOKENS_PER_IMAGE) + "\n") * n_valid
            )
            if pil_history_views:
                history_pixel_values = self.processor.image_processor(
                    images=[image.convert("RGB") for image in pil_history_views],
                    return_tensors="pt",
                )["pixel_values"]
        for label, image in zip(self.image_prompts, pil_views, strict=True):
            if user_content[-1]["type"] == "text":
                user_content[-1]["text"] += f"{label} image: "
            else:
                user_content.append({"type": "text", "text": f"{label} image: "})
            user_content.append({"type": "image", "image": image})
        messages = [{"role": "user", "content": user_content}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if (
            self.max_length is not None
            and inputs["input_ids"].shape[1] > self.max_length
        ):
            prompt_token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            overflow = inputs["input_ids"].shape[1] - self.max_length
            keep_tokens = max(0, len(prompt_token_ids) - overflow - 16)
            if keep_tokens < len(prompt_token_ids):
                shortened = self.tokenizer.decode(
                    prompt_token_ids[:keep_tokens], skip_special_tokens=False
                ).strip()
                user_content[0]["text"] = user_content[0]["text"].replace(
                    prompt, shortened, 1
                )
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
        if (
            self.max_length is not None
            and inputs["input_ids"].shape[1] > self.max_length
        ):
            raise ValueError(
                f"DM05 sequence length {inputs['input_ids'].shape[1]} exceeds "
                f"max_length={self.max_length}; truncating would split image "
                "tokens from pixel_values."
            )
        if history_pixel_values is not None:
            history_mask = inputs["input_ids"] == self.history_token_id
            expected_tokens = (
                int(history_pixel_values.shape[0]) * HISTORY_TOKENS_PER_IMAGE
            )
            actual_tokens = int(history_mask.sum().item())
            if actual_tokens != expected_tokens:
                raise ValueError(
                    "DM05 history placeholder count does not match history "
                    f"images: expected {expected_tokens}, got {actual_tokens}."
                )
            inputs["token_type_ids"] = inputs["token_type_ids"].clone()
            inputs["token_type_ids"][history_mask] = 1
            inputs["history_pixel_values"] = history_pixel_values
            inputs["history_mask"] = history_mask
        return inputs

    def _tokenize_sample(self, instance, prompt, pil_views, pil_history_views):
        return self._tokenize_instance(prompt, pil_views, pil_history_views)

    def __call__(self, instances: Sequence[dict]) -> dict[str, torch.Tensor]:
        tokenized = []
        actions = []
        for inst in instances:
            vals = [int(x) for x in inst["input_ids"].tolist()]
            prompt = bytes(vals).decode("utf-8", errors="ignore") if any(vals) else ""
            image = inst["image"]
            if image.ndim == 3:
                image = image[None]
            pil_views = [
                Image.fromarray(view.permute(1, 2, 0).to(torch.uint8).cpu().numpy())
                for view in image
            ]
            history_images = inst.get("history_images")
            pil_history_views = []
            if history_images is not None:
                if history_images.ndim == 3:
                    history_images = history_images[None]
                pil_history_views = [
                    Image.fromarray(view.permute(1, 2, 0).to(torch.uint8).cpu().numpy())
                    for view in history_images
                ]
            tokenized.append(
                self._tokenize_sample(inst, prompt, pil_views, pil_history_views)
            )
            actions.append(inst["action"].float())

        max_len = max(item["input_ids"].shape[1] for item in tokenized)
        labels = []
        input_ids, attention_mask, token_type_ids, pixel_values = [], [], [], []
        has_history = any("history_mask" in item for item in tokenized)
        history_masks = []
        per_sample_history_pixel_values = []
        for item in tokenized:
            pad_len = max_len - item["input_ids"].shape[1]
            if "labels" in item:
                labels.append(
                    torch.nn.functional.pad(item["labels"], (0, pad_len), value=-100)
                )
            input_ids.append(
                torch.cat(
                    [
                        item["input_ids"],
                        torch.full(
                            (1, pad_len),
                            self.pad_token_id,
                            dtype=item["input_ids"].dtype,
                        ),
                    ],
                    dim=1,
                )
            )
            attention_mask.append(
                torch.cat(
                    [
                        item["attention_mask"],
                        torch.zeros((1, pad_len), dtype=item["attention_mask"].dtype),
                    ],
                    dim=1,
                )
            )
            token_type_ids.append(
                torch.cat(
                    [
                        item["token_type_ids"],
                        torch.zeros((1, pad_len), dtype=item["token_type_ids"].dtype),
                    ],
                    dim=1,
                )
            )
            pixel_values.append(item["pixel_values"])
            per_sample_history_pixel_values.append(item.get("history_pixel_values"))
            if has_history:
                item_history_mask = item.get(
                    "history_mask",
                    torch.zeros_like(item["input_ids"], dtype=torch.bool),
                )
                history_masks.append(
                    torch.cat(
                        [
                            item_history_mask,
                            torch.zeros((1, pad_len), dtype=torch.bool),
                        ],
                        dim=1,
                    )
                )

        action = torch.stack(actions, dim=0)
        action_mask = torch.zeros(
            len(instances), self.chunk_size, self.model_action_dim, dtype=action.dtype
        )
        action_mask[..., : self.valid_action_dim] = 1.0
        batch = {
            "input_ids": torch.cat(input_ids, dim=0),
            "attention_mask": torch.cat(attention_mask, dim=0),
            "token_type_ids": torch.cat(token_type_ids, dim=0),
            "pixel_values": torch.cat(pixel_values, dim=0),
            "action": action,
            "action_mask": action_mask,
        }
        if labels:
            if len(labels) != len(instances):
                raise ValueError("DM05 samples must either all have labels or none")
            batch["labels"] = torch.cat(labels, dim=0)
        if has_history:
            history_batch = collate_history_image_tensors(
                per_sample_history_pixel_values,
                max_images=self.history_spec.max_images,
            )
            if history_batch.pixel_values is None:
                raise RuntimeError("history masks exist without history pixel values")
            batch["history_mask"] = torch.cat(history_masks, dim=0)
            batch["history_pixel_values"] = history_batch.pixel_values
        return batch
