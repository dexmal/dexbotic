import argparse
import copy
import io
import math
import os
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from distutils.util import strtobool
from typing import List, Optional, Union

import numpy as np
import torch
import transformers
from easydict import EasyDict
from flask import jsonify, request
from loguru import logger
from PIL import Image, ImageDraw
from transformers import AutoTokenizer, BaseImageProcessor

from dexbotic.constants import IMAGE_TOKEN_INDEX
from dexbotic.data.dataset.dex_navila_dataset import DexNavilaDataset
from dexbotic.data.dataset.transform.common import (
    Pipeline,
    ToDict,
    ToList,
    ToNumpy,
    ToTensor,
)
from dexbotic.data.dataset.transform.multimodal import LoadMultiModal
from dexbotic.exp.base_exp import (
    ActionConfig,
    BaseExp,
    DataConfig,
    InferenceConfig,
    ModelConfig,
    OptimizerConfig,
    TokenizerConfig,
    TrainerConfig,
)
from dexbotic.exp.navila_trainer import DexboticNaVILATrainer
from dexbotic.model.navila.navila_arch import NaVILAForCausalLM
from dexbotic.tokenization import conversation as conversation_lib
from dexbotic.tokenization.conversation import KeywordsStoppingCriteria
from dexbotic.tokenization.process import NaVILATokenization
from dexbotic.tokenization.tokenization import tokenizer_image_token


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        choices=["train", "inference"],
    )
    args, unknown = parser.parse_known_args()
    return args


@dataclass
class NaVILAOptimizerConfig(OptimizerConfig):
    base_lr: float = field(default=2e-5)


@dataclass
class NaVILATrainerConfig(TrainerConfig):
    output_dir: str = field(
        default=f"./checkpoints/navila-sft-{datetime.now().strftime('%m%d')}"
    )

    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=10)
    gradient_accumulation_steps: int = field(default=2)

    save_strategy: str = field(default="steps")
    save_steps: int = field(default=10000)
    save_total_limit: int = field(default=1)
    logging_steps: int = field(default=10)
    gradient_checkpointing: bool = field(default=True)
    dataloader_num_workers: int = field(default=16)
    model_max_length: int = field(default=4096)
    tune_mm_mlp_adapter: bool = field(default=False)


@dataclass
class NaVILAActionConfig(ActionConfig):
    # pass
    def build_action_process_func(self) -> Pipeline:
        action_config = Pipeline(
            [
                ToDict(),
                ToNumpy(),
                LoadMultiModal(),
                ToList(),
            ]
        )

        return action_config


@dataclass
class NaVILADataConfig(DataConfig):
    dataset_name: str = field(default="navila_sft")
    num_images: int = field(default=8)
    action_config: ActionConfig = field(default_factory=NaVILAActionConfig)

    def _build_dataset(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        chat_template: str,
        image_processor: BaseImageProcessor,
    ) -> DexNavilaDataset:
        # FIXME: DO NOT USE EASYDICT IN NEXT VERSION
        data_args = EasyDict(
            {
                "dataset_name": self.dataset_name,
                "num_images": self.num_images,
                "data_keys": self.data_keys,
                "images_keys": self.images_keys,
                "aug_policy": self.aug_policy,
                "image_aspect_ratio": self.image_aspect_ratio,
                "image_processor": image_processor,
                "chat_template": chat_template,
                "image_pad_mode": self.image_pad_mode,
            }
        )
        action_process_func = self.action_config.build_action_process_func()
        tokenization_func = NaVILATokenization(tokenizer, data_args)
        dataset = DexNavilaDataset(
            data_args=data_args,
            tokenization_func=tokenization_func,
            action_process_func=action_process_func,
        )
        return dataset


@dataclass
class NaVILAModelConfig(ModelConfig):
    model_name_or_path: str = field(default="./checkpoints/dex_navila")
    mm_projector_type: str = field(default="mlp_downsample")
    mm_vision_tower: str = field(default="google/siglip-so400m-patch14-384")
    chat_template: str = field(default="llama_3")
    freeze_llm: bool = field(default=False)
    freeze_mm_projector: bool = field(default=False)
    freeze_mm_vision: bool = field(default=False)
    freeze_lm_head: bool = field(default=False)

    def build_model(self) -> NaVILAForCausalLM:
        model_config_args = {
            "model_name_or_path": self.model_name_or_path,
            "chat_template": self.chat_template,
            "mm_projector_type": self.mm_projector_type,
            "mm_vision_tower": self.mm_vision_tower,
        }
        model = NaVILAForCausalLM.from_pretrained(self.model_name_or_path)
        model.model.initialize_model(model_config_args)

        self._freeze_model(model)

        return model

    def _freeze_model(self, model: NaVILAForCausalLM):
        # set requires_grad to True for all parameters
        for param in model.model.parameters():
            param.requires_grad = True

        if self.freeze_llm:
            for param in model.model.backbone.parameters():
                param.requires_grad = False
        if self.freeze_mm_projector:
            for param in model.model.mm_projector_module.parameters():
                param.requires_grad = False
        if self.freeze_mm_vision:
            for param in model.model.mm_vision_module.parameters():
                param.requires_grad = False
        if self.freeze_lm_head:
            for param in model.lm_head.parameters():
                param.requires_grad = False


@dataclass
class NaVILATokenizerConfig(TokenizerConfig):
    use_fast_tokenizer: bool = field(default=False)


@dataclass
class NaVILAInferenceConfig(InferenceConfig):
    model_name_or_path: Optional[str] = field(default=None)
    port: int = field(default=7891)
    save_image: bool = field(default=False)
    save_image_dir: str = field(default="./debug_data")
    num_images: int = field(default=8)
    chat_template: str = field(default="llama_3")

    def _load_model(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model from {self.model_name_or_path}")
        logger.info(f"Using device: {self.device}")
        model = NaVILAForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map={"": "cuda:0"},
        ).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model.config
        if (
            not hasattr(self.model_config, "chat_template")
            or self.model_config.chat_template is None
        ):
            self.model_config.chat_template = self.chat_template

        logger.info("Model loaded successfully")

        self.input_transform = Pipeline(
            [
                ToTensor(),
            ]
        )
        self.output_transform = Pipeline(
            [
                ToNumpy(),
            ]
        )

    def _initialize_inference(self) -> None:
        super()._initialize_inference()
        self.history_buffer = deque()
        if self.save_image:
            self.save_dir = os.path.join(
                self.save_image_dir,
                self.policy_name,
                datetime.now().strftime("%Y%m%d_%H%M%S"),
            )
            os.makedirs(self.save_dir, exist_ok=True)

    def sample_and_pad_images(self, width=512, height=512):
        frames = copy.deepcopy(self.history_buffer)
        if len(frames) < self.num_images:
            while len(frames) < self.num_images:
                padding_img = Image.new("RGB", (width, height), color=(0, 0, 0))
                padding_stream = io.BytesIO()
                padding_img.save(padding_stream, format="JPEG")
                padding_stream.seek(0)
                frames.insert(0, padding_stream)

        latest_frame = frames[-1]
        sampled_indices = np.linspace(
            0, len(frames) - 1, num=self.num_images - 1, endpoint=False, dtype=int
        )
        sampled_frames = [frames[i] for i in sampled_indices] + [latest_frame]
        return sampled_frames

    def _prepare_images(self, image_bytes: bytes | None) -> list:
        reset_memory = getattr(self, "meta_data", {}).get("reset_memory", False)
        run_model = getattr(self, "meta_data", {}).get("run_model", True)

        if reset_memory:
            self.history_buffer.clear()

        if image_bytes is not None:
            self.history_buffer.append(io.BytesIO(image_bytes))

        if not run_model:  # only update history buffer, not run model
            return []

        return self.sample_and_pad_images()

    def _parse_bool(self, value: str | None, default: bool = False) -> bool:
        if value is None:
            return default
        try:
            return bool(strtobool(str(value)))
        except Exception:
            return default

    def process_frame(self) -> None:
        text = request.form.get("text", "")
        episode_first_frame = request.form.get("episode_first_frame")
        run_model = self._parse_bool(request.form.get("run_model"), default=True)

        reset_memory = self._parse_bool(episode_first_frame, default=False)
        self.meta_data = {"reset_memory": reset_memory, "run_model": run_model}

        images = request.files.getlist("image")
        image_bytes = images[0].read() if images else None

        images_list = self._prepare_images(image_bytes)
        for stream in images_list:
            stream.seek(0)

        results = []
        if images_list:
            results = self._get_response(
                text=text,
                images=images_list,
                reset_memory=reset_memory,
            )
        return jsonify({"response": results})

    def _get_response(
        self, text: str, images: List[str], max_new_tokens: int = 32, **kwargs
    ) -> str:
        # Check if images are provided
        assert (
            len(images) == self.num_images
        ), f"Expected {self.num_images} images, but got {len(images)}"

        images = [
            Image.fromarray(np.array(Image.open(i).convert("RGB"))[..., ::-1])
            for i in images
        ]

        image_tensor = (
            self.model.process_images(images).to(dtype=self.model.dtype).unsqueeze(0)
        )

        self._save_image(images, text, **kwargs)

        # Build conversation prompt
        interleaved_images = "<image>\n" * (len(images) - 1)

        question = (
            f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
            f'of historical observations {interleaved_images}, and current observation <image>\n. Your assigned task is: "{text}" '
            f"Analyze this series of images to decide your next action, which could be turning left or right by a specific "
            f"degree, moving forward a certain distance, or stop if the task is completed."
        )

        conv = conversation_lib.conv_templates[self.model_config.chat_template].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.model.device)
        )

        stop_str = (
            conv.sep
            if conv.sep_style != conversation_lib.SeparatorStyle.TWO
            else conv.sep2
        )
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        with torch.inference_mode():
            generate_output = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            generated_ids = generate_output.sequences[0, input_ids.shape[1] :]
        outputs = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        outputs = outputs.strip()

        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()

        # Define the regex patterns for each action
        patterns = {
            0: re.compile(r"\bstop\b", re.IGNORECASE),
            1: re.compile(r"\bis move forward\b", re.IGNORECASE),
            2: re.compile(r"\bis turn left\b", re.IGNORECASE),
            3: re.compile(r"\bis turn right\b", re.IGNORECASE),
        }

        # Function to map a string to an action integer
        def map_string_to_action(s):
            for action, pattern in patterns.items():
                if pattern.search(s):
                    return action
            return None  # Return None if no match is found

        try:
            actions = [map_string_to_action(outputs)]
        except Exception:
            actions = [1]

        queue_actions = []

        if actions[0] == 1:
            try:
                match = re.search(r"move forward (\d+) cm", outputs)
                distance = int(match.group(1))
            except Exception:
                distance = 25
            if (distance % 25) != 0:
                distance = min([25, 50, 75], key=lambda x: abs(x - distance))
            for _ in range(int(distance // 25)):
                queue_actions.append(1)

        elif actions[0] == 2:
            try:
                match = re.search(r"turn left (\d+) degree", outputs)
                degree = int(match.group(1))
            except Exception:
                degree = 15
            if (degree % 15) != 0:
                degree = min([15, 30, 45], key=lambda x: abs(x - degree))
            for _ in range(int(degree // 15)):
                queue_actions.append(2)

        elif actions[0] == 3:
            try:
                match = re.search(r"turn right (\d+) degree", outputs)
                degree = int(match.group(1))
            except Exception:
                degree = 15
            if (degree % 15) != 0:
                degree = min([15, 30, 45], key=lambda x: abs(x - degree))
            for _ in range(int(degree // 15)):
                queue_actions.append(3)

        else:
            queue_actions.append(0)
        print(f"outputs: {outputs}, queue_actions: {queue_actions}")
        return queue_actions

    def _save_image(
        self,
        images: Union[List[str], List[Image.Image]],
        text: str,
        reset_memory: bool = False,
        **kwargs,
    ) -> None:
        if not self.save_image:
            return
        if reset_memory:
            self.episode += 1
            self.timestep = 0
            self.prev_prompt = text
        else:
            self.timestep += 1

        save_image_dir_episode = os.path.join(self.save_dir, str(self.episode))
        os.makedirs(save_image_dir_episode, exist_ok=True)

        if len(images) > 0:
            cell_w, cell_h = images[0].size
            n = len(images)
            cols = math.ceil(math.sqrt(n))
            rows = math.ceil(n / cols)
            gap = 4
            canvas_w = cols * cell_w + (cols - 1) * gap
            canvas_h = rows * cell_h + (rows - 1) * gap
            concat_img = Image.new(
                images[0].mode, (canvas_w, canvas_h), color=(255, 255, 255)
            )
            draw = ImageDraw.Draw(concat_img)

            for c in range(1, cols):
                x = c * cell_w + (c - 1) * gap - 1
                draw.rectangle([x, 0, x + gap, canvas_h], fill=0)
            for r in range(1, rows):
                y = r * cell_h + (r - 1) * gap - 1
                draw.rectangle([0, y, canvas_w, y + gap], fill=0)

            idx = 0
            for r in range(rows):
                for c in range(cols):
                    x = c * (cell_w + gap)
                    y = r * (cell_h + gap)
                    if idx < n:
                        img = images[idx]
                        if img.size != (cell_w, cell_h):
                            img = img.resize((cell_w, cell_h))
                        concat_img.paste(img, (x, y))
                        idx += 1

            concat_image_path = os.path.join(
                save_image_dir_episode, f"{self.timestep}.jpg"
            )
            concat_img.save(concat_image_path)
            logger.info(f"Saved concat image to {concat_image_path}")


@dataclass
class NaVILAExp(BaseExp):
    model_config: NaVILAModelConfig = field(default_factory=NaVILAModelConfig)
    optimizer_config: NaVILAOptimizerConfig = field(
        default_factory=NaVILAOptimizerConfig
    )
    trainer_config: NaVILATrainerConfig = field(default_factory=NaVILATrainerConfig)
    data_config: NaVILADataConfig = field(default_factory=NaVILADataConfig)
    tokenizer_config: NaVILATokenizerConfig = field(
        default_factory=NaVILATokenizerConfig
    )
    inference_config: NaVILAInferenceConfig = field(
        default_factory=NaVILAInferenceConfig
    )
    logger_level: str = field(default="INFO")

    def _initialize_train(self):
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        logger.info(f"Local rank: {self.local_rank}")
        if self.local_rank != 0:
            logger.remove()
            logger.add(lambda msg: None)

        tokenizer_kwargs = {
            "model_max_length": self.trainer_config.model_max_length,
            "padding_side": "right",
            "use_fast": self.tokenizer_config.use_fast_tokenizer,
        }
        tokenizer = self.tokenizer_config.build_tokenizer(
            self.model_config.model_name_or_path, **tokenizer_kwargs
        )
        self.tokenizer = tokenizer

        model = self.model_config.build_model()
        self.model = model
        self.tokenizer = self.tokenizer_config.add_special_tokens(
            self.data_config.action_config.string_format,
            self.data_config.action_config.vocab_size,
            self.tokenizer,
            self.model,
        )
        self.model.config.use_cache = False

        train_dataset, data_collator = self.data_config.build_data(
            self.tokenizer,
            self.model_config.chat_template,
            self.model.model.mm_vision_module.image_processor,
        )

        trainer_kwargs = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "exp_config": self,
            "train_dataset": train_dataset,
            "data_collator": data_collator,
        }
        trainer = DexboticNaVILATrainer(**trainer_kwargs)
        self.trainer = trainer


if __name__ == "__main__":
    args = parse_args()
    exp = NaVILAExp()

    if args.task == "train":
        exp.train()
    elif args.task == "inference":
        exp.inference()
