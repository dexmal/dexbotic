import argparse
import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import megfile
import torch
import json
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image
from loguru import logger
import transformers
from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoImageProcessor, BaseImageProcessor

import dexbotic.data.utils.normalize as normalize
from dexbotic.data.dataset.dex_dataset import DexDataset
from dexbotic.data.dataset.rgb_preprocess import DummyRGBProcessor
from dexbotic.data.dataset.tokenization import DummyTokenization

from dexbotic.data.dataset.transform.action import (
    ActionNorm,
    AddAction,
    AddTrajectory,
    DeltaAction,
    PadAction,
    PadState,
)
from dexbotic.data.dataset.transform.common import (
    Pipeline,
    ToDict,
    ToList,
    ToNumpy,
    ToTensor,
)
from dexbotic.data.dataset.transform.multimodal import LoadMultiModal
from dexbotic.data.dataset.transform.output import AbsoluteAction, ActionDenorm
from dexbotic.exp.base_exp import (
    ActionConfig,
    BaseExp,
    ComputeNormActionConfig,
    Config,
    DataConfig,
    ModelConfig,
    OPENAI_CLIP_PATH,
    OptimizerConfig,
    TokenizerConfig,
    TrainerConfig,
)
from dexbotic.model.dm0.dm0_arch import DM0ForCausalLM
from dexbotic.tokenization.process import DM0Tokenization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        choices=["train", "inference", "compute_norm_stats"],
    )
    args, unknown = parser.parse_known_args()
    return args


@dataclass
class DM0TokenizerConfig(TokenizerConfig):
    use_fast_tokenizer: bool = field(default=False)


@dataclass
class DM0ComputeNormActionConfig(ComputeNormActionConfig):
    def compute_norm_stats(self, dataset_name: str) -> None:
        self.norm_save_path = os.path.join(
            os.path.dirname(self.norm_save_path),
            hashlib.md5(dataset_name.encode()).hexdigest()[:8],
        )
        dataset_name_list = dataset_name.split("+")
        action_process_func = self.build_action_process_func()
        dataset_list = self._get_dataset(action_process_func, dataset_name_list)
        norm_files = {}

        for dataset_name, dataset in dataset_list:
            norm_file = self._process_one_dataset(dataset_name, dataset)
            norm_files[dataset_name] = (norm_file, dataset.dataset_map[0])

        self._merge_norm_stats(norm_files)

    def build_action_process_func(self) -> Pipeline:
        action_config = Pipeline(
            [
                ToDict(),
                ToNumpy(),
                AddAction(predict_length=1),
                PadState(ndim=32, axis=-1),
                PadAction(ndim=32, axis=-1),
                AddTrajectory(trajectory_length=50, flatten=False, padding_mode="last"),
                DeltaAction(enable=True),
                ToList(),
            ]
        )

        return action_config

    def _get_dataset(
        self, action_process_func: Pipeline, dataset_name_list: list[str]
    ) -> list[tuple[str, DexDataset]]:
        robot_dataset_list = []
        for dataset_name in dataset_name_list:
            robot_dataset = DexDataset(
                data_args=EasyDict(
                    dataset_name=dataset_name,
                    num_images=1,
                    data_keys=["action", "state"],
                    image_processor=AutoImageProcessor.from_pretrained(
                        OPENAI_CLIP_PATH
                    ),
                    image_aspect_ratio=None,
                    aug_policy=None,
                ),
                tokenization_func=DummyTokenization(),
                action_process_func=action_process_func,
                image_process_func=DummyRGBProcessor(),
            )
            robot_dataset_list.append((dataset_name, robot_dataset))
        return robot_dataset_list

    def _process_one_dataset(self, dataset_name: str, dataset: DexDataset) -> str:
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=64)

        norm_keys = ["state", "action"]
        stats = {key: normalize.RunningStats() for key in norm_keys}
        for batch_idx, batch in tqdm(
            enumerate(dataloader), desc="Computing norm stats"
        ):
            if batch_idx > 2500:
                break
            for key in norm_keys:
                values = batch[key].numpy()
                stats[key].update(values.reshape(-1, values.shape[-1]))
        norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

        save_path = os.path.join(self.norm_save_path, dataset_name)
        logger.info(f"Saving norm stats to {save_path}")
        normalize.save(save_path, norm_stats)

        return os.path.join(save_path, "norm_stats.json")

    def _merge_norm_stats(
        self,
        norm_files: dict[str, tuple[str, str]],
        per_task_norm: bool = False,
        norm_keys: list[str] = ["action", "state"],
    ) -> None:
        norm_stats = {
            "default": {"min": -1, "max": 1},
        }
        for norm_key in norm_keys:
            min_list = []
            max_list = []
            mean_list = []
            std_list = []
            for dataset_name, (norm_file, dataset_path) in norm_files.items():
                with open(norm_file, "r") as f:
                    stats = json.load(f)["norm_stats"][norm_key]
                if per_task_norm:
                    norm_stats[dataset_path] = {
                        "default": {
                            "min": stats["q01"],
                            "max": stats["q99"],
                            "mean": stats["mean"],
                            "std": stats["std"],
                        }
                    }
                min_list.append(stats["q01"])
                max_list.append(stats["q99"])
                mean_list.append(stats["mean"])
                std_list.append(stats["std"])
            min_list = np.array(min_list).min(axis=0).tolist()
            max_list = np.array(max_list).max(axis=0).tolist()
            mean_list = np.array(mean_list).mean(axis=0).tolist()
            std_list = np.array(std_list).mean(axis=0).tolist()
            norm_stats[norm_key] = {
                "min": min_list,
                "max": max_list,
                "mean": mean_list,
                "std": std_list,
            }

        with open(os.path.join(self.norm_save_path, "norm_stats.json"), "w") as f:
            json.dump({"norm_stats": norm_stats}, f, indent=2)


@dataclass
class DM0ModelConfig(ModelConfig):
    model_name_or_path: str = field(default="./checkpoints/DM0-base")

    def build_model(self) -> DM0ForCausalLM:
        model = DM0ForCausalLM.from_pretrained(self.model_name_or_path)
        return model


@dataclass
class DM0OptimizerConfig(OptimizerConfig):
    base_lr: float = field(default=2.5e-5)
    adam_beta2: float = field(default=0.95)
    warmup_steps: int = field(default=1000)
    weight_decay: float = field(default=1e-10)

    def _get_optimizer_grouped_parameters(self, model: torch.nn.Module) -> list:
        """Returns a list of dictionaries containing parameters grouped by their names
        and whether they require weight decay.
        """
        parameters = [
            {
                "params": [p for n, p in model.named_parameters() if p.requires_grad],
                "weight_decay": self.weight_decay,
            }
        ]
        return parameters


@dataclass
class DM0TrainerConfig(TrainerConfig):
    model_max_length: int = field(default=200)
    bf16: bool = field(default=True)
    num_train_steps: int = field(default=30000)
    save_steps: int = field(default=10000)
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=1)
    gradient_checkpointing: bool = field(default=True)
    dataloader_num_workers: int = field(default=16)
    logging_steps: int = field(default=1)
    lr_scheduler_type: str = field(default="cosine_with_min_lr")
    lr_scheduler_kwargs: dict = field(default_factory=lambda: {"min_lr_rate": 0.1})


@dataclass
class DM0ActionConfig(ActionConfig):
    trajectory_length: int = field(default=50)

    def build_action_process_func(self) -> Pipeline:
        statistic_mapping = self._read_norm_stats(self.statistic_mapping)
        action_config = Pipeline(
            [
                ToDict(),
                ToNumpy(),
                AddAction(predict_length=1),
                PadState(ndim=32, axis=-1),
                PadAction(ndim=32, axis=-1),
                AddTrajectory(trajectory_length=50, flatten=False, padding_mode="last"),
                DeltaAction(enable=True),
                ActionNorm(statistic_mapping=statistic_mapping, use_quantiles=True),
                LoadMultiModal(return_masks=True),
                ToList(),
            ]
        )

        return action_config


@dataclass
class DM0DataConfig(DataConfig):
    num_images: int = field(default=3)
    data_keys: list[str] = field(
        default_factory=lambda: [
            "input_ids",
            "labels",
            "action",
            "image",
            "state",
            "image_masks",
        ]
    )
    aug_policy: str | list[str] = field(
        default_factory=lambda: ["dm0", "dm0_color", "dm0_color"]
    )
    action_config: DM0ActionConfig = field(default_factory=DM0ActionConfig)
    image_pad_mode: str = field(default="zero")

    def _build_dataset(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        chat_template: str,
        image_processor: BaseImageProcessor,
    ) -> DexDataset:
        # FIXME: DO NOT USE EASYDICT IN NEXT VERSION
        action_process_func = self.action_config.build_action_process_func()
        tokenization_func = DM0Tokenization(tokenizer, chat_template="step")

        data_args = EasyDict({
            "dataset_name": self.dataset_name,
            "num_images": self.num_images,
            "data_keys": self.data_keys,
            "images_keys": self.images_keys,
            "aug_policy": self.aug_policy,
            "image_aspect_ratio": self.image_aspect_ratio,
            "image_processor": image_processor,
            "chat_template": chat_template,
            "image_pad_mode": self.image_pad_mode,
        })
        dataset = DexDataset(
            data_args=data_args,
            tokenization_func=tokenization_func,
            action_process_func=action_process_func,
        )
        return dataset


@dataclass
class DM0InferenceConfig(Config):
    model_name_or_path: Optional[str] = field(default=None)
    port: int = field(default=7891)
    save_image: bool = field(default=False)
    save_image_dir: str = field(default="./debug_data")
    norm_stats: Optional[dict] = field(default=None)
    num_images: int = field(default=3)
    non_delta_mask: list[int] = field(default_factory=lambda: [6])
    action_dim: int = field(default=7)

    def _load_model(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model from {self.model_name_or_path}")
        logger.info(f"Using device: {self.device}")
        model = DM0ForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
        ).to(self.device)

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, use_fast=False, trust_remote_code=True
        )
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model.config
        self.tokenization_func = DM0Tokenization(self.tokenizer)
        logger.info("Model loaded successfully")

        self.input_transform = Pipeline(
            [
                PadState(ndim=self.model.model.config.action_dim, axis=-1),
                ActionNorm(
                    statistic_mapping=self.norm_stats, strict=False, use_quantiles=True
                ),
                ToTensor(),
            ]
        )
        self.output_transform = Pipeline(
            [
                ToNumpy(),
                ActionDenorm(
                    statistic_mapping=self.norm_stats, strict=False, use_quantiles=True
                ),
                AbsoluteAction(),
            ]
        )

    def run(self) -> None:
        self._initialize_inference()
        self.app = Flask(__name__)
        self.app.add_url_rule(
            "/process_frame", "process_frame", self.process_frame, methods=["POST"]
        )
        self.app.run(host="0.0.0.0", port=self.port, debug=False, threaded=False)

    def _initialize_inference(self) -> None:
        if self.norm_stats is None:
            norm_stats_file = os.path.join(self.model_name_or_path, "norm_stats.json")
            self.norm_stats = self.read_normalization_stats(norm_stats_file)
        logger.info(f"Normalization stats: {self.norm_stats}")

        self._load_model()
        self.prev_text = None
        self.timestep = 0
        self.episode = 0

    def read_normalization_stats(self, action_norm_file: str | None) -> dict:
        logger.info(f"Reading normalization stats from {action_norm_file}")
        if action_norm_file is None or not megfile.smart_exists(action_norm_file):
            return {"min": -1, "max": 1}
        with megfile.smart_open(action_norm_file, "r") as f:
            norm_stats = json.load(f)
            if "norm_stats" in norm_stats:
                norm_stats = norm_stats["norm_stats"]
        return ToNumpy()(norm_stats)

    def process_frame(self) -> None:
        results = self._get_response(
            text=request.form.get("text", ""),
            images=request.files.getlist("image", None),
            states=request.form.get("states", None),
            batch_size=request.form.get("batch_size", 1),
        )
        return jsonify({"response": results})

    def _get_response(
        self,
        text: str | list[str],
        images: list[str],
        states: Optional[str | list[str]] = None,
        batch_size: int = 1,
    ) -> list[list[float]]:
        t0 = time.monotonic()
        batch_size = int(batch_size)
        assert (
            len(images) % batch_size == 0
        ), f"Number of images {len(images)} is not divisible by batch size {batch_size}"
        num_images = len(images) // batch_size
        images = [
            images[i * num_images : (i + 1) * num_images] for i in range(batch_size)
        ]
        if isinstance(text, str):
            text = [text] * batch_size

        batch_images = [
            [Image.open(i).convert("RGB") for i in image_items]
            for image_items in images
        ]
        batch_images_tensor = [
            self.model.process_images(image_items).to(dtype=self.model.dtype)
            for image_items in batch_images
        ]

        if num_images != self.num_images:
            batch_images_tensor = [
                torch.cat(
                    [
                        image_tensor,
                        torch.zeros_like(image_tensor[0:1]).repeat(
                            self.num_images - num_images, 1, 1, 1
                        ),
                    ],
                    dim=0,
                )
                if len(image_tensor) < self.num_images
                else image_tensor[: self.num_images]
                for image_tensor in batch_images_tensor
            ]

        batch_image_masks = [
            torch.tensor(
                [True for _ in range(num_images)]
                + [False for _ in range(self.num_images - num_images)],
                device=image_tensor.device,
            )
            for image_tensor in batch_images_tensor
        ]
        batch_images_tensor = torch.stack(batch_images_tensor, dim=0)
        batch_image_masks = torch.stack(batch_image_masks, dim=0)

        self._save_image(images[0], text[0])

        batch_input_ids = np.array(
            [self.tokenization_func([{"from": "human", "value": p}])["input_ids"] for p in text]
        )
        logger.info(f"prompt: {self.tokenization_func.tokenizer.decode(batch_input_ids[0])}")
        batch_attention_mask = np.array(
            [np.array(ids != self.tokenizer.pad_token_id) for ids in batch_input_ids]
        )

        if states is not None:   
            if isinstance(states, str):
                batch_states = np.array(json.loads(states))
                if batch_states.ndim == 1:
                    batch_states = batch_states[None]
                assert batch_states.shape[0] == batch_size, (
                    f"Batch inference requires states to be a list with length {batch_size}, "
                    f"but got length {len(batch_states)}."
                )
            elif isinstance(states, (list, tuple)) and all(
                isinstance(s, str) for s in states
            ):
                assert len(states) == batch_size, (
                    f"Batch inference requires states to be a list with length {batch_size}, "
                    f"but got {type(states)} with length {len(states)}."
                )
                batch_states = [json.loads(s) for s in states]
                batch_states = np.array(batch_states)
        else:
            batch_states = np.zeros(
                (
                    batch_size,
                    self.model.model.config.action_dim,
                ),
                dtype=np.float32,
            )

        inference_args = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "images": batch_images_tensor,
            "image_masks": batch_image_masks,
            "state": batch_states,
            "meta_data": {
                "non_delta_mask": np.array(self.non_delta_mask),
            },
        }

        inputs = self.input_transform(inference_args)
        inputs["states"] = inputs["state"]
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        actions = self.model.inference_action(**inputs)
        outputs = {
            k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        outputs["action"] = actions.detach().cpu().numpy()
        outputs = self.output_transform(outputs)
        logger.info(f"Processing time: {time.monotonic() - t0}")
        return outputs["action"][..., : self.action_dim].tolist()

    def _save_image(self, images: list[Image.Image], text: str) -> None:
        if not self.save_image:
            return
        if text == self.prev_text:
            self.timestep += 1
        else:
            self.timestep = 0
            self.prev_text = text
            self.episode += 1
        save_image_dir_episode = os.path.join(self.save_image_dir, str(self.episode))
        os.makedirs(save_image_dir_episode, exist_ok=True)
        for idx, image in enumerate(images):
            image.save(
                os.path.join(save_image_dir_episode, f"{self.timestep}_{idx}.png")
            )
        if self.timestep == 0:
            with open(os.path.join(save_image_dir_episode, "text.txt"), "w") as f:
                f.write(text)


@dataclass
class DM0Exp(BaseExp):
    model_config: DM0ModelConfig = field(default_factory=DM0ModelConfig)
    optimizer_config: DM0OptimizerConfig = field(default_factory=DM0OptimizerConfig)
    trainer_config: DM0TrainerConfig = field(default_factory=DM0TrainerConfig)
    data_config: DM0DataConfig = field(default_factory=DM0DataConfig)
    tokenizer_config: DM0TokenizerConfig = field(default_factory=DM0TokenizerConfig)
    inference_config: DM0InferenceConfig = field(default_factory=DM0InferenceConfig)

    def inference(self) -> None:
        self.inference_config.run()

    def compute_norm_stats(self) -> None:
        self.data_config.action_config = DM0ComputeNormActionConfig()
        self.data_config.action_config.compute_norm_stats(self.data_config.dataset_name)

    def _auto_compute_norm_stats(self) -> None:
        if (
            not self.data_config.auto_norm
            or self.data_config.action_config.statistic_mapping is not None
        ):
            return
        if self.local_rank == 0:
            print(
                f"Action config before auto compute norm: {self.data_config.action_config}"
            )
        _action_config = self.data_config.action_config
        norm_config = DM0ComputeNormActionConfig()
        save_name = hashlib.md5(self.data_config.dataset_name.encode()).hexdigest()[:8]
        norm_config.norm_save_path = os.path.join(
            os.path.dirname(norm_config.norm_save_path), save_name
        )
        norm_file_path = os.path.join(norm_config.norm_save_path, "norm_stats.json")
        if self.local_rank == 0 and not megfile.smart_exists(norm_file_path):
            logger.info("Auto-computing norm stats on rank0")
            self.compute_norm_stats()
        else:
            while not megfile.smart_exists(norm_file_path):
                time.sleep(5)
                print(
                    f"Waiting for norm stats: {norm_file_path} to be computed on rank{self.local_rank}"
                )
        _action_config.statistic_mapping = norm_file_path
        self.data_config.action_config = _action_config
        if self.local_rank == 0:
            print(
                f"Action config after auto compute norm: {self.data_config.action_config}"
            )


if __name__ == "__main__":
    args = parse_args()
    exp = DM0Exp()
    if args.task == "train":
        exp.train()
    elif args.task == "inference":
        exp.inference()
    elif args.task == "compute_norm_stats":
        exp.compute_norm_stats()
