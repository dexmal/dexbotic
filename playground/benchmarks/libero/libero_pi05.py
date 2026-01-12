import argparse
import hashlib
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import megfile
import torch
from loguru import logger
from transformers import AutoTokenizer

from dexbotic.data.dataset.transform.action import (
    ActionNorm,
    AddTrajectory,
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
from dexbotic.data.dataset.transform.output import ActionDenorm
from dexbotic.exp.pi0_exp import Pi0ActionConfig as _Pi0ActionConfig
from dexbotic.exp.pi05_exp import (
    Pi0ComputeNormActionConfig as _Pi0ComputeNormActionConfig,
)
from dexbotic.exp.pi05_exp import Pi0DataConfig as _Pi0DataConfig
from dexbotic.exp.pi05_exp import Pi0InferenceConfig as _Pi0InferenceConfig
from dexbotic.exp.pi05_exp import Pi0OptimizerConfig as _Pi0OptimizerConfig
from dexbotic.exp.pi05_exp import Pi0TokenizerConfig as _Pi0TokenizerConfig
from dexbotic.exp.pi05_exp import Pi0TrainerConfig as _Pi0TrainerConfig
from dexbotic.exp.pi05_exp import Pi05Exp as _Pi05Exp
from dexbotic.exp.pi05_exp import Pi05ModelConfig as _Pi05ModelConfig
from dexbotic.model.pi05.pi05_arch import Pi05ForCausalLM
from dexbotic.tokenization.process import Pi0Tokenization


def parse_args():
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
class Pi05OptimizerConfig(_Pi0OptimizerConfig):
    base_lr: float = field(default=5e-5)
    adam_beta2: float = field(default=0.95)
    warmup_steps: int = field(default=10_000)
    weight_decay: float = field(default=1e-10)


@dataclass
class Pi05TrainerConfig(_Pi0TrainerConfig):
    wandb_project: str = field(default="dexbotic-pi05-libero-all")
    bf16: bool = field(default=True)
    num_train_steps: int = field(default=30000)
    save_steps: int = field(default=5000)
    save_total_limit: int = field(default=10)
    per_device_train_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=1)
    model_max_length: int = field(default=200)
    output_dir: str = field(
        default=f"./user_checkpoints/dexbotic/libero_all_pi05/all-{datetime.now().strftime('%m%d')}"
    )
    lr_scheduler_type: str = field(default="cosine_with_min_lr")
    lr_scheduler_kwargs: dict = field(
        default_factory=lambda: {"min_lr": 5e-5}
    )  # 5e-5 -> 5e-5


class Pi05ComputeNormActionConfig(_Pi0ComputeNormActionConfig):
    def build_action_process_func(self) -> Pipeline:
        action_config = Pipeline(
            [
                ToDict(),
                ToNumpy(),
                PadState(ndim=32, axis=-1),
                PadAction(ndim=32, axis=-1),
                AddTrajectory(trajectory_length=10, flatten=False, padding_mode="last"),
                ToList(),
            ]
        )

        return action_config


@dataclass
class Pi05ActionConfig(_Pi0ActionConfig):
    statistic_mapping: str = field(default=None)
    trajectory_length: int = field(default=10)

    def build_action_process_func(self) -> Pipeline:
        statistic_mapping = self._read_norm_stats(self.statistic_mapping)
        action_config = Pipeline(
            [
                ToDict(),
                ToNumpy(),
                PadState(ndim=32, axis=-1),
                PadAction(ndim=32, axis=-1),
                AddTrajectory(trajectory_length=10, flatten=False, padding_mode="last"),
                ActionNorm(statistic_mapping=statistic_mapping),
                LoadMultiModal(return_masks=True),
                ToList(),
            ]
        )
        return action_config


@dataclass
class Pi05DataConfig(_Pi0DataConfig):
    dataset_name: str = field(default="libero_pi0_all")
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
        default_factory=lambda: ["pi0", "color", "identity"]
    )
    action_config: Pi05ActionConfig = field(default_factory=Pi05ActionConfig)


@dataclass
class Pi05ModelConfig(_Pi05ModelConfig):
    model_name_or_path: str = field(default="./checkpoints/Dexbotic-PI05")

    def build_model(self) -> Pi05ForCausalLM:
        model = Pi05ForCausalLM.from_pretrained(self.model_name_or_path)
        model.model.config.chunk_size = 10
        return model


@dataclass
class Pi05TokenizerConfig(_Pi0TokenizerConfig):
    use_fast_tokenizer: bool = field(default=False)


@dataclass
class Pi05InferenceConfig(_Pi0InferenceConfig):
    model_name_or_path: Optional[str] = field(
        default="./checkpoints/libero/libero_pi05"
    )
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
        model = Pi05ForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
        ).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, use_fast=False
        )
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model.config
        self.tokenization_func = Pi0Tokenization(self.tokenizer)
        logger.info("Model loaded successfully")

        self.input_transform = Pipeline(
            [
                PadState(ndim=self.model.model.config.action_dim, axis=-1),
                ActionNorm(statistic_mapping=self.norm_stats, strict=False),
                ToTensor(),
            ]
        )
        self.output_transform = Pipeline(
            [
                ToNumpy(),
                ActionDenorm(statistic_mapping=self.norm_stats, strict=False),
            ]
        )


@dataclass
class Pi05Exp(_Pi05Exp):
    model_config: Pi05ModelConfig = field(default_factory=Pi05ModelConfig)
    optimizer_config: Pi05OptimizerConfig = field(default_factory=Pi05OptimizerConfig)
    trainer_config: Pi05TrainerConfig = field(default_factory=Pi05TrainerConfig)
    data_config: Pi05DataConfig = field(default_factory=Pi05DataConfig)
    tokenizer_config: Pi05TokenizerConfig = field(default_factory=Pi05TokenizerConfig)
    inference_config: Pi05InferenceConfig = field(default_factory=Pi05InferenceConfig)

    def inference(self) -> None:
        self.inference_config.run()

    def compute_norm_stats(self) -> None:
        self.data_config.action_config = Pi05ComputeNormActionConfig()
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
        norm_config = Pi05ComputeNormActionConfig()
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
    exp = Pi05Exp()
    if args.task == "train":
        exp.train()
    elif args.task == "inference":
        exp.inference()
    elif args.task == "compute_norm_stats":
        exp.compute_norm_stats()
