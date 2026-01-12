import argparse
import hashlib
import os
import time
from dataclasses import dataclass, field

import megfile
import torch
from loguru import logger
from transformers import AutoTokenizer

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
from dexbotic.exp.base_exp import ActionConfig, BaseExp, ModelConfig
from dexbotic.exp.pi0_exp import (
    Pi0ComputeNormActionConfig,
    Pi0DataConfig,
    Pi0InferenceConfig,
    Pi0OptimizerConfig,
    Pi0TokenizerConfig,
    Pi0TrainerConfig,
)
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
class Pi05ModelConfig(ModelConfig):
    model_name_or_path: str = field(default="./checkpoints/Dexbotic-PI05")

    def build_model(self) -> Pi05ForCausalLM:
        model = Pi05ForCausalLM.from_pretrained(self.model_name_or_path)
        model.model.config.chunk_size = 50
        return model


class Pi05TrainerConfig(Pi0TrainerConfig):
    model_max_length: int = field(default=200)


@dataclass
class Pi05ActionConfig(ActionConfig):
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
class Pi05DataConfig(Pi0DataConfig):
    action_config: Pi05ActionConfig = field(default_factory=Pi05ActionConfig)


@dataclass
class Pi05InferenceConfig(Pi0InferenceConfig):
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


@dataclass
class Pi05Exp(BaseExp):
    model_config: Pi05ModelConfig = field(default_factory=Pi05ModelConfig)
    optimizer_config: Pi0OptimizerConfig = field(default_factory=Pi0OptimizerConfig)
    trainer_config: Pi05TrainerConfig = field(default_factory=Pi05TrainerConfig)
    data_config: Pi05DataConfig = field(default_factory=Pi05DataConfig)
    tokenizer_config: Pi0TokenizerConfig = field(default_factory=Pi0TokenizerConfig)
    inference_config: Pi0InferenceConfig = field(default_factory=Pi0InferenceConfig)

    def inference(self) -> None:
        self.inference_config.run()

    def compute_norm_stats(self) -> None:
        self.data_config.action_config = Pi0ComputeNormActionConfig()
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
        norm_config = Pi0ComputeNormActionConfig()
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
