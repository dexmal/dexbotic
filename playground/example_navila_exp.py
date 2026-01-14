"""
There is a example of navila inference.
Usage:

python playground/example_navila_exp.py --task inference_single --image_path test_data/navila_test.png --prompt "Go around the right side of the center unit and stop by the right side doorway with the dining table and mirror in it."

"""
import argparse
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from dexbotic.data.dataset.transform.common import Pipeline, ToDict, ToList, ToNumpy
from dexbotic.data.dataset.transform.multimodal import LoadMultiModal
from dexbotic.exp.navila_exp import (
    NaVILAActionConfig,
    NaVILADataConfig,
    NaVILAExp,
    NaVILAInferenceConfig,
    NaVILAModelConfig,
    NaVILAOptimizerConfig,
    NaVILATokenizerConfig,
    NaVILATrainerConfig,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        choices=["train", "inference", "inference_single"],
    )
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    args, unknown = parser.parse_known_args()
    return args


@dataclass
class OptimizerConfig(NaVILAOptimizerConfig):
    base_lr: float = field(default=1e-5)
    warmup_steps: int = field(default=100)


@dataclass
class TrainerConfig(NaVILATrainerConfig):
    bf16: bool = field(default=True)
    num_train_epochs: int = field(default=1)
    save_steps: int = field(default=200)
    save_total_limit: int = field(default=5)
    per_device_train_batch_size: int = field(default=10)
    gradient_accumulation_steps: int = field(default=2)
    model_max_length: int = field(default=4096)
    output_dir: str = field(
        default=f"./user_checkpoints/dexbotic/navigation_navila/{datetime.now().strftime('%m%d')}"
    )


@dataclass
class ActionConfig(NaVILAActionConfig):
    def build_action_process_func(self) -> Pipeline:
        action_config = Pipeline(
            [
                ToDict(),
                ToNumpy(),
                LoadMultiModal(return_masks=True),
                ToList(),
            ]
        )
        return action_config


@dataclass
class DataConfig(NaVILADataConfig):
    dataset_name: str = field(default="navila_R2R")
    num_images: int = field(default=8)
    data_keys: list[str] = field(
        default_factory=lambda: [
            "input_ids",
            "labels",
            "image",
            "image_masks",
        ]
    )
    aug_policy: str | list[str] = field(default_factory=lambda: ["identity"] * 8)
    action_config: ActionConfig = field(default_factory=ActionConfig)


@dataclass
class ModelConfig(NaVILAModelConfig):
    model_name_or_path: str = field(default="./checkpoints/dex_navila")


@dataclass
class TokenizerConfig(NaVILATokenizerConfig):
    use_fast_tokenizer: bool = field(default=False)


@dataclass
class InferenceConfig(NaVILAInferenceConfig):
    model_name_or_path: Optional[str] = field(default="./checkpoints/dex_navila")
    port: int = field(default=7891)
    save_image: bool = field(default=False)
    save_image_dir: str = field(default="./debug_data")
    num_images: int = field(default=8)
    policy_name: str = field(default="navila")
    history_buffer: deque = field(default=None, init=False, repr=False)


@dataclass
class Exp(NaVILAExp):
    model_config: ModelConfig = field(default_factory=ModelConfig)
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    trainer_config: TrainerConfig = field(default_factory=TrainerConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    tokenizer_config: TokenizerConfig = field(default_factory=TokenizerConfig)
    inference_config: InferenceConfig = field(default_factory=InferenceConfig)

    def inference(self) -> None:
        self.inference_config.run()

    def inference_single(
        self,
        image_path: str,
        prompt: str,
        reset_memory: bool = True,
        run_model: bool = True,
    ):
        self.inference_config._initialize_inference()

        self.inference_config.meta_data = {
            "reset_memory": reset_memory,
            "run_model": run_model,
        }
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
        except FileNotFoundError:
            print(f"Error: image file not found {image_path}")
            return None

        images_list = self.inference_config._prepare_images(image_bytes)
        for stream in images_list:
            stream.seek(0)

        result = self.inference_config._get_response(
            text=prompt,
            images=images_list,
        )
        print(f"Inference result: {result}")
        return result


if __name__ == "__main__":
    args = parse_args()
    exp = Exp()
    if args.task == "train":
        exp.train()
    elif args.task == "inference":
        exp.inference()
    elif args.task == "inference_single":
        exp.inference_single(args.image_path, args.prompt)
