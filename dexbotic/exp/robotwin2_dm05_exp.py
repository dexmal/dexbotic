"""DM05 RoboTwin2 generalist absolute-action SFT, using the LIBERO architecture."""

import json
import os
from dataclasses import dataclass, field

import megfile
import numpy as np
import torch
from easydict import EasyDict
from torch.utils.data import ConcatDataset, DataLoader

from dexbotic.data.data_source.robotwin2_sft import ROBOTWIN2_SETTINGS, ROBOTWIN2_TASKS
from dexbotic.data.dataset.dm05_data import DM05Tokenization
from dexbotic.data.dataset.rgb_preprocess import DummyRGBProcessor
from dexbotic.data.dataset.robotwin2_dm05_data import (
    RoboTwin2ActionNorm,
    RoboTwin2Dataset,
    RoboTwin2DM05DataCollator,
    RoboTwin2ImagePreprocess,
    SampleRoboTwin2Prompt,
)
from dexbotic.data.dataset.transform.action import (
    AddAction,
    AddTrajectory,
    PadAction,
    PadState,
)
from dexbotic.data.dataset.transform.common import Pipeline, ToDict, ToList, ToNumpy
from dexbotic.data.dataset.transform.language import BuildConversationRoleContent
from dexbotic.data.dataset.transform.multimodal import LoadMultiModal
from dexbotic.data.utils import normalize
from dexbotic.exp.dm05_exp import (
    DM05ActionConfig,
    DM05ComputeNormActionConfig,
    DM05DataConfig,
    DM05Exp,
    DM05ModelConfig,
    DM05OptimizerConfig,
    DM05TrainerConfig,
)
from dexbotic.optimizer.muon_adamw import MUON_ENABLED_ATTR, MuonAdamW

CRITICAL_TASKS = frozenset(
    {
        "hanging_mug",
        "move_stapler_pad",
        "turn_switch",
        "place_dual_shoes",
        "scan_object",
        "place_mouse_pad",
        "beat_block_hammer",
        "stack_bowls_three",
    }
)
WATCH_TASKS = frozenset(
    {
        "blocks_ranking_size",
        "place_can_basket",
        "stamp_seal",
        "put_object_cabinet",
        "place_bread_skillet",
        "put_bottles_dustbin",
        "place_object_basket",
        "pick_diverse_bottles",
    }
)
DATASET_NAMES = [
    f"robotwin2_example_{task}_{setting}"
    for task in ROBOTWIN2_TASKS
    for setting in ROBOTWIN2_SETTINGS
]


def dataset_weight(name):
    task, setting = name.removeprefix("robotwin2_example_").rsplit("_", 1)
    weight = 0.2 if setting == "clean" else 1.0
    return weight * (
        1.8 if task in CRITICAL_TASKS else 1.25 if task in WATCH_TASKS else 1.0
    )


def prepare_muon_parameters(model):
    for name, param in model.named_parameters():
        selected = (
            param.requires_grad
            and param.ndim == 2
            and "action_expert.layers." in name
            and name.endswith(".weight")
            and not any(
                marker in name.lower()
                for marker in ("embed", "layernorm.weight", "_norm.weight", ".norm.")
            )
        )
        setattr(param, MUON_ENABLED_ATTR, selected)


@dataclass
class RoboTwin2ModelConfig(DM05ModelConfig):
    chunk_size: int = 50
    vlm_gradient_checkpointing: bool = True
    ae_gradient_checkpointing: bool = True

    def build_model(self, use_lora=False):
        model = super().build_model(use_lora=use_lora)
        if model.config.action_dim != 32:
            raise ValueError("RoboTwin2 DM05 SFT requires a 32-D action head")
        prepare_muon_parameters(model)
        return model


@dataclass
class RoboTwin2OptimizerConfig(DM05OptimizerConfig):
    base_lr: float = 4e-5
    optim: str = "muon_adamw"

    def build_optimizer(self, model, training_args):
        kwargs = dict(
            lr=self.base_lr,
            betas=(self.adam_beta1, self.adam_beta2),
            eps=self.adam_epsilon,
            weight_decay=self.weight_decay,
        )
        if self.optim == "adamw_torch":
            return torch.optim.AdamW(
                self._get_optimizer_grouped_parameters(model), **kwargs
            )
        if self.optim != "muon_adamw":
            raise ValueError(f"Unsupported RoboTwin2 optimizer: {self.optim}")
        if not any(getattr(p, MUON_ENABLED_ATTR, False) for p in model.parameters()):
            raise ValueError(
                "Muon parameters must be selected before constructing the optimizer"
            )
        return MuonAdamW.from_named_parameters(model.named_parameters(), **kwargs)


@dataclass
class RoboTwin2TrainerConfig(DM05TrainerConfig):
    train_backend: str = "fsdp2"
    fsdp_config: dict = field(
        default_factory=lambda: {"activation_checkpointing": False}
    )
    model_max_length: int = 1152
    num_train_steps: int = 100000
    save_steps: int = 10000
    save_total_limit: int = 100
    save_only_model: bool = False
    per_device_train_batch_size: int = 32
    wandb_project: str = "dm05_sft_robotwin2"
    output_dir: str = "./user_checkpoints/dexbotic/robotwin2_dm05/generalist-absolute"


class ValidateRoboTwin2State:
    def __call__(self, data, **kwargs):
        state = np.asarray(data["state"])
        if state.ndim != 2 or state.shape[-1] != 14 or not np.isfinite(state).all():
            raise ValueError(
                f"Expected finite RoboTwin2 state [N,14], got {state.shape}"
            )
        data["meta_data"]["non_delta_mask"] = np.ones(22, dtype=bool)
        data.setdefault("is_robot", np.ones(len(state), dtype=bool))
        return data


def absolute_action_steps(trajectory_length):
    return [
        ToDict(),
        ToNumpy(),
        ValidateRoboTwin2State(),
        PadState(ndim=22),
        AddAction(predict_length=1),
        AddTrajectory(
            trajectory_length=trajectory_length,
            flatten=False,
            padding_mode="last",
            padding_action=True,
        ),
    ]


@dataclass
class RoboTwin2ActionConfig(DM05ActionConfig):
    trajectory_length: int = 50

    def _read_norm_stats(self, norm_stats_path):
        if not norm_stats_path or not megfile.smart_exists(norm_stats_path):
            raise FileNotFoundError(
                f"RoboTwin2 absolute-action norm stats not found: {norm_stats_path}"
            )
        with megfile.smart_open(norm_stats_path, "r") as stream:
            stats = json.load(stream)
        stats = stats.get("norm_stats", stats)
        result = {}
        for key in ("state", "action"):
            result[key] = {
                bound: np.asarray(stats[key][bound], dtype=np.float32)
                for bound in ("min", "max")
            }
            if any(
                value.shape != (22,) or not np.isfinite(value).all()
                for value in result[key].values()
            ):
                raise ValueError(
                    f"RoboTwin2 {key} norm stats must have 22 finite min/max values"
                )
            if (result[key]["max"] < result[key]["min"]).any():
                raise ValueError(f"RoboTwin2 {key} max must be >= min")
        return result

    def build_action_process_func(self):
        return Pipeline(
            [
                *absolute_action_steps(self.trajectory_length),
                RoboTwin2ActionNorm(
                    statistic_mapping=self._read_norm_stats(self.statistic_mapping)
                ),
                SampleRoboTwin2Prompt(),
                BuildConversationRoleContent(),
                PadState(ndim=32),
                PadAction(ndim=32),
                LoadMultiModal(return_masks=True),
                ToList(),
            ]
        )


@dataclass
class RoboTwin2DataConfig(DM05DataConfig):
    dataset_name: str = "+".join(DATASET_NAMES)
    num_images: int = 3
    images_keys: list[str] = field(
        default_factory=lambda: ["images_1", "images_2", "images_3"]
    )
    valid_action_dim: int = 14
    tokenizer_max_length: int = 1152
    action_config: RoboTwin2ActionConfig = field(default_factory=RoboTwin2ActionConfig)

    def __post_init__(self):
        if self.dataset_weights is None:
            self.dataset_weights = [
                dataset_weight(name) for name in self.dataset_name.split("+")
            ]

    def build_data(self, processor, max_length=None):
        names = self.dataset_name.split("+")
        if len(self.dataset_weights) != len(names):
            raise ValueError("dataset_weights must have one entry per dataset")
        transform = self.action_config.build_action_process_func()
        datasets = [
            RoboTwin2Dataset(
                data_args=EasyDict(
                    dataset_name=name,
                    num_images=3,
                    data_keys=self.data_keys,
                    images_keys=self.images_keys,
                    image_processor=processor.image_processor,
                ),
                tokenization_func=DM05Tokenization(),
                action_process_func=transform,
                image_process_func=RoboTwin2ImagePreprocess(),
                depth_process_func=DummyRGBProcessor(),
            )
            for name in names
        ]
        dataset = ConcatDataset(datasets)
        dataset.action_process_func = transform
        collator = RoboTwin2DM05DataCollator(
            processor,
            max_length=max_length or self.tokenizer_max_length,
            valid_action_dim=14,
            model_action_dim=32,
            chunk_size=self.action_config.trajectory_length,
        )
        return dataset, collator


@dataclass
class RoboTwin2ComputeNormActionConfig(DM05ComputeNormActionConfig):
    norm_save_path: str = "./dexbotic/norm_assets/robotwin2_absolute_action/default"
    trajectory_length: int = 50
    batch_size: int = 128
    num_workers: int = 4
    max_compute_iters: int = 500

    def build_action_process_func(self):
        return Pipeline([*absolute_action_steps(self.trajectory_length), ToList()])

    def _get_dataset(self, action_process_func, dataset_name_list):
        return [
            (
                name,
                RoboTwin2Dataset(
                    data_args=EasyDict(
                        dataset_name=name, num_images=1, data_keys=["action", "state"]
                    ),
                    tokenization_func=DM05Tokenization(),
                    action_process_func=action_process_func,
                    image_process_func=DummyRGBProcessor(),
                    depth_process_func=DummyRGBProcessor(),
                ),
            )
            for name in dataset_name_list
        ]

    def _process_one_dataset(self, dataset_name, dataset):
        stats = {key: normalize.RunningStats() for key in ("state", "action")}
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        for index, batch in enumerate(loader):
            if index >= self.max_compute_iters:
                break
            for key in stats:
                values = batch[key].numpy()
                stats[key].update(values.reshape(-1, values.shape[-1]))
        path = os.path.join(self.norm_save_path, dataset_name)
        normalize.save(
            path, {key: value.get_statistics() for key, value in stats.items()}
        )
        return os.path.join(path, "norm_stats.json")


@dataclass
class RoboTwin2SFTExp(DM05Exp):
    model_config: RoboTwin2ModelConfig = field(default_factory=RoboTwin2ModelConfig)
    optimizer_config: RoboTwin2OptimizerConfig = field(
        default_factory=RoboTwin2OptimizerConfig
    )
    trainer_config: RoboTwin2TrainerConfig = field(
        default_factory=RoboTwin2TrainerConfig
    )
    data_config: RoboTwin2DataConfig = field(default_factory=RoboTwin2DataConfig)

    def _build_compute_norm_config(self):
        return RoboTwin2ComputeNormActionConfig(
            trajectory_length=self.data_config.action_config.trajectory_length
        )

    def compute_norm_stats(self):
        config = self._build_compute_norm_config()
        config.compute_norm_stats(self.data_config.dataset_name)
        self.data_config.action_config.statistic_mapping = os.path.join(
            config.norm_save_path, "norm_stats.json"
        )
