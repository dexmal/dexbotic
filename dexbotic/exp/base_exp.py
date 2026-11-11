import base64
import hashlib
import json
import os
import pathlib
import random
import sys
import time
import uuid
from dataclasses import dataclass, field, fields
from datetime import datetime
from io import BytesIO
from typing import Callable, Dict, Optional, Union

import megfile
import numpy as np
import torch
import tqdm
import transformers
from easydict import EasyDict
from flask import Flask, jsonify, request
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoTokenizer, BaseImageProcessor
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import get_last_checkpoint

try:
    from transformers.trainer import ALL_LAYERNORM_LAYERS
except ImportError:
    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

import dexbotic.data.utils.normalize as normalize
from dexbotic.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from dexbotic.data.collator import DataCollatorForSupervisedDataset
from dexbotic.data.dataset.dex_dataset import DexDataset
from dexbotic.data.dataset.rgb_preprocess import DummyRGBProcessor
from dexbotic.data.dataset.tokenization import DummyTokenization
from dexbotic.data.dataset.transform.action import (
    ActionNormAnd2String,
    AddAction,
    AddTrajectory,
    DeltaAction,
)
from dexbotic.data.dataset.transform.common import Pipeline, ToDict, ToList, ToNumpy
from dexbotic.data.dataset.transform.language import (
    AddPromptTemplate,
    ReplaceAnswer,
    defalut_prompt_template,
)
from dexbotic.data.dataset.transform.multimodal import LoadMultiModal
from dexbotic.exp.backend_resolver import (
    BackendResolution,
    FSDPProfile,
    apply_backend_defaults,
    resolve_backend_mode,
)
from dexbotic.exp.trainer import DexboticTrainer, safe_save_model_for_hf_trainer
from dexbotic.exp.utils import NumpyEncoder, enter_debug_mode
from dexbotic.model.dexbotic_arch import (
    DexboticConfig,
    DexboticForCausalLM,
    DexboticVLMModel,
)
from dexbotic.policy.types import GenSamplingConfig, SamplingConfig
from dexbotic.tokenization import conversation as conversation_lib
from dexbotic.tokenization.conversation import KeywordsStoppingCriteria
from dexbotic.tokenization.process import LLMTokenization
from dexbotic.tokenization.tokenization import tokenizer_image_token

OPENAI_CLIP_PATH = os.environ.get(
    "OPENAI_CLIP_PATH", "openai/clip-vit-large-patch14-336"
)


@dataclass
class Config:
    pass


@dataclass
class OptimizerConfig(Config):
    """
    Optimizer configuration class - controls optimizer parameters during model training

    Configuration details:
    - optim: Optimizer type
    - base_lr: Base learning rate
    - weight_decay: Weight decay
    - warmup_ratio: Learning rate warmup ratio
    - warmup_steps: Learning rate warmup steps
    - adam_beta1/adam_beta2: Adam optimizer momentum parameters
    - adam_epsilon: Adam optimizer numerical stability parameter
    - mm_projector_lr/mm_vision_lr/action_head_lr: Independent learning rates for different modules
    """

    optim: str = field(default="adamw_torch")
    base_lr: float = field(default=2e-5)

    weight_decay: float = field(default=0.0)
    warmup_ratio: float = field(default=0.03)
    warmup_steps: int = field(default=0)

    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    mm_projector_lr: Optional[float] = field(default=None)
    mm_vision_lr: Optional[float] = field(default=None)
    action_head_lr: Optional[float] = field(default=None)

    def build_optimizer(
        self,
        model: DexboticVLMModel,
        training_args: transformers.TrainingArguments,
    ) -> torch.optim.Optimizer:
        """Build the default HF optimizer; recipes may override this factory."""
        parameters = self._get_optimizer_grouped_parameters(model)
        (
            optimizer_cls,
            optimizer_kwargs,
        ) = transformers.Trainer.get_optimizer_cls_and_kwargs(training_args)
        return optimizer_cls(parameters, **optimizer_kwargs)

    def _get_optimizer_grouped_parameters(self, model: DexboticVLMModel) -> list:
        """Returns a list of dictionaries containing parameters grouped by their names
        and whether they require weight decay.
        """

        decay_params_name = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        decay_params_name = [name for name in decay_params_name if "bias" not in name]

        optimizer_grouped_parameters = []

        # split params into base, mm_projector, mm_vision, action_head
        mm_projector_params_name = []
        mm_vision_params_name = []
        action_head_params_name = []

        if self.mm_projector_lr is not None:
            logger.info("Using mm_projector_lr: {}", self.mm_projector_lr)
            mm_projector_params_name = [
                name
                for name, _ in model.named_parameters()
                if model.mm_projector_prefix in name
            ]

            mm_projector_decay_params = {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (
                        p.requires_grad
                        and n in decay_params_name
                        and n in mm_projector_params_name
                    )
                ],
                "weight_decay": self.weight_decay,
                "lr": self.mm_projector_lr,
            }
            mm_projector_no_decay_params = {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (
                        p.requires_grad
                        and n not in decay_params_name
                        and n in mm_projector_params_name
                    )
                ],
                "weight_decay": 0.0,
                "lr": self.mm_projector_lr,
            }

            optimizer_grouped_parameters.append(mm_projector_decay_params)
            optimizer_grouped_parameters.append(mm_projector_no_decay_params)

        if self.mm_vision_lr is not None:
            logger.info("Using mm_vision_lr: {}", self.mm_vision_lr)
            mm_vision_params_name = [
                name
                for name, _ in model.named_parameters()
                if model.mm_vision_prefix in name
            ]

            mm_vision_decay_params = {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (
                        p.requires_grad
                        and n in decay_params_name
                        and n in mm_vision_params_name
                    )
                ],
                "weight_decay": self.weight_decay,
                "lr": self.mm_vision_lr,
            }
            mm_vision_no_decay_params = {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (
                        p.requires_grad
                        and n not in decay_params_name
                        and n in mm_vision_params_name
                    )
                ],
                "weight_decay": 0.0,
                "lr": self.mm_vision_lr,
            }
            optimizer_grouped_parameters.append(mm_vision_decay_params)
            optimizer_grouped_parameters.append(mm_vision_no_decay_params)

        if self.action_head_lr is not None:
            logger.info("Using action_head_lr: {}", self.action_head_lr)
            action_head_params_name = [
                name
                for name, _ in model.named_parameters()
                if model.action_head_prefix in name
            ]

            action_head_decay_params = {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (
                        p.requires_grad
                        and n in decay_params_name
                        and n in action_head_params_name
                    )
                ],
                "weight_decay": self.weight_decay,
                "lr": self.action_head_lr,
            }
            action_head_no_decay_params = {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (
                        p.requires_grad
                        and n not in decay_params_name
                        and n in action_head_params_name
                    )
                ],
                "weight_decay": 0.0,
                "lr": self.action_head_lr,
            }
            optimizer_grouped_parameters.append(action_head_decay_params)
            optimizer_grouped_parameters.append(action_head_no_decay_params)

        base_decay_params = {
            "params": [
                p
                for n, p in model.named_parameters()
                if (
                    p.requires_grad
                    and n in decay_params_name
                    and n not in mm_projector_params_name
                    and n not in mm_vision_params_name
                    and n not in action_head_params_name
                )
            ],
            "weight_decay": self.weight_decay,
            "lr": self.base_lr,
        }

        base_no_decay_params = {
            "params": [
                p
                for n, p in model.named_parameters()
                if (
                    p.requires_grad
                    and n not in decay_params_name
                    and n not in mm_projector_params_name
                    and n not in mm_vision_params_name
                    and n not in action_head_params_name
                )
            ],
            "weight_decay": 0.0,
            "lr": self.base_lr,
        }
        optimizer_grouped_parameters.append(base_decay_params)
        optimizer_grouped_parameters.append(base_no_decay_params)

        return optimizer_grouped_parameters


@dataclass
class TrainerConfig(Config):
    """
    Trainer configuration class - controls training process parameters

    Configuration details:
    - deepspeed: DeepSpeed configuration file path
    - output_dir: Training output directory
    - num_train_epochs: Number of training epochs
    - per_device_train_batch_size: Batch size per device
    - gradient_accumulation_steps: Gradient accumulation steps
    - save_strategy/save_steps/save_total_limit/save_only_model: Model saving parameters
    - logging_steps: Logging frequency
    - wandb_project: Weights & Biases project name
    - gradient_checkpointing: Gradient checkpointing
    - dataloader_num_workers: Number of worker processes for data loader
    - model_max_length: Maximum sequence length for the model
    - debug_mode: Debug mode
    - bf16/tf32: Training precision settings
    - lr_scheduler_type: Learning rate scheduler type
    - tune_mm_mlp_adapter: Whether to enter mm_mlp_adapter-only training mode
    """

    train_backend: str = field(default="deepspeed")
    deepspeed: Optional[str] = field(default="./script/deepspeed/zero2.json")
    fsdp: Optional[str] = field(default=None)
    fsdp_config: Optional[dict] = field(default=None)
    fsdp_version: Optional[int] = field(default=None)
    fsdp_profile: FSDPProfile = field(default_factory=FSDPProfile)
    output_dir: Optional[str] = field(default=None)

    num_train_epochs: int = field(default=1)
    num_train_steps: Optional[int] = field(default=-1)
    per_device_train_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=2)

    save_strategy: str = field(default="steps")
    save_steps: int = field(default=20000)
    save_total_limit: int = field(default=1)
    save_only_model: bool = field(default=True)

    logging_steps: int = field(default=10)
    wandb_project: str = field(default="dexbotic")

    gradient_checkpointing: bool = field(default=True)

    dataloader_num_workers: int = field(default=8)

    model_max_length: int = field(default=2048)

    debug_mode: bool = field(default=False)

    bf16: bool = field(default=True)
    tf32: bool = field(default=True)

    lr_scheduler_type: str = field(default="cosine")
    lr_scheduler_kwargs: dict = field(default_factory=dict)

    tune_mm_mlp_adapter: bool = field(default=False)

    def ensure_backend_defaults(self) -> None:
        """Populate backend-specific defaults lazily when a backend is selected."""
        apply_backend_defaults(self)

    def __post_init__(self):
        if self.train_backend not in {"deepspeed", "fsdp", "fsdp2", "ddp"}:
            raise ValueError(
                f"Unsupported train_backend: {self.train_backend}. "
                "Expected one of: deepspeed, fsdp, fsdp2, ddp."
            )
        if self.output_dir is not None:
            self.run_name = os.path.basename(self.output_dir)
        if self.wandb_project is not None:
            os.environ["WANDB_PROJECT"] = self.wandb_project


@dataclass
class ModelConfig(Config):
    """
    Model configuration class - controls model architecture and initialization parameters

    Configuration details:
    - model_name_or_path: Pre-trained model path or HuggingFace model name
    - chat_template: Chat template type
    - mm_projector_type: Multi-modal projector type
    - mm_vision_tower: Vision encoder path
    - from_llm: Whether to build model purely from LLM
    - freeze_llm/freeze_mm_projector/freeze_mm_vision: Controls whether different modules are frozen during training
    """

    model_name_or_path: str = field(default=None)
    chat_template: str = field(default="dexbotic")

    mm_projector_type: str = field(default="mlp2x_gelu")
    mm_vision_tower: str = field(default=OPENAI_CLIP_PATH)
    from_llm: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_mm_projector: bool = field(default=False)
    freeze_mm_vision: bool = field(default=False)

    def build_model(self) -> DexboticForCausalLM:
        if self.from_llm:
            model_config_args = {
                "llm_config": self.model_name_or_path,
                "chat_template": self.chat_template,
                "mm_projector_type": self.mm_projector_type,
                "mm_vision_tower": self.mm_vision_tower,
                "init_llm_weights": True,
            }
            model_config = DexboticConfig(**model_config_args)
            model = DexboticForCausalLM(model_config)
        else:
            model_config_args = {
                "model_name_or_path": self.model_name_or_path,
                "mm_projector_type": self.mm_projector_type,
                "mm_vision_tower": self.mm_vision_tower,
            }
            model = DexboticForCausalLM.from_pretrained(self.model_name_or_path)
            model.model.initialize_model(model_config_args)

        self._freeze_model(model)

        return model

    def _freeze_model(self, model: DexboticForCausalLM):
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


@dataclass
class TokenizerConfig(Config):
    """
    Tokenizer configuration class - controls text tokenization and special token processing

    Configuration details:
    - use_special_tokens: Whether to use special tokens, used for action prediction tasks
    - use_fast_tokenizer: Whether to use fast tokenizer
    """

    use_special_tokens: bool = field(default=False)
    use_fast_tokenizer: bool = field(default=True)

    def build_tokenizer(
        self, model_name_or_path: str, **kwargs
    ) -> transformers.PreTrainedTokenizer:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path, **kwargs
        )
        if tokenizer.unk_token is not None and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
        return tokenizer

    def add_special_tokens(
        self,
        special_token_format: str,
        vocab_size: int,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
    ) -> transformers.PreTrainedTokenizer:
        if not self.use_special_tokens:
            return tokenizer

        special_tokens = [special_token_format.format(i) for i in range(vocab_size)]
        tokenizer.add_tokens(special_tokens, special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
        return tokenizer


@dataclass
class ActionConfig(Config):
    """
    Action configuration class - controls action data processing and normalization parameters

    Configuration details:
    - statistic_mapping: Action statistics mapping file path, contains action normalization parameters
    - replace_with_default_answer: String to replace answer with default answer, used for models that don't output text
    - trajectory_length: Trajectory length, controls the length of action sequences
    - delta: Whether to use delta action representation
    - trajectory_padding_model: Trajectory padding mode. choices: 'zero', 'last'.
    - padding_action: Whether to pad actions when episode is shorter than a trajectory
    - vocab_size: Action vocabulary size, used for string formatting action
    - string_format: Action string formatting template, used for string formatting action
    - prompt_template: Prompt template
    """

    statistic_mapping: str = field(default=None)
    replace_with_default_answer: str = field(default=" ")
    trajectory_length: int = field(default=16)
    delta: bool = field(default=True)
    trajectory_padding_model: str = field(default="zero")
    padding_action: bool = field(default=False)
    vocab_size: int = field(default=255)
    string_format: str = field(default=" {value}")
    prompt_template: Union[str, Callable[[str], str]] = field(
        default=defalut_prompt_template
    )

    def build_action_process_func(self) -> Pipeline:
        statistic_mapping = self._read_norm_stats(self.statistic_mapping)
        action_config = Pipeline(
            [
                ToDict(),
                ToNumpy(),
                AddAction(predict_length=1),
                DeltaAction(enable=self.delta),
                AddTrajectory(
                    trajectory_length=self.trajectory_length,
                    padding_mode=self.trajectory_padding_model,
                    padding_action=self.padding_action,
                ),
                ActionNormAnd2String(
                    statistic_mapping=statistic_mapping,
                    vocab_size=self.vocab_size,
                    string_format=self.string_format,
                ),
                LoadMultiModal(),
                AddPromptTemplate(prompt_template=self.prompt_template),
                ReplaceAnswer(default_answer=self.replace_with_default_answer),
                ToList(),
            ]
        )

        return action_config

    def _read_norm_stats(self, norm_stats_path):
        assert megfile.smart_exists(
            norm_stats_path
        ), f"Norm stats file {norm_stats_path} not found"
        with megfile.smart_open(norm_stats_path, "r") as f:
            norm_stats = json.load(f)["norm_stats"]
            norm_stats = ToNumpy()(norm_stats)
        return norm_stats


@dataclass
class ComputeNormActionConfig(ActionConfig):
    """
    Configuration class for computing action normalization parameters
    """

    norm_method: str = field(default="default")

    norm_save_path: str = field(
        default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "norm_assets",
            f'{datetime.now().strftime("%m%d-%H%M")}-default',
        )
    )

    def build_action_process_func(self) -> Pipeline:
        action_config = Pipeline(
            [
                ToDict(),
                ToNumpy(),
                AddAction(predict_length=1),
                DeltaAction(enable=self.delta),
                ToList(),
            ]
        )

        return action_config

    def compute_norm_stats(self, dataset_name: str) -> None:
        dataset_name_list = dataset_name.split("+")
        action_process_func = self.build_action_process_func()
        dataset_list = self._get_dataset(action_process_func, dataset_name_list)
        norm_files = {}

        for dataset_name, dataset in dataset_list:
            if dataset_name.startswith("general"):
                continue
            norm_file = self._process_one_dataset(dataset_name, dataset)
            norm_files[dataset_name] = (norm_file, dataset.dataset_map[0])

        self._merge_norm_stats(norm_files)

    def _get_dataset(self, action_process_func, dataset_name_list):
        robot_dataset_list = []
        for dataset_name in dataset_name_list:
            robot_dataset = DexDataset(
                data_args=EasyDict(
                    dataset_name=dataset_name,
                    num_images=1,
                    data_keys=["action"],
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

    def _process_one_dataset(self, dataset_name, dataset):
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=64)

        norm_keys = ["action"]
        stats = {key: normalize.RunningStats() for key in norm_keys}
        for batch_idx, batch in tqdm.tqdm(
            enumerate(dataloader), desc="Computing norm stats"
        ):
            # only use the first 500 * 128 samples
            if batch_idx > 500:
                break
            for key in norm_keys:
                values = batch[key].numpy()
                stats[key].update(values)
        norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

        save_path = os.path.join(self.norm_save_path, dataset_name)
        logger.info(f"Saving norm stats to {save_path}")
        normalize.save(save_path, norm_stats)

        return os.path.join(save_path, "norm_stats.json")

    def _merge_norm_stats(self, norm_files, per_task_norm=False):
        norm_stats = {
            "default": {"min": -1, "max": 1},
        }
        min_list = []
        max_list = []
        for dataset_name, (norm_file, dataset_path) in norm_files.items():
            with open(norm_file, "r") as f:
                stats = json.load(f)["norm_stats"]["action"]
            if per_task_norm:
                if self.norm_method == "default":
                    norm_stats[dataset_path] = {
                        "default": {
                            "min": stats["q01"],
                            "max": stats["q99"],
                        }
                    }
                else:
                    norm_stats[dataset_path] = {
                        "default": {
                            "min": stats["min"],
                            "max": stats["max"],
                        }
                    }
            if self.norm_method == "default":
                min_list.append(stats["q01"])
                max_list.append(stats["q99"])
            else:
                min_list.append(stats["min"])
                max_list.append(stats["max"])

        min_list = np.array(min_list).min(axis=0).tolist()
        max_list = np.array(max_list).max(axis=0).tolist()
        norm_stats["default"] = {
            "min": min_list,
            "max": max_list,
        }

        with open(os.path.join(self.norm_save_path, "norm_stats.json"), "w") as f:
            json.dump({"norm_stats": norm_stats}, f, indent=2)

    def __post_init__(self):
        if self.norm_method not in ["default", "minmax"]:
            raise ValueError(f"Invalid norm method: {self.norm_method}")


@dataclass
class DataConfig(Config):
    """
    Data configuration class - controls dataset and data processing parameters

    Configuration details:
    - dataset_name: Dataset name
    - dataset_weights: Optional sampling weights for the children of a ConcatDataset
    - num_images: Number of images
    - data_keys: Data key list, used for loading specified data fields
    - images_keys: Image key list, specifies image data field names. If None, use all image data fields
    - aug_policy: Data augmentation strategy
    - image_aspect_ratio: Image aspect ratio processing method
    - action_config: Action configuration instance, contains action-related parameters
    - auto_norm: Whether to automatically compute action normalization parameters, if True, automatically computes and saves to norm_assets directory
    - auto_norm_method: Method for automatically computing action normalization parameters
    """

    dataset_name: str = field(default=None)
    dataset_weights: Optional[list[float]] = field(default=None)
    num_images: int = field(default=1)
    data_keys: list[str] = field(
        default_factory=lambda: ["input_ids", "labels", "action", "image"]
    )
    images_keys: list[str] = field(default=None)
    aug_policy: str | list[str] = field(default="v3")
    image_aspect_ratio: str = field(default="pad")
    action_config: ActionConfig = field(default_factory=ActionConfig)
    auto_norm: bool = field(default=True)
    auto_norm_method: str = field(default="default")
    image_pad_mode: str = field(default="mean")

    def build_data(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        chat_template: str,
        image_processor: BaseImageProcessor,
    ) -> Dict:
        dataset = self._build_dataset(tokenizer, chat_template, image_processor)
        data_collator = self._build_data_collator(tokenizer)
        return dataset, data_collator

    def _build_dataset(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        chat_template: str,
        image_processor: BaseImageProcessor,
    ) -> DexDataset:
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
            }
        )
        action_process_func = self.action_config.build_action_process_func()
        tokenization_func = LLMTokenization(tokenizer, data_args)
        dataset = DexDataset(
            data_args=data_args,
            tokenization_func=tokenization_func,
            action_process_func=action_process_func,
        )
        return dataset

    def _build_data_collator(
        self, tokenizer: transformers.PreTrainedTokenizer
    ) -> DataCollatorForSupervisedDataset:
        return DataCollatorForSupervisedDataset(tokenizer)


@dataclass
class InferenceConfig(Config):
    """
    Inference configuration class - controls inference service parameters

    Configuration details:
    - model_name_or_path: Model path used for inference
    - port: Inference service port number
    - save_image: Whether to save input images for debugging
    - save_image_dir: Directory path for saving images
    - norm_stats: Action normalization statistics
    """

    model_name_or_path: Optional[str] = field(default=None)
    port: int = field(default=7891)
    save_image: bool = field(default=False)
    save_image_dir: str = field(default="./debug_data")
    norm_stats: Optional[dict] = field(default=None)
    camera_order: list = field(default_factory=lambda: ["front"])

    def _apply_inference_seed(self, seed) -> None:
        if seed is None or seed == "":
            return
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def process_frame(self) -> None:
        self._apply_inference_seed(request.form.get("seed"))
        results = self._get_response(
            text=request.form.get("text"),
            images=request.files.getlist("image"),
        )
        return jsonify({"response": results})

    def run(self) -> None:
        self._initialize_inference()
        self.app = Flask(__name__)
        # legacy route — kept for backward compatibility
        self.app.add_url_rule(
            "/process_frame", "process_frame", self.process_frame, methods=["POST"]
        )
        # v1 routes
        self.app.add_url_rule("/health", "health", self._v1_health, methods=["GET"])
        self.app.add_url_rule(
            "/v1/models", "v1_models", self._v1_models, methods=["GET"]
        )
        self.app.add_url_rule(
            "/v1/capabilities",
            "v1_capabilities",
            self._v1_capabilities,
            methods=["GET"],
        )
        self.app.add_url_rule("/v1/infer", "v1_infer", self._v1_infer, methods=["POST"])
        self.app.add_url_rule("/v1/reset", "v1_reset", self._v1_reset, methods=["POST"])
        self.app.add_url_rule(
            "/v1/chat/completions", "v1_chat", self._v1_chat, methods=["POST"]
        )
        self.app.run(host="0.0.0.0", port=self.port, debug=False, threaded=False)

    def _load_model(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model from {self.model_name_or_path}")
        logger.info(f"Using device: {self.device}")
        model = DexboticForCausalLM.from_pretrained(
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
        logger.info("Model loaded successfully")

    def _get_response(self, text: str, images: list[str]) -> str:
        t0 = time.monotonic()
        if len(images) == 1:
            images = [Image.open(images[0]).convert("RGB")]
            image_tensor = self.model.process_images(images).to(dtype=self.model.dtype)
        else:
            images = [Image.open(image).convert("RGB") for image in images]
            image_tensor = (
                self.model.process_images(images)
                .to(dtype=self.model.dtype)
                .unsqueeze(0)
            )

        self._save_image(images, text)

        conv = conversation_lib.conv_templates[self.model_config.chat_template].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + text)
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

        logger.debug(f"input_ids: {input_ids}")
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                images=image_tensor,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                return_dict_in_generate=True,
                stopping_criteria=[stopping_criteria],
            )
            outputs = outputs.sequences[0, input_ids.shape[1] :]
            outputs = self.tokenizer.decode(outputs, skip_special_tokens=False)
            outputs = outputs.strip(stop_str)

        logger.info(f"prompt: <start>{prompt}<end>\noutput: {outputs}")
        logger.info(f"Processing time: {time.monotonic() - t0}")
        return outputs

    def _save_image(self, images: list[str], text: str) -> None:
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
        # save image
        for idx, image in enumerate(images):
            image.save(
                os.path.join(save_image_dir_episode, f"{self.timestep}_{idx}.png")
            )
        # save text
        if self.timestep == 0:
            with open(os.path.join(save_image_dir_episode, "text.txt"), "w") as f:
                f.write(text)

    def _build_policy(self):
        """Return a BasePolicy for this model, or None to stay in legacy mode."""
        return None

    def _initialize_inference(self) -> None:
        self._load_model()
        self.prev_prompt = None
        self.timestep = 0
        self.episode = 0

        if self.norm_stats is None:
            norm_stats_file = os.path.join(self.model_name_or_path, "norm_stats.json")
            self.norm_stats = self.read_normalization_stats(norm_stats_file)
        elif isinstance(self.norm_stats, str):
            self.norm_stats = self.read_normalization_stats(self.norm_stats)
        logger.info(f"Normalization stats: {self.norm_stats}")

        self.policy = self._build_policy()

    def read_normalization_stats(self, action_norm_file):
        logger.info(f"Reading normalization stats from {action_norm_file}")
        if action_norm_file is None or not megfile.smart_exists(action_norm_file):
            return {"min": -1, "max": 1}
        with megfile.smart_open(action_norm_file, "r") as f:
            norm_stats = json.load(f)
            if "norm_stats" in norm_stats:
                norm_stats = norm_stats["norm_stats"]
            norm_stats = norm_stats["default"]
        return norm_stats

    # ── v1 route handlers ────────────────────────────────────────────────────

    def _v1_health(self):
        return jsonify({"status": "ok"})

    def _v1_models(self):
        model_id = os.path.basename(self.model_name_or_path or "unknown")
        return jsonify(
            {
                "object": "list",
                "data": [{"id": model_id, "object": "model"}],
            }
        )

    def _v1_capabilities(self):
        # camera_order has len == num_images; None entries are always zero-padded.
        # Public API slots are 1-based; policy observations remain 0-based image/{i}.
        # Only non-None slots require an image key from the caller.
        policy_caps = (
            self.policy.get_capabilities()
            if getattr(self, "policy", None) is not None
            else {}
        )
        state_caps = policy_caps.get("state", {})
        history_caps = policy_caps.get("history", {})
        slots = [
            {"slot": i + 1, "name": name, "required": name is not None}
            for i, name in enumerate(self.camera_order)
        ]
        modalities = {
            "images": {
                "slots": slots,
                "format": "image/{slot_index}",
            },
            "state": {
                "used": state_caps.get("used", False),
                "required": state_caps.get("required", False),
                "dim": state_caps.get("dim", None),
            },
            "prompt": {"required": True},
        }
        if history_caps:
            modalities["history_images"] = history_caps
        return jsonify(
            {
                "model_family": self.__class__.__name__,
                "vla": policy_caps.get(
                    "vla", getattr(self, "policy", None) is not None
                ),
                "vlm": policy_caps.get("vlm", False),
                "modalities": modalities,
                "action_spec": {
                    "action_dim": getattr(self, "action_dim", None),
                    "chunk_size": getattr(self, "action_horizon", None),
                    "action_mode": policy_caps.get("action_mode", "unknown"),
                },
                "max_batch_size": policy_caps.get("max_batch_size", 1),
                "sampling_defaults": {
                    "num_steps": getattr(self, "num_inference_steps", 10),
                    "cfg_scale": getattr(self, "cfg_scale", 1.0),
                },
            }
        )

    def _v1_infer(self):
        if getattr(self, "policy", None) is None:
            return jsonify({"error": "no policy configured"}), 501
        try:
            body = request.get_json(force=True)
            if not isinstance(body, dict):
                raise ValueError("request body must be a JSON object")
            obs_raw = body.get("observation", {})
            obs = self._decode_observation(obs_raw)
        except (TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        if self._state_required() and "state" not in obs_raw:
            return jsonify({"error": "observation must contain state"}), 400
        if not any(k.startswith("image/") for k in obs):
            return (
                jsonify({"error": "observation must contain at least one image"}),
                400,
            )
        sampling_raw = body.get("sampling") or {}
        if sampling_raw and not isinstance(sampling_raw, dict):
            return jsonify({"error": "sampling must be a JSON object"}), 400
        supported_sampling_keys = {f.name for f in fields(SamplingConfig)}
        sampling = {
            k: v for k, v in sampling_raw.items() if k in supported_sampling_keys
        }
        sc = SamplingConfig(**sampling) if sampling_raw else None
        if sc is not None:
            setattr(sc, "_provided_fields", set(sampling.keys()))
        if sc is not None:
            self._apply_inference_seed(sc.seed)
        t0 = time.monotonic()
        out = self.policy.select_action(obs, sc)[0]
        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        return jsonify(
            {
                "actions": out.actions.tolist(),
                "metadata": {"latency_ms": latency_ms},
            }
        )

    def _v1_reset(self):
        if getattr(self, "policy", None) is not None:
            self.policy.reset()
        return jsonify({"status": "ok"})

    def _v1_chat(self):
        if getattr(self, "policy", None) is None or not self.policy.supports_vlm():
            return jsonify({"error": "model does not support VLM generation"}), 501
        body = request.get_json(force=True)
        obs = self._decode_chat_messages(body.get("messages", []))
        sc = GenSamplingConfig(
            max_new_tokens=body.get("max_tokens", 128),
            temperature=body.get("temperature", 1.0),
            top_p=body.get("top_p", 0.95),
            do_sample=body.get("temperature", 1.0) != 0,
        )
        out = self.policy.generate(obs, sc)
        return jsonify(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "model": body.get("model", "unknown"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": out.text},
                        "finish_reason": out.finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": len(out.tokens),
                    "total_tokens": len(out.tokens),
                },
            }
        )

    def _decode_observation(self, obs_raw: dict) -> dict:
        """Convert v1/infer HTTP payload → policy observation dict (single sample, non-list values).

        Public image keys are 1-based to match dexdata images_1/images_2/...
        Policy observation keys are still internal 0-based image/0, image/1, ...
          new format: {"images": {"1": b64, "2": b64}}
          legacy compat: top-level image_1, image_2, ... keys
        """
        obs = {}

        if not isinstance(obs_raw, dict):
            raise ValueError("observation must be a JSON object")

        if "images" in obs_raw and isinstance(obs_raw["images"], dict):
            # new format: numeric string keys "1", "2", ...
            for slot_str, b64 in sorted(
                obs_raw["images"].items(), key=lambda x: self._parse_image_slot(x[0])
            ):
                external_slot = self._parse_image_slot(slot_str)
                internal_slot = external_slot - 1
                obs[f"image/{internal_slot}"] = self._decode_b64_image(
                    b64, f"images.{slot_str}"
                )
        else:
            # legacy compat: image_1, image_2, ... → image/0, image/1, ...
            import re

            old_keys = sorted(
                (k for k in obs_raw if re.fullmatch(r"image_\d+", k)),
                key=lambda k: int(k.split("_")[1]),
            )
            for key in old_keys:
                external_slot = self._parse_image_slot(key.split("_")[1])
                internal_slot = external_slot - 1
                obs[f"image/{internal_slot}"] = self._decode_b64_image(
                    obs_raw[key], key
                )

        history_images = obs_raw.get("history_images")
        if history_images is not None:
            if not isinstance(history_images, list):
                raise ValueError(
                    "observation.history_images must be a JSON array of "
                    "base64 images"
                )
            obs["history_images"] = [
                self._decode_b64_image(value, f"history_images[{index}]")
                for index, value in enumerate(history_images)
            ]

        state = obs_raw.get("state")
        if state is not None:
            obs["state"] = np.array(state, dtype=np.float32)
        obs["prompt"] = obs_raw.get("prompt", "")
        return obs

    def _state_required(self) -> bool:
        if getattr(self, "policy", None) is None:
            return False
        return bool(
            self.policy.get_capabilities().get("state", {}).get("required", False)
        )

    def _parse_image_slot(self, slot) -> int:
        try:
            slot = int(slot)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"image slot must be a numeric string, got {slot!r}"
            ) from exc
        if slot < 1:
            raise ValueError(f"image slot must be 1-based and positive, got {slot}")
        return slot

    def _decode_b64_image(self, value, field: str) -> Image.Image:
        if not isinstance(value, str):
            raise ValueError(f"{field} must be a base64 string")
        try:
            raw = base64.b64decode(value, validate=True)
            return Image.open(BytesIO(raw)).convert("RGB")
        except Exception as exc:
            raise ValueError(f"{field} is not a valid base64 image") from exc

    def _decode_chat_messages(self, messages: list) -> dict:
        """Convert OpenAI chat messages → policy observation dict."""
        images, text = [], ""
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                text = content
            else:
                for part in content:
                    if part.get("type") == "text":
                        text = part["text"]
                    elif part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        if url.startswith("data:"):
                            b64 = url.split(",", 1)[1]
                            img = Image.open(BytesIO(base64.b64decode(b64))).convert(
                                "RGB"
                            )
                            images.append(img)
        return {"prompt": text, "images": images}


@dataclass
class BaseExp(Config):
    """
    Base experiment class - integrates all configurations and executes training/inference

    Configuration details:
    This class contains instances of all configurations, integrating model, optimizer, trainer, data, and tokenizer configurations
    """

    model_config: ModelConfig = field(default_factory=ModelConfig)
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    trainer_config: TrainerConfig = field(default_factory=TrainerConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    tokenizer_config: TokenizerConfig = field(default_factory=TokenizerConfig)

    logger_level: str = field(default="INFO")
    backend_resolution: Optional[BackendResolution] = field(
        default=None, init=False, repr=False
    )

    def _log_backend_startup_summary(self) -> None:
        if self.local_rank != 0 or self.backend_resolution is None:
            return

        resolution = self.backend_resolution
        normalized_fsdp_config = resolution.normalized_fsdp_config or {}
        fsdp_version = normalized_fsdp_config.get(
            "fsdp_version", normalized_fsdp_config.get("version")
        )
        package_versions = resolution.package_versions

        logger.info(
            "Backend startup summary: requested_backend={}, resolved_mode={}, fsdp_version={}, plugin_kwargs={}, torch={}, transformers={}, accelerate={}",
            resolution.requested_backend,
            resolution.resolved_mode,
            fsdp_version,
            resolution.plugin_kwargs,
            package_versions.torch if package_versions else torch.__version__,
            package_versions.transformers
            if package_versions
            else transformers.__version__,
            package_versions.accelerate if package_versions else "unknown",
        )

    def _validate_train_backend(self) -> None:
        self.trainer_config.ensure_backend_defaults()
        self.backend_resolution = resolve_backend_mode(self.trainer_config)

        if self.local_rank == 0:
            logger.info(
                "Resolved training backend: requested_backend={}, resolved_mode={}, framework={}",
                self.backend_resolution.requested_backend,
                self.backend_resolution.resolved_mode,
                self.backend_resolution.framework,
            )
            self._log_backend_startup_summary()
            for warning in self.backend_resolution.warnings:
                logger.warning("Backend resolver: {}", warning)

    def _is_fsdp_compatible_checkpoint(self, checkpoint_dir: str) -> bool:
        """Check if checkpoint is compatible with FSDP resume.

        Both SHARDED_STATE_DICT (pytorch_model_fsdp_*/) and
        FULL_STATE_DICT (model.safetensors / model-*.safetensors) layouts
        can be resumed by HF Trainer under FSDP.
        """
        try:
            from transformers.trainer import FSDP_MODEL_NAME
        except ImportError:
            FSDP_MODEL_NAME = "pytorch_model_fsdp"

        checkpoint_path = pathlib.Path(checkpoint_dir)
        if not checkpoint_path.is_dir():
            return False

        for child in checkpoint_path.iterdir():
            # SHARDED_STATE_DICT layout
            if child.is_dir() and FSDP_MODEL_NAME in child.name:
                return True
            # FULL_STATE_DICT layout (safetensors or bin)
            if child.name == "model.safetensors" or child.name == "pytorch_model.bin":
                return True
            if child.name.startswith("model-") and child.name.endswith(".safetensors"):
                return True
        return (checkpoint_path / f"{FSDP_MODEL_NAME}.bin").is_file()

    def _resolve_auto_resume_checkpoint(self) -> Optional[str]:
        output_dir = self.trainer_config.output_dir
        if output_dir is None or not os.path.isdir(output_dir):
            return None

        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is None:
            return None

        if (
            self.backend_resolution is not None
            and self.backend_resolution.is_fsdp_backend
            and not self._is_fsdp_compatible_checkpoint(last_checkpoint)
        ):
            logger.warning(
                "Skipping auto-resume for FSDP backend because checkpoint {} has no recognizable model weights.",
                last_checkpoint,
            )
            return None

        return last_checkpoint

    def _patch_peft_resume_tensor_parallel_import(self) -> None:
        try:
            import transformers.integrations.tensor_parallel as tensor_parallel
        except Exception:
            return

        if hasattr(tensor_parallel, "EmbeddingParallel"):
            return

        class EmbeddingParallel:
            pass

        tensor_parallel.EmbeddingParallel = EmbeddingParallel
        if self.local_rank == 0:
            logger.info(
                "Patched missing transformers.integrations.tensor_parallel.EmbeddingParallel for PEFT resume compatibility"
            )

    def _log_fsdp_runtime_state(self) -> None:
        if self.local_rank != 0:
            return

        import accelerate

        logger.info("torch: {}", torch.__version__)
        logger.info("transformers: {}", transformers.__version__)
        logger.info("accelerate: {}", accelerate.__version__)

        if self.backend_resolution is not None:
            logger.info(
                "backend_resolution = {}",
                {
                    "requested_backend": self.backend_resolution.requested_backend,
                    "resolved_mode": self.backend_resolution.resolved_mode,
                    "framework": self.backend_resolution.framework,
                },
            )

        logger.info(
            "is_fsdp_enabled = {}",
            getattr(self.trainer, "is_fsdp_enabled", None),
        )
        logger.info(
            "distributed_type = {}",
            self.trainer.accelerator.distributed_type,
        )

        fsdp_plugin = getattr(self.trainer.accelerator.state, "fsdp_plugin", None)
        logger.info("fsdp_plugin = {}", fsdp_plugin)
        if fsdp_plugin is not None:
            logger.info("fsdp_version = {}", getattr(fsdp_plugin, "fsdp_version", None))

        try:
            from torch.distributed._tensor import DTensor

            has_dtensor = any(
                isinstance(param, DTensor) for param in self.trainer.model.parameters()
            )
            logger.info("has DTensor param = {}", has_dtensor)
        except Exception as e:
            logger.info("DTensor check failed: {}", e)
            return

        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP1
        except Exception as e:
            logger.info("FSDP wrapper check failed: {}", e)
            return

        logger.info(
            "is FSDP1 wrapper = {}",
            isinstance(self.trainer.model, FSDP1),
        )
        logger.info("has reshard = {}", hasattr(self.trainer.model, "reshard"))
        logger.info("has unshard = {}", hasattr(self.trainer.model, "unshard"))
        logger.info("model type = {}", type(self.trainer.model))

    def _apply_fsdp_model_dtype(self) -> None:
        profile = self.trainer_config.fsdp_profile
        if self.trainer_config.train_backend not in profile.cast_model_to_bf16_backends:
            return
        if self.local_rank == 0:
            logger.info(
                "Casting model to bfloat16 for {} backend",
                self.trainer_config.train_backend,
            )
        self.model.to(dtype=torch.bfloat16)

    def _training_runtime_model(self):
        """Return the underlying VLM when the trainable model is PEFT-wrapped."""

        model = self.model
        if hasattr(model, "get_base_model"):
            try:
                return model.get_base_model()
            except Exception:
                pass

        base_model = getattr(model, "base_model", None)
        candidate = getattr(base_model, "model", None)
        if candidate is not None:
            return candidate

        candidate = getattr(model, "model", None)
        if candidate is not None and hasattr(candidate, "model"):
            return candidate

        return model

    def _set_training_use_cache(self, enabled: bool) -> None:
        runtime_model = self._training_runtime_model()
        if hasattr(runtime_model, "config"):
            runtime_model.config.use_cache = enabled

        inner_model = getattr(runtime_model, "model", None)
        llm = getattr(inner_model, "llm", None)
        if llm is not None and hasattr(llm, "config"):
            llm.config.use_cache = enabled

    def _training_image_processor(self):
        runtime_model = self._training_runtime_model()
        inner_model = getattr(runtime_model, "model", None)
        vision_module = getattr(inner_model, "mm_vision_module", None)
        if vision_module is None:
            vision_module = getattr(inner_model, "mm_vision_tower", None)
        if vision_module is None or not hasattr(vision_module, "image_processor"):
            raise AttributeError(
                "Unable to locate image_processor on the training model"
            )
        return vision_module.image_processor

    def _initialize_train(self):
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        logger.info(f"Local rank: {self.local_rank}")
        if self.local_rank != 0:
            logger.remove()
            logger.add(lambda msg: None)

        self._validate_train_backend()

        # Step 0: compute norm stats
        self._auto_compute_norm_stats()

        # Step 1: build tokenizer
        tokenizer_kwargs = {
            "model_max_length": self.trainer_config.model_max_length,
            "padding_side": "right",
            "use_fast": self.tokenizer_config.use_fast_tokenizer,
        }
        tokenizer = self.tokenizer_config.build_tokenizer(
            self.model_config.model_name_or_path, **tokenizer_kwargs
        )
        self.tokenizer = tokenizer

        # Step 2: build model
        model = self.model_config.build_model()
        self.model = model
        self.tokenizer = self.tokenizer_config.add_special_tokens(
            self.data_config.action_config.string_format,
            self.data_config.action_config.vocab_size,
            self.tokenizer,
            self.model,
        )
        self._set_training_use_cache(False)

        # Step 3: build dataloader
        train_dataset, data_collator = self.data_config.build_data(
            self.tokenizer,
            self.model_config.chat_template,
            self._training_image_processor(),
        )

        # step 4: build trainer
        trainer_kwargs = {
            "model": self.model,
            "processing_class": self.tokenizer,
            "exp_config": self,
            "train_dataset": train_dataset,
            "data_collator": data_collator,
        }
        trainer = DexboticTrainer(**trainer_kwargs)
        self.trainer = trainer
        self._log_fsdp_runtime_state()

        # step 5: save action norm config
        if self.local_rank == 0 and hasattr(
            train_dataset.action_process_func, "statistic_mapping"
        ):
            logger.info(
                f"Saving action norm config to {self.trainer_config.output_dir}/norm_stats.json"
            )
            os.makedirs(self.trainer_config.output_dir, exist_ok=True)
            action_norm_config = train_dataset.action_process_func.statistic_mapping
            with open(
                os.path.join(self.trainer_config.output_dir, "norm_stats.json"), "w"
            ) as f:
                json.dump(action_norm_config, f, indent=2, cls=NumpyEncoder)

        self._apply_fsdp_model_dtype()

    def _auto_compute_norm_stats(self) -> None:
        if (
            not self.data_config.auto_norm
            or self.data_config.action_config.statistic_mapping is not None
        ):
            return
        norm_config = ComputeNormActionConfig(
            delta=self.data_config.action_config.delta,
            norm_method=self.data_config.auto_norm_method,
        )
        save_name = hashlib.md5(self.data_config.dataset_name.encode()).hexdigest()[:8]
        norm_config.norm_save_path = os.path.join(
            os.path.dirname(norm_config.norm_save_path), save_name
        )
        norm_file_path = os.path.join(norm_config.norm_save_path, "norm_stats.json")
        if self.local_rank == 0 and not megfile.smart_exists(norm_file_path):
            logger.info("Auto-computing norm stats on rank0")
            norm_config.compute_norm_stats(self.data_config.dataset_name)
        else:
            while not megfile.smart_exists(norm_file_path):
                time.sleep(5)
                print(
                    f"Waiting for norm stats: {norm_file_path} to be computed on rank{self.local_rank}"
                )
        self.data_config.action_config.statistic_mapping = norm_file_path

    def __post_init__(self):
        if self.trainer_config.debug_mode:
            enter_debug_mode(enable=True)
            self.trainer_config.dataloader_num_workers = 1
            self.logger_level = "DEBUG"
        logger.remove()
        logger.add(sys.stdout, level=self.logger_level)

    def train(self):
        self._initialize_train()

        try:
            resume_checkpoint = self._resolve_auto_resume_checkpoint()
            if resume_checkpoint is not None:
                logger.info("Resuming training from checkpoint {}", resume_checkpoint)
                self._patch_peft_resume_tensor_parallel_import()
                self.trainer.train(resume_from_checkpoint=resume_checkpoint)
            else:
                self.trainer.train()

            self.trainer.save_state()
            self._set_training_use_cache(True)
            safe_save_model_for_hf_trainer(
                trainer=self.trainer, output_dir=self.trainer_config.output_dir
            )
            logger.info(
                f"Training completed and model saved to {self.trainer_config.output_dir}"
            )
        finally:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                try:
                    torch.distributed.destroy_process_group()
                except Exception as exc:
                    logger.warning("Failed to destroy process group cleanly: {}", exc)


if __name__ == "__main__":
    exp = BaseExp()
    exp.train()
