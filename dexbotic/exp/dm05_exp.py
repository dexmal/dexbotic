import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

import megfile
import numpy as np
import torch
from easydict import EasyDict
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoProcessor

import dexbotic.data.utils.normalize as normalize
from dexbotic.data.dataset.dex_dataset import DexDataset
from dexbotic.data.dataset.dm05_data import (
    DM05ActionNorm,
    DM05DataCollator,
    DM05ImagePreprocess,
    DM05Tokenization,
    DM05ToTensor,
)
from dexbotic.data.dataset.rgb_preprocess import DummyRGBProcessor
from dexbotic.data.dataset.tokenization import DummyTokenization
from dexbotic.data.dataset.transform.action import AddTrajectory, PadAction, PadState
from dexbotic.data.dataset.transform.common import Pipeline, ToDict, ToList, ToNumpy
from dexbotic.data.dataset.transform.multimodal import LoadMultiModal
from dexbotic.data.dataset.transform.output import ActionDenorm
from dexbotic.exp.base_exp import (
    OPENAI_CLIP_PATH,
    ActionConfig,
    BaseExp,
    ComputeNormActionConfig,
    DataConfig,
    FSDPProfile,
)
from dexbotic.exp.base_exp import InferenceConfig as BaseInferenceConfig
from dexbotic.exp.base_exp import ModelConfig, OptimizerConfig, TrainerConfig
from dexbotic.exp.trainer import DexboticTrainer, safe_save_model_for_hf_trainer
from dexbotic.exp.utils import NumpyEncoder
from dexbotic.model.dm05 import (
    DM05Config,
    DM05ForConditionalGeneration,
    apply_lora_to_dm05_model,
    load_dm05_model_for_inference,
    unwrap_dm05_model,
)
from dexbotic.model.dm05.dm05_lora import (
    _is_lora_checkpoint,
    _read_lora_base_model_path,
)
from dexbotic.policy.dm05_policy import DM05Policy


class DM05Trainer(DexboticTrainer):
    def _save_checkpoint(self, model, trial, metrics=None) -> None:
        super()._save_checkpoint(model, trial, metrics)
        if not self.exp_config.trainer_config.save_hf_sidecar:
            return
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

        run_dir = self._get_output_dir(trial=trial)
        hf_dir = os.path.join(
            run_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}-hf"
        )
        pre_marker = os.path.join(hf_dir, "preprocessor_config.json")
        weights_done = (
            os.path.isfile(os.path.join(hf_dir, "model.safetensors"))
            and os.path.isfile(os.path.join(hf_dir, "config.json"))
        ) or (
            os.path.isfile(os.path.join(hf_dir, "adapter_model.safetensors"))
            and os.path.isfile(os.path.join(hf_dir, "adapter_config.json"))
        )
        if weights_done and os.path.isfile(pre_marker):
            if getattr(self, "accelerator", None) is not None:
                self.accelerator.wait_for_everyone()
            return
        os.makedirs(hf_dir, exist_ok=True)
        if not weights_done:
            safe_save_model_for_hf_trainer(self, hf_dir)
        if self.args.local_rank == 0 or self.args.local_rank == -1:
            processing = getattr(self, "processing_class", None)
            if processing is not None:
                processing.save_pretrained(hf_dir)
            proc_marker = os.path.join(hf_dir, "processor_config.json")
            if not os.path.isfile(pre_marker) and os.path.isfile(proc_marker):
                img = json.load(open(proc_marker)).get("image_processor", {})
                with open(pre_marker, "w") as f:
                    json.dump(img, f, indent=2)
                    f.write("\n")
            self._copy_norm_stats_to_checkpoint(hf_dir)
        if getattr(self, "accelerator", None) is not None:
            self.accelerator.wait_for_everyone()


@dataclass
class DM05ModelConfig(ModelConfig):
    model_name_or_path: str = field(default="./checkpoints/DM05")
    chunk_size: int = field(default=10)
    bf16: bool = field(default=True)
    llm_attn_implementation: Literal["auto", "eager", "sdpa", "flex_attention"] = field(
        default="flex_attention"
    )
    vision_attn_implementation: Literal[
        "auto", "eager", "sdpa", "flash_attention_2"
    ] = field(default="flash_attention_2")
    action_attn_implementation: Literal[
        "auto", "eager", "sdpa", "flex_attention"
    ] = field(default="sdpa")
    liger_kernel: bool = field(default=True)
    freeze_vlm_embedding: bool = field(default=True)
    vlm_gradient_checkpointing: bool = field(default=False)
    ae_gradient_checkpointing: bool = field(default=False)

    def build_model(self, use_lora: bool = False) -> torch.nn.Module:
        dtype = torch.bfloat16 if self.bf16 else torch.float32
        overrides = {"chunk_size": self.chunk_size}
        if _is_lora_checkpoint(self.model_name_or_path):
            model = load_dm05_model_for_inference(
                self.model_name_or_path,
                config_overrides=overrides,
                merge_and_unload=not use_lora,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
            if use_lora and hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        else:
            config = DM05Config.from_pretrained(self.model_name_or_path)
            config.chunk_size = self.chunk_size
            model = DM05ForConditionalGeneration.from_pretrained(
                self.model_name_or_path,
                config=config,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            if self.liger_kernel and torch.cuda.is_available():
                model._apply_liger_kernel()
            if use_lora:
                model = apply_lora_to_dm05_model(model, self.model_name_or_path or "")
        dm05_model = unwrap_dm05_model(model)
        dm05_model.set_attention_implementation(
            llm_attn_implementation=self.llm_attn_implementation,
            vision_attn_implementation=self.vision_attn_implementation,
            action_attn_implementation=self.action_attn_implementation,
            bf16=self.bf16,
        )
        dm05_model.enable_gradient_checkpointing(
            vlm_gradient_checkpointing=self.vlm_gradient_checkpointing,
            ae_gradient_checkpointing=self.ae_gradient_checkpointing,
        )
        if self.freeze_vlm_embedding:
            dm05_model.freeze_vlm_embedding()
        return model


@dataclass
class DM05OptimizerConfig(OptimizerConfig):
    base_lr: float = field(default=2.5e-5)
    adam_beta2: float = field(default=0.95)
    warmup_steps: int = field(default=1000)
    weight_decay: float = field(default=1e-10)


@dataclass
class DM05TrainerConfig(TrainerConfig):
    fsdp_profile: FSDPProfile = field(
        default_factory=lambda: FSDPProfile(
            enabled=True, cpu_ram_efficient_loading=False
        )
    )
    train_backend: str = field(default="fsdp2")
    model_max_length: int = field(default=768)
    bf16: bool = field(default=True)
    num_train_steps: int = field(default=50000)
    save_steps: int = field(default=10000)
    save_total_limit: int = field(default=20)
    save_hf_sidecar: bool = field(default=True)
    per_device_train_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=1)
    gradient_checkpointing: bool = field(default=False)
    dataloader_num_workers: int = field(default=4)
    logging_steps: int = field(default=1)
    lr_scheduler_type: str = field(default="cosine_with_min_lr")
    lr_scheduler_kwargs: dict = field(default_factory=lambda: {"min_lr_rate": 0.1})
    wandb_project: str = field(default="dm05_sft_libero")
    output_dir: str = field(
        default="./user_checkpoints/dexbotic/libero_dm05/libero-sft"
    )


@dataclass
class DM05ComputeNormActionConfig(ComputeNormActionConfig):
    def compute_norm_stats(self, dataset_name: str) -> None:
        self.norm_save_path = os.path.join(
            os.path.dirname(self.norm_save_path),
            hashlib.md5(dataset_name.encode()).hexdigest()[:8],
        )
        dataset_name_list = dataset_name.split("+")
        action_process_func = self.build_action_process_func()
        dataset_list = self._get_dataset(action_process_func, dataset_name_list)
        norm_files = {}
        for name, dataset in dataset_list:
            norm_file = self._process_one_dataset(name, dataset)
            norm_files[name] = (norm_file, dataset.dataset_map[0])
        self._merge_norm_stats(norm_files)

    def build_action_process_func(self) -> Pipeline:
        return Pipeline(
            [
                ToDict(),
                ToNumpy(),
                PadState(ndim=32, axis=-1),
                PadAction(ndim=32, axis=-1),
                AddTrajectory(trajectory_length=10, flatten=False, padding_mode="last"),
                ToList(),
            ]
        )

    def _get_dataset(self, action_process_func, dataset_name_list):
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

    def _process_one_dataset(self, dataset_name, dataset):
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=64)
        norm_keys = ["state", "action"]
        stats = {key: normalize.RunningStats() for key in norm_keys}
        for batch_idx, batch in tqdm(
            enumerate(dataloader), desc="Computing norm stats"
        ):
            if batch_idx > 500:
                break
            for key in norm_keys:
                values = batch[key].numpy()
                stats[key].update(values.reshape(-1, values.shape[-1]))
        norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
        save_path = os.path.join(self.norm_save_path, dataset_name)
        logger.info(f"Saving norm stats to {save_path}")
        normalize.save(save_path, norm_stats)
        return os.path.join(save_path, "norm_stats.json")

    def _merge_norm_stats(self, norm_files, norm_keys=["action", "state"]):
        norm_stats = {}
        for norm_key in norm_keys:
            min_list = []
            max_list = []
            for _, (norm_file, _) in norm_files.items():
                with open(norm_file, "r") as f:
                    stats = json.load(f)["norm_stats"][norm_key]
                min_list.append(stats["q01"])
                max_list.append(stats["q99"])
            norm_stats[norm_key] = {
                "min": np.array(min_list).min(axis=0).tolist(),
                "max": np.array(max_list).max(axis=0).tolist(),
            }
        with open(os.path.join(self.norm_save_path, "norm_stats.json"), "w") as f:
            json.dump({"norm_stats": norm_stats}, f, indent=2)


@dataclass
class DM05ActionConfig(ActionConfig):
    trajectory_length: int = field(default=10)

    def _quantile_minmax(self, stats: dict, pad_dim: int = 32) -> dict:
        if "q01" in stats and "q99" in stats:
            min_vals = np.array(stats["q01"], dtype=np.float32)
            max_vals = np.array(stats["q99"], dtype=np.float32)
        else:
            min_vals = np.array(stats["min"], dtype=np.float32)
            max_vals = np.array(stats["max"], dtype=np.float32)
        if min_vals.shape[-1] < pad_dim:
            pad = pad_dim - min_vals.shape[-1]
            min_vals = np.pad(min_vals, (0, pad), constant_values=0.0)
            max_vals = np.pad(max_vals, (0, pad), constant_values=1.0)
        return {"min": min_vals, "max": max_vals}

    def _read_norm_stats(self, norm_stats_path):
        if not norm_stats_path or not megfile.smart_exists(norm_stats_path):
            raise FileNotFoundError(f"Norm stats file {norm_stats_path} not found")
        with megfile.smart_open(norm_stats_path, "r") as f:
            raw = json.load(f)
        norm_stats = raw["norm_stats"] if "norm_stats" in raw else raw
        action_stats = (
            norm_stats["action"] if "action" in norm_stats else norm_stats["default"]
        )
        mapping = {"action": self._quantile_minmax(action_stats)}
        if "state" in norm_stats:
            mapping["state"] = self._quantile_minmax(norm_stats["state"])
        return ToNumpy()(mapping)

    def build_action_process_func(self) -> Pipeline:
        statistic_mapping = self._read_norm_stats(self.statistic_mapping)
        return Pipeline(
            [
                ToDict(),
                ToNumpy(),
                PadState(ndim=32, axis=-1),
                PadAction(ndim=32, axis=-1),
                AddTrajectory(
                    trajectory_length=self.trajectory_length,
                    flatten=False,
                    padding_mode="last",
                ),
                DM05ActionNorm(statistic_mapping=statistic_mapping, use_quantiles=True),
                LoadMultiModal(return_masks=True),
                ToList(),
            ]
        )


@dataclass
class DM05DataConfig(DataConfig):
    dataset_name: str = field(default="libero_pi0_all")
    num_images: int = field(default=2)
    data_keys: list[str] = field(
        default_factory=lambda: ["input_ids", "labels", "action", "image", "state"]
    )
    aug_policy: str | list[str] = field(default=None)
    image_aspect_ratio: str = field(default=None)
    valid_action_dim: int = field(default=7)
    tokenizer_max_length: int = field(default=768)
    action_config: DM05ActionConfig = field(default_factory=DM05ActionConfig)

    def build_data(self, processor: AutoProcessor, max_length: int | None = None):
        image_preprocess = DM05ImagePreprocess()
        dataset = DexDataset(
            data_args=EasyDict(
                {
                    "dataset_name": self.dataset_name,
                    "num_images": self.num_images,
                    "data_keys": self.data_keys,
                    "images_keys": self.images_keys,
                    "aug_policy": self.aug_policy,
                    "image_aspect_ratio": self.image_aspect_ratio,
                    "image_processor": getattr(processor, "image_processor", processor),
                }
            ),
            tokenization_func=DM05Tokenization(),
            action_process_func=self.action_config.build_action_process_func(),
            image_process_func=[image_preprocess for _ in range(self.num_images)],
        )
        collator = DM05DataCollator(
            processor=processor,
            max_length=max_length or self.tokenizer_max_length,
            valid_action_dim=self.valid_action_dim,
            model_action_dim=32,
            chunk_size=self.action_config.trajectory_length,
        )
        return dataset, collator


@dataclass
class DM05InferenceConfig(BaseInferenceConfig):
    num_images: int = field(default=2)
    action_dim: int = field(default=7)
    model_action_dim: int = field(default=32)
    chunk_size: int = field(default=10)
    diffusion_steps: int = field(default=10)
    model_max_length: int | None = field(default=None)
    history_enabled: bool = field(default=False)
    max_history_images: int = field(default=5)
    llm_attn_implementation: str = field(default="eager")
    camera_order: list = field(default_factory=lambda: ["agentview", "wrist"])
    backend: Literal["default", "fast"] = field(default="default")
    vision_trt_engine_path: str = field(
        default="checkpoints/trt_engines/dm05_vision.engine"
    )
    build_engine_if_missing: bool = field(default=True)
    force_rebuild_engine: bool = field(default=False)
    prefix_seq_len_buckets: list[int] = field(
        default_factory=lambda: [576, 704, 768, 896, 1024]
    )
    fast_overflow_policy: Literal["error", "fallback"] = field(default="fallback")
    fast_prefix_qkv_mode: Literal["packed", "separate"] = field(default="packed")

    @property
    def action_horizon(self) -> int:
        return self.chunk_size

    def _load_model(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_dm05_model_for_inference(
            self.model_name_or_path,
            config_overrides={"chunk_size": self.chunk_size},
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model = unwrap_dm05_model(model)
        if self.backend == "default":
            model.set_attention_implementation(
                llm_attn_implementation=self.llm_attn_implementation,
                vision_attn_implementation="sdpa",
                action_attn_implementation="sdpa",
                bf16=torch.cuda.is_available(),
            )
        model.to(self.device)
        model.eval()
        self.model = model
        processor_path = self.model_name_or_path
        if _is_lora_checkpoint(self.model_name_or_path):
            processor_path = (
                _read_lora_base_model_path(self.model_name_or_path) or processor_path
            )
        self.processor = AutoProcessor.from_pretrained(
            processor_path, trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer
        self.model_config = model.config
        self.input_transform = Pipeline(
            [
                PadState(ndim=self.model_action_dim, axis=-1),
                DM05ActionNorm(
                    statistic_mapping=self.norm_stats, strict=False, use_quantiles=True
                ),
                DM05ToTensor(),
            ]
        )
        self.output_transform = Pipeline(
            [
                ToNumpy(),
                ActionDenorm(
                    statistic_mapping=self.norm_stats, strict=False, use_quantiles=True
                ),
            ]
        )

    def _initialize_inference(self) -> None:
        if self.backend not in {"default", "fast"}:
            raise ValueError(f"Unsupported DM05 backend: {self.backend!r}")
        if self.backend == "fast":
            try:
                __import__("dexbotic_dm05_fast")
            except ImportError as exc:
                raise ImportError(
                    "DM05 fast inference requires the dexbotic-dm05-fast plugin. "
                    "Run 'bash plugins/dm05-fast/install.sh' in the active "
                    "Dexbotic environment."
                ) from exc
        if self.norm_stats is None:
            self.norm_stats = self.read_normalization_stats(
                os.path.join(self.model_name_or_path, "norm_stats.json")
            )
        self._load_model()
        self.policy = self._build_policy()
        if self.backend == "default":
            return

        from dexbotic.model.dm05.infer.backend import (
            DM05FastBackendConfig,
            build_dm05_runtime,
        )

        fast_config = DM05FastBackendConfig(
            vision_engine_path=self.vision_trt_engine_path,
            build_engine_if_missing=self.build_engine_if_missing,
            force_rebuild_engine=self.force_rebuild_engine,
            prefix_buckets=tuple(self.prefix_seq_len_buckets),
            overflow_policy=self.fast_overflow_policy,
            prefix_qkv_mode=self.fast_prefix_qkv_mode,
        )
        self.inference_runtime = build_dm05_runtime(
            policy=self.policy,
            checkpoint=self.model_name_or_path,
            num_images=self.num_images,
            diffusion_steps=self.diffusion_steps,
            fast_config=fast_config,
            history_enabled=self.history_enabled,
        )
        self.policy.inference_runtime = self.inference_runtime

    def _effective_max_history_images(self) -> int:
        if self.backend == "fast":
            from dexbotic.model.dm05.infer.fast.vision_trt import MAX_HISTORY_IMAGES

            return min(self.max_history_images, MAX_HISTORY_IMAGES)
        return self.max_history_images

    def _build_policy(self):
        return DM05Policy(
            model=self.model,
            processor=self.processor,
            norm_stats=self.norm_stats,
            input_pipeline=self.input_transform,
            output_pipeline=self.output_transform,
            device=self.device,
            num_images=self.num_images,
            action_dim=self.action_dim,
            model_action_dim=self.model_action_dim,
            chunk_size=self.chunk_size,
            diffusion_steps=self.diffusion_steps,
            model_max_length=self.model_max_length,
            camera_order=self.camera_order,
            history_enabled=self.history_enabled,
            max_history_images=self._effective_max_history_images(),
        )

    def read_normalization_stats(self, action_norm_file: str | None) -> dict:
        return DM05ActionConfig()._read_norm_stats(action_norm_file)

    def process_frame(self) -> None:
        from flask import jsonify, request

        self._apply_inference_seed(request.form.get("seed"))
        results = self._get_response(
            text=request.form.get("text", ""),
            images=request.files.getlist("image"),
            history_images=request.files.getlist("history_images"),
            states=request.form.get("states", None),
        )
        action = np.asarray(results, dtype=np.float64)
        expected = (self.chunk_size, self.action_dim)
        if action.shape != expected:
            raise ValueError(f"infer action shape {tuple(action.shape)} != {expected}")
        if not np.isfinite(action).all():
            raise ValueError("infer action contains non-finite values")
        return jsonify({"response": action.tolist()})

    def _get_response(
        self,
        text: str,
        images: list,
        history_images: list | None = None,
        states: Optional[str] = None,
    ) -> list[list[float]]:
        pil_images = [Image.open(img).convert("RGB") for img in images]
        pil_history_images = [
            Image.open(img).convert("RGB") for img in (history_images or [])
        ]
        if states is None:
            state = [0.0] * self.model_action_dim
        else:
            state = json.loads(states)
        obs = {
            "prompt": text,
            "state": state,
            "history_images": pil_history_images,
        }
        for i, pil in enumerate(pil_images):
            obs[f"image/{i}"] = pil
        action = self.policy.select_action(obs)[0].actions
        return action.tolist()


@dataclass
class DM05Exp(BaseExp):
    use_lora: bool = field(default=False)
    model_config: DM05ModelConfig = field(default_factory=DM05ModelConfig)
    optimizer_config: DM05OptimizerConfig = field(default_factory=DM05OptimizerConfig)
    trainer_config: DM05TrainerConfig = field(default_factory=DM05TrainerConfig)
    data_config: DM05DataConfig = field(default_factory=DM05DataConfig)
    inference_config: DM05InferenceConfig = field(default_factory=DM05InferenceConfig)

    def inference(self) -> None:
        self.inference_config.model_name_or_path = (
            self.inference_config.model_name_or_path
            or self.model_config.model_name_or_path
        )
        self.inference_config.run()

    def compute_norm_stats(self) -> None:
        self.data_config.action_config = self._build_compute_norm_config()
        self.data_config.action_config.compute_norm_stats(self.data_config.dataset_name)

    def _build_compute_norm_config(self):
        return DM05ComputeNormActionConfig()

    def _auto_compute_norm_stats(self) -> None:
        if (
            not self.data_config.auto_norm
            or self.data_config.action_config.statistic_mapping is not None
        ):
            return
        _action_config = self.data_config.action_config
        save_name = hashlib.md5(self.data_config.dataset_name.encode()).hexdigest()[:8]
        norm_file_path = os.path.join(
            os.path.dirname(self._build_compute_norm_config().norm_save_path),
            save_name,
            "norm_stats.json",
        )
        if int(
            os.environ.get("RANK", self.local_rank)
        ) == 0 and not megfile.smart_exists(norm_file_path):
            self.compute_norm_stats()
        else:
            deadline = time.monotonic() + 3600
            while not megfile.smart_exists(norm_file_path):
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Timed out waiting for norm stats: {norm_file_path}"
                    )
                time.sleep(5)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        _action_config.statistic_mapping = norm_file_path
        self.data_config.action_config = _action_config

    def _set_training_use_cache(self, enabled: bool) -> None:
        dm05_model = unwrap_dm05_model(self.model)
        dm05_model.config.use_cache = enabled
        dm05_model.model.vlm.config.use_cache = enabled

    def _initialize_train(self):
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if self.local_rank != 0:
            logger.remove()
            logger.add(lambda msg: None)
        self._validate_train_backend()
        self._auto_compute_norm_stats()
        processor_path = self.model_config.model_name_or_path
        if _is_lora_checkpoint(processor_path):
            processor_path = (
                _read_lora_base_model_path(processor_path) or processor_path
            )
        self.processor = AutoProcessor.from_pretrained(
            processor_path, trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer
        self.model = self.model_config.build_model(use_lora=self.use_lora)
        self._set_training_use_cache(False)
        train_dataset, data_collator = self.data_config.build_data(
            self.processor, max_length=self.trainer_config.model_max_length
        )
        self.trainer = DM05Trainer(
            model=self.model,
            processing_class=self.processor,
            exp_config=self,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        if self.local_rank == 0 and hasattr(
            getattr(train_dataset, "action_process_func", None), "statistic_mapping"
        ):
            os.makedirs(self.trainer_config.output_dir, exist_ok=True)
            with open(
                os.path.join(self.trainer_config.output_dir, "norm_stats.json"), "w"
            ) as f:
                json.dump(
                    {"norm_stats": train_dataset.action_process_func.statistic_mapping},
                    f,
                    indent=2,
                    cls=NumpyEncoder,
                )
        if self.trainer_config.bf16:
            self.model.to(dtype=torch.bfloat16)

    def train(self):
        self._initialize_train()
        try:
            resume_checkpoint = self._resolve_auto_resume_checkpoint()
            if resume_checkpoint is not None:
                self._patch_peft_resume_tensor_parallel_import()
                self.trainer.train(resume_from_checkpoint=resume_checkpoint)
            else:
                self.trainer.train()
            self.trainer.save_state()
            self._set_training_use_cache(True)
            safe_save_model_for_hf_trainer(
                trainer=self.trainer, output_dir=self.trainer_config.output_dir
            )
        finally:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
