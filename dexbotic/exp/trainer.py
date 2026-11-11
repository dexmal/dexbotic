import json
import math
import os
import re
import shutil
from typing import TYPE_CHECKING, Optional
from unittest.mock import patch

import torch
import transformers
from easydict import EasyDict
from loguru import logger
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset
from transformers import Trainer, TrainingArguments

from dexbotic.data.sampler import WeightedMultiDatasetSampler
from dexbotic.exp.backend_resolver import resolve_backend_mode
from dexbotic.exp.utils import get_mm_adapter_state_maybe_zero_3

if TYPE_CHECKING:
    from dexbotic.exp.base_exp import BaseExp


EXTRA_TRAINABLE_STATE_NAME = "extra_trainable_state.safetensors"
EXTRA_TRAINABLE_META_NAME = "extra_trainable_state.json"


def _lora_extra_trainable_config(exp_config) -> tuple[str | None, set[str]]:
    model_config = getattr(exp_config, "model_config", None)
    lora_config = getattr(model_config, "lora_config", None)
    if lora_config is None:
        return None, set()
    names = getattr(lora_config, "extra_trainable_names", None) or []
    return getattr(lora_config, "extra_trainable_regex", None), set(names)


def _save_lora_extra_trainable_state(
    model: torch.nn.Module,
    output_dir: str,
    extra_trainable_regex: str | None = None,
    extra_trainable_names: set[str] | None = None,
) -> None:
    """Persist full-rank LoRA companion weights that PEFT adapter saving ignores."""

    extra_trainable_names = extra_trainable_names or set()
    pattern = re.compile(extra_trainable_regex) if extra_trainable_regex else None
    if pattern is None and not extra_trainable_names:
        return

    state = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".lora_" in name:
            continue
        if ".modules_to_save." in name:
            continue
        if not ((pattern and pattern.search(name)) or name in extra_trainable_names):
            continue
        state[name] = parameter.detach().cpu()

    if not state:
        return

    os.makedirs(output_dir, exist_ok=True)
    state_path = os.path.join(output_dir, EXTRA_TRAINABLE_STATE_NAME)
    meta_path = os.path.join(output_dir, EXTRA_TRAINABLE_META_NAME)
    try:
        from safetensors.torch import save_file

        save_file(state, state_path)
    except Exception:
        torch.save(state, state_path + ".pt")
        state_path = state_path + ".pt"

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "extra_trainable_regex": extra_trainable_regex,
                "extra_trainable_names": sorted(extra_trainable_names),
                "state_file": os.path.basename(state_path),
                "parameter_count": len(state),
                "numel": int(sum(t.numel() for t in state.values())),
                "parameter_names": sorted(state),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(
        "Saved LoRA extra trainable full-rank state: {} tensors to {}",
        len(state),
        state_path,
    )


def _save_lora_extra_trainable_state_for_trainer(
    trainer: transformers.Trainer,
    output_dir: str,
) -> None:
    exp_config = getattr(trainer, "exp_config", None)
    if exp_config is None:
        return
    regex, names = _lora_extra_trainable_config(exp_config)
    _save_lora_extra_trainable_state(trainer.model, output_dir, regex, names)


class DexboticTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.exp_config: BaseExp = kwargs.pop("exp_config")
        self.backend_resolution = self.exp_config.backend_resolution
        if self.backend_resolution is None:
            self.backend_resolution = resolve_backend_mode(
                self.exp_config.trainer_config
            )
            self.exp_config.backend_resolution = self.backend_resolution
        self._explicit_fsdp_plugin_kwargs = self.backend_resolution.plugin_kwargs
        self._deepspeed_detection_patcher = None
        self._maybe_disable_deepspeed_detection()
        training_args = self._link_exp_config()
        super().__init__(*args, args=training_args, **kwargs)
        self.loss_cache = {}

    def _maybe_disable_deepspeed_detection(self):
        if self.backend_resolution.resolved_mode == "deepspeed_trainer":
            return

        from accelerate.utils import other as accelerate_other

        self._deepspeed_detection_patcher = patch.object(
            accelerate_other,
            "is_deepspeed_available",
            return_value=False,
        )
        self._deepspeed_detection_patcher.start()

    def __del__(self):
        patcher = getattr(self, "_deepspeed_detection_patcher", None)
        if patcher is not None:
            patcher.stop()

    def _build_explicit_fsdp_plugin(self):
        if not self._explicit_fsdp_plugin_kwargs:
            return None

        from accelerate.utils.dataclasses import FullyShardedDataParallelPlugin

        return FullyShardedDataParallelPlugin(**self._explicit_fsdp_plugin_kwargs)

    def _build_explicit_accelerator_dataloader_config(self):
        from accelerate.utils.dataclasses import DataLoaderConfiguration
        from transformers.trainer import logger as trainer_logger

        grad_acc_kwargs = {}
        if self.args.accelerator_config.gradient_accumulation_kwargs is not None:
            grad_acc_kwargs = self.args.accelerator_config.gradient_accumulation_kwargs

        if "num_steps" in grad_acc_kwargs:
            if self.args.gradient_accumulation_steps > 1:
                raise ValueError(
                    "The `AcceleratorConfig`'s `num_steps` is set but "
                    "`gradient_accumulation_steps` is greater than 1 in the passed `TrainingArguments`."
                )
            self.args.gradient_accumulation_steps = grad_acc_kwargs["num_steps"]

        accelerator_config = self.args.accelerator_config.to_dict()
        dataloader_params = [
            "split_batches",
            "dispatch_batches",
            "even_batches",
            "use_seedable_sampler",
        ]
        dataloader_config = DataLoaderConfiguration(
            **{param: accelerator_config.pop(param) for param in dataloader_params}
        )
        dataloader_config.data_seed = self.args.data_seed

        non_blocking = accelerator_config.pop("non_blocking")
        if non_blocking and not self.args.dataloader_pin_memory:
            trainer_logger.warning(
                "`non_blocking` is enabled but `dataloader_pin_memory` is not."
            )
        dataloader_config.non_blocking = non_blocking
        accelerator_config.pop("gradient_accumulation_kwargs")
        return dataloader_config

    def _build_explicit_accelerator_kwargs(self):
        args = {
            "deepspeed_plugin": self.args.deepspeed_plugin,
            "fsdp_plugin": self._build_explicit_fsdp_plugin(),
            "dataloader_config": self._build_explicit_accelerator_dataloader_config(),
        }

        if self.args.parallelism_config is not None:
            args["parallelism_config"] = self.args.parallelism_config

        if (
            hasattr(self.model, "tp_size")
            and self.model.tp_size is not None
            and self.model.tp_size > 1
        ):
            self.is_tp_enabled = True
            from accelerate import TorchTensorParallelPlugin

            args["torch_tp_plugin"] = TorchTensorParallelPlugin(
                tp_size=self.model.tp_size
            )

        return args

    def _postprocess_accelerator_state(self):
        import functools
        import inspect

        self.gather_function = self.accelerator.gather_for_metrics
        if "use_gather_object" in inspect.signature(self.gather_function).parameters:
            self.gather_function = functools.partial(
                self.gather_function,
                use_gather_object=self.args.eval_use_gather_object,
            )

        self.is_deepspeed_enabled = (
            getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        )
        self.is_fsdp_enabled = (
            getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        )
        self.is_tp_enabled = (
            getattr(self.accelerator.state, "torch_tp_plugin", None) is not None
        )

        if self.is_fsdp_enabled:
            fsdp_plugin = self.accelerator.state.fsdp_plugin
            for param in ["limit_all_gathers", "activation_checkpointing"]:
                setattr(
                    fsdp_plugin,
                    param,
                    self.args.fsdp_config.get(param, getattr(fsdp_plugin, param)),
                )
            if (
                fsdp_plugin.activation_checkpointing
                and self.args.gradient_checkpointing
            ):
                raise ValueError(
                    "The activation_checkpointing in FSDP config and gradient_checkpointing "
                    "in training args can't be set to True simultaneously."
                )

        if (
            self.is_deepspeed_enabled
            and getattr(self.args, "hf_deepspeed_config", None) is None
        ):
            self.propagate_args_to_deepspeed()

        if (
            self.args.save_only_model
            and (self.is_deepspeed_enabled or self.is_fsdp_enabled)
            and self.args.load_best_model_at_end
        ):
            wrapper = "DeepSpeed" if self.is_deepspeed_enabled else "FSDP"
            raise ValueError(
                f"{wrapper} can't be used with `save_only_model` along with `load_best_model_at_end`."
            )

        if (
            self.is_deepspeed_enabled
            and self.accelerator.state.deepspeed_plugin.zero_stage == 3
            and self.args.auto_find_batch_size
        ):
            raise ValueError(
                "`auto_find_batch_size` isn't supported yet with DeepSpeed Zero-3."
            )
        if (
            self.args.save_only_model
            and self.is_fsdp_enabled
            and "SHARDED_STATE_DICT"
            in str(self.accelerator.state.fsdp_plugin.state_dict_type)
        ):
            raise ValueError(
                "save_only_model option is not compatible with FSDP state dict type 'SHARDED_STATE_DICT'"
            )

    def _sync_fsdp_state_dict_type(self):
        """Sync state_dict_type from TrainingArguments.fsdp_config to the FSDP plugin.

        HF Trainer's native ``create_accelerator_and_postprocess`` creates the
        ``FullyShardedDataParallelPlugin`` without passing ``state_dict_type``,
        so the plugin falls back to its default (``SHARDED_STATE_DICT`` for
        FSDP2).  If the user explicitly configured ``state_dict_type`` in
        ``fsdp_config``, we apply it here after the accelerator is created.
        """
        if not self.is_fsdp_enabled:
            return
        desired = (self.args.fsdp_config or {}).get("state_dict_type")
        if desired is None:
            return
        fsdp_plugin = self.accelerator.state.fsdp_plugin
        fsdp_plugin.set_state_dict_type(desired)

    def create_accelerator_and_postprocess(self):
        if not self._explicit_fsdp_plugin_kwargs:
            super().create_accelerator_and_postprocess()
            self._sync_fsdp_state_dict_type()
            return

        from accelerate import Accelerator
        from transformers.trainer import is_accelerate_available

        if not is_accelerate_available("1.10.0"):
            raise ImportError(
                "Explicit FSDP2 plugin integration requires accelerate v1.10.0 and above."
            )

        self.accelerator = Accelerator(**self._build_explicit_accelerator_kwargs())
        self._postprocess_accelerator_state()

    def _get_train_sampler(self, train_dataset=None):
        weights = self.exp_config.data_config.dataset_weights
        if weights is None:
            return super()._get_train_sampler(train_dataset)
        dataset = self.train_dataset if train_dataset is None else train_dataset
        if not isinstance(dataset, ConcatDataset):
            raise ValueError("dataset_weights requires a ConcatDataset")
        return WeightedMultiDatasetSampler(
            dataset.datasets, weights, seed=self.args.seed
        )

    def create_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer is None:
            # FSDP2 rebinds these parameters to DTensors during prepare().
            self.optimizer = self.exp_config.optimizer_config.build_optimizer(
                self.model, self.args
            )
        return self.optimizer

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer = None
    ):
        use_raw_warmup = getattr(
            self.exp_config.trainer_config, "use_raw_warmup", False
        )

        if use_raw_warmup:
            if optimizer is None:
                optimizer = self.optimizer

            num_warmup_steps = self.args.warmup_steps
            min_lr_rate = self.exp_config.trainer_config.lr_scheduler_kwargs.get(
                "min_lr_rate", 0.1
            )

            def lr_lambda(current_step: int):
                if current_step < num_warmup_steps:
                    init_ratio = 1.0 / (num_warmup_steps + 1)
                    return (
                        init_ratio
                        + (1.0 - init_ratio) * current_step / num_warmup_steps
                    )

                progress = min(
                    1.0,
                    (current_step - num_warmup_steps)
                    / max(1, num_training_steps - num_warmup_steps),
                )
                cos = 0.5 * (1 + math.cos(math.pi * progress))
                return min_lr_rate + (1.0 - min_lr_rate) * cos

            self.lr_scheduler = LambdaLR(optimizer, lr_lambda)
            logger.info(
                f"Using native warmup scheduler: warmup_steps={num_warmup_steps}, min_lr_rate={min_lr_rate}"
            )
            return self.lr_scheduler
        else:
            return super().create_scheduler(num_training_steps, optimizer)

    def _save_checkpoint(self, model, trial, metrics=None) -> None:
        logger.info(f"Saving checkpoint at step {self.state.global_step}")
        if getattr(self.added_args, "tune_mm_mlp_adapter", False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ["mm_projector"]
            weight_to_save = get_mm_adapter_state_maybe_zero_3(
                self.model.named_parameters(), keys_to_match
            )

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, "mm_projector.bin"))

        else:
            super(DexboticTrainer, self)._save_checkpoint(model, trial)
            # Copy norm_stats.json to checkpoint directory after saving
            if self.args.local_rank == 0 or self.args.local_rank == -1:
                from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                run_dir = self._get_output_dir(trial=trial)
                output_dir = os.path.join(run_dir, checkpoint_folder)
                self._copy_norm_stats_to_checkpoint(output_dir)
                _save_lora_extra_trainable_state_for_trainer(self, output_dir)

    def _copy_norm_stats_to_checkpoint(self, checkpoint_dir: str) -> None:
        """Copy norm_stats.json from main output directory to checkpoint directory"""

        main_output_dir = self.args.output_dir
        norm_stats_src = os.path.join(main_output_dir, "norm_stats.json")
        norm_stats_dst = os.path.join(checkpoint_dir, "norm_stats.json")

        if os.path.exists(norm_stats_src):
            try:
                shutil.copy2(norm_stats_src, norm_stats_dst)
                logger.info(
                    f"Copied norm_stats.json to checkpoint directory: {checkpoint_dir}"
                )
            except Exception as e:
                logger.warning(f"Failed to copy norm_stats.json to checkpoint: {e}")

    def _save(self, output_dir: Optional[str] = None, state_dict=None) -> None:
        if getattr(self.added_args, "tune_mm_mlp_adapter", False):
            pass
        else:
            super(DexboticTrainer, self)._save(output_dir, state_dict)

    def _link_exp_config(self) -> TrainingArguments:
        """Link the exp config to the trainer args"""
        trainer_config = self.exp_config.trainer_config
        resolution = self.backend_resolution
        linked_args = {
            "output_dir": trainer_config.output_dir,
            "num_train_epochs": trainer_config.num_train_epochs,
            "max_steps": trainer_config.num_train_steps,
            "per_device_train_batch_size": trainer_config.per_device_train_batch_size,
            "gradient_accumulation_steps": trainer_config.gradient_accumulation_steps,
            "save_strategy": trainer_config.save_strategy,
            "save_steps": trainer_config.save_steps,
            "save_total_limit": trainer_config.save_total_limit,
            "save_only_model": trainer_config.save_only_model,
            "logging_steps": trainer_config.logging_steps,
            "gradient_checkpointing": trainer_config.gradient_checkpointing,
            "dataloader_num_workers": trainer_config.dataloader_num_workers,
            # "model_max_length": self.exp_config.trainer_config.model_max_length,
            "bf16": trainer_config.bf16,
            "tf32": trainer_config.tf32,
            "lr_scheduler_type": trainer_config.lr_scheduler_type,
            "lr_scheduler_kwargs": trainer_config.lr_scheduler_kwargs,
            "run_name": trainer_config.run_name,
            "remove_unused_columns": False,
            "learning_rate": self.exp_config.optimizer_config.base_lr,
            "adam_beta1": self.exp_config.optimizer_config.adam_beta1,
            "adam_beta2": self.exp_config.optimizer_config.adam_beta2,
            "warmup_steps": self.exp_config.optimizer_config.warmup_steps,
            "weight_decay": self.exp_config.optimizer_config.weight_decay,
            "seed": getattr(trainer_config, "seed", 42),
            "data_seed": getattr(trainer_config, "seed", 42),
        }
        if resolution.resolved_mode == "deepspeed_trainer":
            linked_args["deepspeed"] = trainer_config.deepspeed
            # DeepSpeed resume requires optimizer states saved via the DeepSpeed engine
            # (global_step* subdirectory). save_only_model=True skips that entirely.
            linked_args["save_only_model"] = False
        elif resolution.resolved_mode == "ddp_trainer":
            linked_args["save_only_model"] = trainer_config.save_only_model
        else:
            # FSDP checkpoints default to sharded state dicts in the current stack,
            # which is incompatible with save_only_model=True in Hugging Face Trainer.
            linked_args["save_only_model"] = False
            linked_args["fsdp"] = resolution.trainer_fsdp
            linked_args["fsdp_config"] = dict(resolution.trainer_fsdp_config or {})

        self.added_args = EasyDict(
            {
                "tune_mm_mlp_adapter": trainer_config.tune_mm_mlp_adapter,
                "train_backend": trainer_config.train_backend,
                "resolved_mode": resolution.resolved_mode,
            }
        )
        fsdp_config = linked_args.get("fsdp_config") or {}
        fsdp_config_summary = None
        if fsdp_config:
            fsdp_config_summary = {
                "fsdp_version": fsdp_config.get(
                    "fsdp_version", fsdp_config.get("version")
                ),
                "transformer_layer_cls_to_wrap": fsdp_config.get(
                    "transformer_layer_cls_to_wrap"
                ),
                "activation_checkpointing": fsdp_config.get("activation_checkpointing"),
                "cpu_ram_efficient_loading": fsdp_config.get(
                    "cpu_ram_efficient_loading"
                ),
                "reshard_after_forward": fsdp_config.get("reshard_after_forward"),
            }
        logger.info(
            "Training backend summary: requested_backend={}, resolved_mode={}, deepspeed={}, fsdp={}, fsdp_version={}, fsdp_config_summary={}, plugin_kwargs={}, bf16={}, transformers_version={}",
            trainer_config.train_backend,
            resolution.resolved_mode,
            linked_args.get("deepspeed"),
            linked_args.get("fsdp"),
            fsdp_config.get("fsdp_version", fsdp_config.get("version")),
            fsdp_config_summary,
            resolution.plugin_kwargs,
            linked_args.get("bf16"),
            resolution.package_versions.transformers
            if resolution.package_versions
            else transformers.__version__,
        )
        linked_args["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
        linked_args["ddp_find_unused_parameters"] = True
        linked_args["max_grad_norm"] = 1.0
        training_args = TrainingArguments(**linked_args)
        return training_args

    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        loss_keys = [_ for _ in outputs if _.endswith("_loss")]

        for loss_key in loss_keys:
            if outputs[loss_key] is None or torch.isclose(
                outputs[loss_key], torch.zeros_like(outputs[loss_key])
            ):
                if loss_key not in self.loss_cache:
                    self.loss_cache[loss_key] = 0.0
                continue
            self.loss_cache[loss_key] = outputs[loss_key].detach().item()
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        logs.update(self.loss_cache)
        super().log(logs, start_time)

    def training_step(self, model, inputs, num_items_in_batch=None):
        use_raw_backward = getattr(
            self.exp_config.trainer_config, "use_raw_backward", False
        )
        if use_raw_backward:
            return self._custom_training_step(
                model, inputs, num_items_in_batch, use_raw_backward
            )

        trainer_scales_loss = (
            not self.model_accepts_loss_kwargs or num_items_in_batch is None
        ) and self.compute_loss_func is None
        if trainer_scales_loss and not self.is_deepspeed_enabled:
            # Trainer divides batch-mean losses by the actual accumulation count.
            # Transformers 5.3 also configures an Accelerator divisor; suppress
            # that second division without changing the plugin outside this step.
            accelerator_steps = self.accelerator.gradient_accumulation_steps
            self.accelerator.gradient_accumulation_steps = 1
            try:
                return super().training_step(model, inputs, num_items_in_batch)
            finally:
                self.accelerator.gradient_accumulation_steps = accelerator_steps

        return super().training_step(model, inputs, num_items_in_batch)

    def _custom_training_step(
        self, model, inputs, num_items_in_batch, use_raw_backward
    ):
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(
                model, inputs, num_items_in_batch=num_items_in_batch
            )

        del inputs

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if (
            not getattr(self, "model_accepts_loss_kwargs", False)
            or num_items_in_batch is None
        ) and self.compute_loss_func is None:
            accumulation_steps = getattr(
                self,
                "current_gradient_accumulation_steps",
                self.args.gradient_accumulation_steps,
            )
            loss = loss / accumulation_steps

        loss.backward()
        return loss.detach()


def safe_save_model_for_hf_trainer(
    trainer: transformers.Trainer, output_dir: str
) -> None:
    """Collects the state dict and dump to disk."""

    if getattr(trainer.added_args, "tune_mm_mlp_adapter", False):
        keys_to_match = ["mm_projector"]
        weight_to_save_mm_projector = get_mm_adapter_state_maybe_zero_3(
            trainer.model.named_parameters(), keys_to_match
        )

        trainer.model.config.save_pretrained(output_dir)
        trainer.processing_class.save_pretrained(output_dir)

        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(
                    weight_to_save_mm_projector,
                    os.path.join(mm_projector_folder, f"{current_folder}.bin"),
                )

            else:
                torch.save(
                    weight_to_save_mm_projector,
                    os.path.join(output_dir, "mm_projector.bin"),
                )
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        if trainer.args.should_save:
            _save_lora_extra_trainable_state_for_trainer(trainer, output_dir)
        return

    if getattr(trainer, "is_fsdp_enabled", False):
        # Keep all ranks aligned around FSDP/FSDP2 state-dict collectives and rank0-only I/O.
        trainer.accelerator.wait_for_everyone()
        state_dict = trainer.accelerator.get_state_dict(trainer.model)
        if trainer.args.should_save:
            trainer._save(output_dir, state_dict=state_dict)  # noqa
            _save_lora_extra_trainable_state_for_trainer(trainer, output_dir)
        trainer.accelerator.wait_for_everyone()
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        _save_lora_extra_trainable_state_for_trainer(trainer, output_dir)
