import os
from typing import TYPE_CHECKING, Optional
import shutil

import torch
import transformers
from loguru import logger
from easydict import EasyDict
from transformers import Trainer, TrainingArguments

from dexbotic.exp.utils import get_mm_adapter_state_maybe_zero_3
from dexbotic.model.dexbotic_arch import DexboticVLMModel

if TYPE_CHECKING:
    from dexbotic.exp.base_exp import BaseExp


class DexboticTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.exp_config: BaseExp = kwargs.pop("exp_config")
        training_args = self._link_exp_config()
        super().__init__(*args, args=training_args, **kwargs)
        self.loss_cache = {}

    def create_optimizer(self) -> torch.optim.Optimizer:
        opt_model: DexboticVLMModel = self.model

        if self.optimizer is None:
            optimizer_grouped_parameters = self.exp_config.optimizer_config._get_optimizer_grouped_parameters(
                opt_model)

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None) -> None:
        logger.info(f"Saving checkpoint at step {self.state.global_step}")
        if getattr(self.added_args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector']
            weight_to_save = get_mm_adapter_state_maybe_zero_3(
                self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(
                    weight_to_save, os.path.join(
                        output_dir, 'mm_projector.bin'))

        else:
            super(DexboticTrainer, self)._save_checkpoint(model, trial)
            # Copy norm_stats.json to checkpoint directory after saving
            if self.args.local_rank == 0 or self.args.local_rank == -1:
                from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                run_dir = self._get_output_dir(trial=trial)
                output_dir = os.path.join(run_dir, checkpoint_folder)
                self._copy_norm_stats_to_checkpoint(output_dir)

    def _copy_norm_stats_to_checkpoint(self, checkpoint_dir: str) -> None:
        """Copy norm_stats.json from main output directory to checkpoint directory"""
        
        main_output_dir = self.args.output_dir
        norm_stats_src = os.path.join(main_output_dir, "norm_stats.json")
        norm_stats_dst = os.path.join(checkpoint_dir, "norm_stats.json")
        
        if os.path.exists(norm_stats_src):
            try:
                shutil.copy2(norm_stats_src, norm_stats_dst)
                logger.info(f"Copied norm_stats.json to checkpoint directory: {checkpoint_dir}")
            except Exception as e:
                logger.warning(f"Failed to copy norm_stats.json to checkpoint: {e}")

    def _save(self, output_dir: Optional[str] = None, state_dict=None) -> None:
        if getattr(self.added_args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(DexboticTrainer, self)._save(output_dir, state_dict)

    def _link_exp_config(self) -> TrainingArguments:
        """Link the exp config to the trainer args"""
        linked_args = {
            "output_dir": self.exp_config.trainer_config.output_dir,
            "num_train_epochs": self.exp_config.trainer_config.num_train_epochs,
            "max_steps": self.exp_config.trainer_config.num_train_steps,
            "per_device_train_batch_size": self.exp_config.trainer_config.per_device_train_batch_size,
            "gradient_accumulation_steps": self.exp_config.trainer_config.gradient_accumulation_steps,
            "save_strategy": self.exp_config.trainer_config.save_strategy,
            "save_steps": self.exp_config.trainer_config.save_steps,
            "save_total_limit": self.exp_config.trainer_config.save_total_limit,
            "save_only_model": self.exp_config.trainer_config.save_only_model,
            "logging_steps": self.exp_config.trainer_config.logging_steps,
            "gradient_checkpointing": self.exp_config.trainer_config.gradient_checkpointing,
            "dataloader_num_workers": self.exp_config.trainer_config.dataloader_num_workers,
            # "model_max_length": self.exp_config.trainer_config.model_max_length,
            "bf16": self.exp_config.trainer_config.bf16,
            "tf32": self.exp_config.trainer_config.tf32,
            "lr_scheduler_type": self.exp_config.trainer_config.lr_scheduler_type,
            "lr_scheduler_kwargs": self.exp_config.trainer_config.lr_scheduler_kwargs,
            "run_name": self.exp_config.trainer_config.run_name,
            'remove_unused_columns': False,
            "deepspeed": self.exp_config.trainer_config.deepspeed,
            "learning_rate": self.exp_config.optimizer_config.base_lr,
            "adam_beta1": self.exp_config.optimizer_config.adam_beta1,
            "adam_beta2": self.exp_config.optimizer_config.adam_beta2,
            "warmup_steps": self.exp_config.optimizer_config.warmup_steps,
            "weight_decay": self.exp_config.optimizer_config.weight_decay,
        }
        self.added_args = EasyDict({
            "tune_mm_mlp_adapter": self.exp_config.trainer_config.tune_mm_mlp_adapter,
        })
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


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str) -> None:
    """Collects the state dict and dump to disk."""

    if getattr(trainer.added_args, "tune_mm_mlp_adapter", False):
        keys_to_match = ['mm_projector']
        weight_to_save_mm_projector = get_mm_adapter_state_maybe_zero_3(
            trainer.model.named_parameters(), keys_to_match)

        trainer.model.config.save_pretrained(output_dir)
        trainer.processing_class.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(
                    weight_to_save_mm_projector,
                    os.path.join(
                        mm_projector_folder,
                        f'{current_folder}.bin'))

            else:
                torch.save(
                    weight_to_save_mm_projector,
                    os.path.join(
                        output_dir,
                        'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
