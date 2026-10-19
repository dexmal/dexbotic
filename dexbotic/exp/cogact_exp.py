import argparse
import time
from dataclasses import dataclass, field
from typing import cast

import torch
from loguru import logger
from PIL import Image
from transformers import AutoTokenizer

from dexbotic.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from dexbotic.device import resolve_device
from dexbotic.exp.base_exp import (ActionConfig, BaseExp,
                                   ComputeNormActionConfig, DataConfig,
                                   FSDPProfile, ModelConfig,
                                   OptimizerConfig, TrainerConfig,
                                   InferenceConfig as BaseInferenceConfig)
from dexbotic.model.cogact.cogact_arch import (CogActConfig, CogACTForCausalLM,
                                               CogActModel)
from dexbotic.tokenization import conversation as conversation_lib
from dexbotic.tokenization.tokenization import tokenizer_image_token


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task',
        type=str,
        default='train',
        choices=[
            'train',
            'inference',
            'compute_norm_stats'])
    parser.add_argument(
        '--train-backend',
        type=str,
        default=None,
        choices=['deepspeed', 'fsdp', 'fsdp2', 'ddp'],
    )
    args, unknown = parser.parse_known_args()
    return args


@dataclass
class CogACTOptimizerConfig(OptimizerConfig):
    base_lr: float = field(default=2e-5)


@dataclass
class CogACTTrainerConfig(TrainerConfig):
    fsdp_profile: FSDPProfile = field(
        default_factory=lambda: FSDPProfile(
            enabled=True,
            transformer_layer_cls_to_wrap=("Qwen2DecoderLayer",),
            cast_model_to_bf16_backends=("fsdp", "fsdp2"),
        )
    )
    num_train_epochs: int = field(default=5)
    save_steps: int = field(default=20000)
    per_device_train_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=2)


@dataclass
class CogACTActionConfig(ActionConfig):
    pass


@dataclass
class CogACTDataConfig(DataConfig):
    action_config: ActionConfig = field(default_factory=CogACTActionConfig)


@dataclass
class CogACTModelConfig(ModelConfig):
    """
    CogACT model configuration class - inherits from base model configuration

    Configuration parameters:
    - action_model_type: Action model type, choices: 'DiT-B', 'DiT-L', 'DiT-S'
    - action_dim: Action dimension, typically 7 (position + rotation + gripper)
    - chunk_size: Action chunk size
    - freeze_action_head: Whether to freeze the action head module
    """
    action_model_type: str = field(default='DiT-B')
    action_dim: int = field(default=7)
    chunk_size: int = field(default=16)
    freeze_action_head: bool = field(default=False)

    def build_model(self) -> CogACTForCausalLM:

        if self.from_llm:
            model_config_args = {
                "llm_config": self.model_name_or_path,
                "chat_template": self.chat_template,
                "mm_projector_type": self.mm_projector_type,
                "mm_vision_tower": self.mm_vision_tower,
                "action_model_type": self.action_model_type,
                "action_dim": self.action_dim,
                "chunk_size": self.chunk_size,
                "init_llm_weights": True,
            }
            model_config = CogActConfig(**model_config_args)
            model = CogACTForCausalLM(model_config)
        else:
            model_config_args = {
                "model_name_or_path": self.model_name_or_path,
                "mm_projector_type": self.mm_projector_type,
                "mm_vision_tower": self.mm_vision_tower,
                "action_model_type": self.action_model_type,
                "action_dim": self.action_dim,
                "chunk_size": self.chunk_size,
            }
            model = CogACTForCausalLM.from_pretrained(self.model_name_or_path)
            model.model.initialize_model(model_config_args)

        # Under meta-device init (transformers>=5.0), CLIPVisionTower defers
        # load_model(). Finalize non-parameter state (image_processor) now that
        # weights have been loaded by from_pretrained.
        vision_tower = model.model.mm_vision_tower
        if getattr(vision_tower, "_meta_initialized", False) and not vision_tower.is_loaded:
            vision_tower.load_model()

        self._freeze_model(model)

        return model

    def _freeze_model(self, model: CogACTForCausalLM):
        model.model = cast(CogActModel, model.model)

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
        if self.freeze_action_head:
            for param in model.model.action_head_module.parameters():
                param.requires_grad = False


@dataclass
class InferenceConfig(BaseInferenceConfig):

    def _load_model(self) -> None:
        self.device = resolve_device(self.device)
        logger.info(f"Loading model from {self.model_name_or_path}")
        logger.info(f"Using device: {self.device}")
        model = CogACTForCausalLM.from_pretrained(self.model_name_or_path,
                                                  torch_dtype=torch.bfloat16,
                                                  low_cpu_mem_usage=True,
                                                  trust_remote_code=True,
                                                  device_map={"": str(self.device)}).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model.config
        logger.info(f"Model loaded successfully")

    def _build_policy(self):
        from dexbotic.policy.cogact_policy import CogACTPolicy
        return CogACTPolicy(
            model=self.model,
            tokenizer=self.tokenizer,
            norm_stats=self.norm_stats,
            camera_order=self.camera_order,
        )

    def _get_response(self, text: str, images: list[str]) -> str:
        t0 = time.monotonic()
        if len(images) == 1:
            images = [Image.open(images[0]).convert('RGB')]
            image_tensor = self.model.process_images(images).to(dtype=self.model.dtype)
        else:
            images = [Image.open(image).convert('RGB') for image in images]
            image_tensor = self.model.process_images(
                images).to(dtype=self.model.dtype).unsqueeze(0)

        self._save_image(images, text)

        conv = conversation_lib.conv_templates[self.model_config.chat_template].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + text)
        conv.append_message(conv.roles[1], ' ')
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt').unsqueeze(0).to(
            self.model.device)
        logger.debug(f'input_ids: {input_ids}')
        inference_args = {
            'cfg_scale': 1.5,
            'num_ddim_steps': 10,
            'action_norms': self.norm_stats}

        outputs = self.model.inference_action(input_ids, image_tensor, inference_args)
        logger.info(f'prompt: <start>{prompt}<end>\naction: {outputs}')
        logger.info(f"Processing time: {time.monotonic() - t0}")
        return outputs


@dataclass
class CogACTExp(BaseExp):
    model_config: CogACTModelConfig = field(default_factory=CogACTModelConfig)
    optimizer_config: CogACTOptimizerConfig = field(
        default_factory=CogACTOptimizerConfig)
    trainer_config: CogACTTrainerConfig = field(default_factory=CogACTTrainerConfig)
    data_config: CogACTDataConfig = field(default_factory=CogACTDataConfig)
    inference_config: InferenceConfig = field(default_factory=InferenceConfig)

    def inference(self) -> None:
        self.inference_config.run()

    def compute_norm_stats(self) -> None:
        self.data_config.action_config = ComputeNormActionConfig()
        self.data_config.action_config.compute_norm_stats(self.data_config.dataset_name)


if __name__ == "__main__":
    args = parse_args()
    exp = CogACTExp()
    if args.train_backend is not None:
        exp.trainer_config.train_backend = args.train_backend
    if args.task == 'train':
        exp.train()
    elif args.task == 'inference':
        exp.inference()
    elif args.task == 'compute_norm_stats':
        exp.compute_norm_stats()
