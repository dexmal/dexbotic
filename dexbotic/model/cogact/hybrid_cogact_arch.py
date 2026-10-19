from typing import List, Optional

import torch
import torch.nn as nn

from dexbotic.constants import IGNORE_INDEX
from dexbotic.model.cogact.action_model.builder import build_action_model
from dexbotic.model.dexbotic_arch import (
    ActionOutputForCausalLM,
    CausalLMOutputDexbotic,
    DexboticConfig,
    DexboticForCausalLM,
    DexboticVLMModel,
)


class CogActConfig(DexboticConfig):
    model_type = "dexbotic_cogact"
    action_model_type: Optional[str] = None
    action_dim: Optional[int] = None
    chunk_size: Optional[int] = None


class CogActModel(DexboticVLMModel):
    def __init__(self, config: CogActConfig):
        super().__init__(config)
        if config.action_model_type is not None:
            self.action_head = self._build_action_head_module(config)

    def _build_action_head_module(self, config: CogActConfig):
        if getattr(self, "action_head", None) is not None:
            return self.action_head
        self.action_head = build_action_model(config)
        return self.action_head

    @property
    def action_head_module(self) -> nn.Module:
        return self.action_head

    @property
    def action_head_prefix(self) -> str:
        return "action_head"

    def initialize_model(self, extra_config: dict):
        for key, value in extra_config.items():
            setattr(self.config, key, value)
        self.mm_vision_tower = self._build_mm_vision_module(self.config.mm_vision_tower)
        self.mm_projector = self._build_mm_projector_module(self.config)
        self.action_head = self._build_action_head_module(self.config)


class HybridCogACTForCausalLM(DexboticForCausalLM, ActionOutputForCausalLM):
    config_class = CogActConfig

    def _real_init(self, config: CogActConfig):
        self.model = CogActModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        actions: Optional[torch.LongTensor] = None,
        states: Optional[torch.LongTensor] = None,
        repeated_diffusion_steps: int = 4,
        has_action: Optional[torch.Tensor] = None,
        has_text: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputDexbotic:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
            cache_position,
        ) = self.model._prepare_inputs_labels_for_multimodal(
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            labels,
            cache_position,
            images,
        )
        outputs = self.model.llm(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_hidden_states=True,
        )

        last_hidden_state = outputs.hidden_states[-1]
        logits = self.lm_head(last_hidden_state)

        text_loss = None
        action_loss = None

        if labels is not None:
            assert has_action is not None, "has_action must be provided"
            text_labels = labels.clone()
            has_text = has_text.bool().view(-1)
            if ~has_text.any():
                # if there is some text in the batch, use the partial labels with one weight
                # else, use the full labels with zero weight
                text_labels[~has_text] = IGNORE_INDEX

            text_loss = (
                self.loss_function(logits, text_labels, self.model.backbone.vocab_size)
                * has_text.any().float()
            )
            # text_loss = self.loss_function(logits, text_labels, self.model.backbone.vocab_size) * (has_text.sum() / has_text.shape[0])

        if attention_mask is not None and actions is not None:
            # extract the cognition feature
            cumulative_sum = attention_mask.cumsum(dim=1)
            last_unmask_indices = (
                (cumulative_sum == cumulative_sum.max(dim=1, keepdim=True)[0])
                .float()
                .argmax(dim=1)
            )
            expanded_indices = last_unmask_indices.unsqueeze(-1).expand(
                -1, last_hidden_state.size(-1)
            )
            cognition_features = last_hidden_state.gather(
                1, expanded_indices.unsqueeze(1)
            )  # [B, 1, D]

        if actions is not None:
            assert has_text is not None, "has_text must be provided"
            actions = actions.reshape(actions.size(0), -1, self.config.action_dim).to(
                cognition_features.dtype
            )
            actions_future = actions[:, : self.config.chunk_size, :]

            actions_repeated = actions_future.repeat(repeated_diffusion_steps, 1, 1)
            has_action_repeated = has_action.squeeze().repeat(
                repeated_diffusion_steps
            )  # [B*repeat]
            cognition_features_repeated = cognition_features.repeat(
                repeated_diffusion_steps, 1, 1
            )

            with torch.amp.autocast("cuda", dtype=torch.float32):
                action_loss = self.model.action_head_module.loss(
                    actions_repeated, cognition_features_repeated, reduction="none"
                )
            action_loss = (action_loss.mean(dim=[1, 2]) * has_action_repeated).sum() / (
                has_action_repeated.sum() + 1e-6
            )

        loss = None
        if text_loss is not None and action_loss is not None:
            loss = text_loss + action_loss
        elif text_loss is not None:
            loss = text_loss
        elif action_loss is not None:
            loss = action_loss

        if not return_dict:
            return (
                (loss,) + last_hidden_state if loss is not None else last_hidden_state
            )

        return CausalLMOutputDexbotic(
            loss=loss,
            text_loss=text_loss,
            action_loss=action_loss,
            logits=last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def inference_action(self, input_ids, image_tensor, inference_args={}, **kwargs):
        cfg_scale = inference_args.get("cfg_scale", 1.5)
        num_ddim_steps = inference_args.get("num_ddim_steps", 10)
        action_norms = inference_args.get("action_norms")

        out_features = self.__call__(
            input_ids=input_ids, images=image_tensor, use_cache=True
        )

        cognition_features = out_features.logits[:, -1, :].unsqueeze(1)  # [B, 1, D]
        B = cognition_features.size(0)

        noise = torch.randn(
            B,
            self.config.chunk_size,
            self.config.action_dim,
            device=cognition_features.device,
            dtype=cognition_features.dtype,
        )  # [B T D]

        if cfg_scale > 1.0:
            noise = torch.cat([noise, noise], 0)

            uncondition = self.model.action_head.net.z_embedder.uncondition  # [1, D]
            uncondition = uncondition.unsqueeze(0).expand(B, 1, -1)  # [B, 1, D]
            z = torch.cat([cognition_features, uncondition], 0)
            model_kwargs = dict(z=z, cfg_scale=cfg_scale)
            sample_fn = self.model.action_head.net.forward_with_cfg
        else:
            model_kwargs = dict(z=cognition_features)
            sample_fn = self.model.action_head.net.forward

        if self.model.action_head.ddim_diffusion is None:
            self.model.action_head.create_ddim(ddim_step=num_ddim_steps)

        samples = self.model.action_head.ddim_diffusion.ddim_sample_loop(
            sample_fn,
            noise.shape,
            noise,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=cognition_features.device,
            eta=0.0,
        )
        if cfg_scale > 1.0:
            samples, _ = samples.chunk(2, dim=0)

        actions = self._denorm(samples[0].cpu().numpy(), action_norms).tolist()
        return actions
