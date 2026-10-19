from typing import List, Optional

import torch
import torch.nn as nn

from dexbotic.model.oft.action_model.builder import build_action_model
from dexbotic.model.dexbotic_arch import (ActionOutputForCausalLM,
                                          CausalLMOutputDexbotic,
                                          DexboticConfig, DexboticForCausalLM,
                                          DexboticVLMModel)


class OFTConfig(DexboticConfig):
    model_type = "dexbotic_oft"
    action_model_type: Optional[str] = None
    action_dim: Optional[int] = None
    chunk_size: Optional[int] = None
    use_proprio: Optional[bool] = False
    proprio_dim: Optional[int] = None


class OFTModel(DexboticVLMModel):
    def __init__(self, config: OFTConfig):
        super().__init__(config)
        if config.action_model_type is not None:
            self.action_head = self._build_action_head_module(config)

    def _build_action_head_module(self, config: OFTConfig):
        if getattr(self, 'action_head', None) is not None:
            return self.action_head
        self.action_head = build_action_model(config)
        return self.action_head

    @property
    def action_head_module(self) -> nn.Module:
        return self.action_head

    @property
    def action_head_prefix(self) -> str:
        return 'action_head'

    def initialize_model(self, extra_config: dict):
        for key, value in extra_config.items():
            setattr(self.config, key, value)
        self.mm_vision_tower = self._build_mm_vision_module(self.config.mm_vision_tower)
        self.mm_projector = self._build_mm_projector_module(self.config)
        self.action_head = self._build_action_head_module(self.config)


class OFTForCausalLM(DexboticForCausalLM, ActionOutputForCausalLM):
    config_class = OFTConfig

    def _real_init(self, config: OFTConfig):
        self.model = OFTModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self,
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
                noisy_dict: Optional[dict[str, torch.FloatTensor]] = None,
                **kwargs,
                ) -> CausalLMOutputDexbotic:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if actions is not None:
            actions = actions.reshape(actions.size(0), -1, self.config.action_dim)
            actions = actions[:, :self.config.chunk_size, :]

        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
            cache_position
        ) = self.model._prepare_inputs_labels_for_multimodal(
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            labels,
            cache_position,
            images
        )
        
        if 'Linear' in self.config.action_model_type:
            action_embeds = self.model.action_head.action_query.expand(inputs_embeds.shape[0], -1, -1)
        else:
            if noisy_dict is None:
                noisy_dict = self.model.action_head.sample_noisy_actions(actions)
            noise, noisy_actions, diffusion_timestep_embeddings = (
                noisy_dict["noise"],
                noisy_dict["noisy_actions"],
                noisy_dict["diffusion_timestep_embeddings"],
            )
            noisy_actions = noisy_actions.reshape(noisy_actions.size(0), -1).unsqueeze(-1)
            action_embeds = self.model.action_head.noisy_action_projector(noisy_actions)
            action_embeds = torch.cat([diffusion_timestep_embeddings, action_embeds], dim=1)
        
        if self.config.use_proprio:
            assert states is not None, "states is required when use_proprio is True"
            state_embeds = self.model.action_head.proprio_projector(states).reshape(states.size(0), -1, self.config.hidden_size)
            action_embeds = torch.cat([state_embeds, action_embeds], dim=1)
            
        inputs_embeds, attention_mask, non_padding_lengths = self.insert_action_embedding(inputs_embeds, attention_mask, action_embeds)
            
        outputs = self.model.llm(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_hidden_states=True,
        )

        last_hidden_state = outputs.hidden_states[-1]
        action_hidden_states = self.extract_action_hidden_states(last_hidden_state, action_embeds.size(1), non_padding_lengths)
        
        if self.config.use_proprio:
            action_hidden_states = action_hidden_states[:, 1:, :]

        loss = None

        if 'Linear' in self.config.action_model_type:
            predicted_actions = self.model.action_head.predict_action(action_hidden_states)
        else:
            predicted_noise = self.model.action_head.predict_noise(action_hidden_states[:,1:,:])
            predicted_actions = predicted_noise
        with torch.amp.autocast('cuda', dtype=torch.float32):
            if actions is not None:
                if 'Linear' in self.config.action_model_type:
                    loss = torch.nn.L1Loss()(actions, predicted_actions)
                else:
                    loss = nn.functional.mse_loss(predicted_noise, noise, reduction="mean")

        if not return_dict:
            return (loss,) + last_hidden_state if loss is not None else last_hidden_state

        return CausalLMOutputDexbotic(
            loss=loss,
            logits=predicted_actions,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,

        )
        
    @staticmethod
    def insert_action_embedding(inputs_embeds, attention_mask, action_embeds):
        if attention_mask is None:
            non_padding_lengths = torch.full((inputs_embeds.size(0),), inputs_embeds.size(1), device=inputs_embeds.device)
            inputs_embeds = torch.cat([inputs_embeds, action_embeds], dim=1)
            return inputs_embeds, attention_mask, non_padding_lengths
        
        non_padding_lengths = attention_mask.sum(dim=1)
        max_length = inputs_embeds.size(1)

        updated_inputs_embeds = torch.zeros(
            inputs_embeds.size(0),
            max_length + action_embeds.size(1),
            inputs_embeds.size(2),
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype
        )

        updated_attention_mask = torch.zeros(
            attention_mask.size(0),
            max_length + action_embeds.size(1),
            device=attention_mask.device,
            dtype=attention_mask.dtype
        )

        for i, length in enumerate(non_padding_lengths):
            length = length.item()  # Convert to integer
            updated_inputs_embeds[i, :length] = inputs_embeds[i, :length]
            updated_inputs_embeds[i, length:length + action_embeds.size(1)] = action_embeds[i]
            updated_inputs_embeds[i, length + action_embeds.size(1):] = inputs_embeds[i, length:]

            updated_attention_mask[i, :length + action_embeds.size(1)] = 1

        return updated_inputs_embeds, updated_attention_mask, non_padding_lengths

    @staticmethod
    def extract_action_hidden_states(last_hidden, action_length, non_padding_lengths):
        action_hidden_states = []
        for i, length in enumerate(non_padding_lengths):
            length = length.item()  # Convert to integer
            action_hidden_states.append(last_hidden[i, length:length + action_length, :])
        action_hidden_states = torch.stack(action_hidden_states, dim=0)
        return action_hidden_states
    
    @torch.no_grad()
    def inference_action(self, input_ids, image_tensor, inference_args={}, **kwargs):
        num_ddim_steps = inference_args.get('num_ddim_steps', 10)
        action_norms = inference_args.get('action_norms')
        states = inference_args.get('states', None)
        
        if 'Linear' in self.config.action_model_type:
            out_features = self.__call__(input_ids,
                                         images=image_tensor,
                                         use_cache=True,
                                         states=states)
            predicted_actions = out_features.logits
        else:
            self.model.action_head.noise_scheduler.set_timesteps(num_ddim_steps)
            noise = torch.randn(
                input_ids.size(0), self.config.chunk_size, self.config.action_dim,
                device=input_ids.device, dtype=image_tensor.dtype
            )
            curr_noisy_actions = noise

            for t in self.model.action_head.noise_scheduler.timesteps:
                timesteps = torch.Tensor([t]).to(input_ids.device)
                diffusion_timestep_embeddings = (
                    self.model.action_head.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device)
                )
                diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)
                noisy_dict = {
                    'noise': noise,
                    'noisy_actions': curr_noisy_actions,
                    'diffusion_timestep_embeddings': diffusion_timestep_embeddings,
                }
                out_features = self.__call__(input_ids,
                                             images=image_tensor,
                                             use_cache=True,
                                             noisy_dict=noisy_dict,
                                             states=states)
                predicted_noise = out_features.logits
                curr_noisy_actions = self.model.action_head.noise_scheduler.step(predicted_noise, t, curr_noisy_actions).prev_sample
            predicted_actions = curr_noisy_actions
        actions = predicted_actions[0]

        actions = self._denorm(actions.float().cpu().numpy(), action_norms).tolist()
        return actions
