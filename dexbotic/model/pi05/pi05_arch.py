from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CONFIG_MAPPING, AutoConfig, AutoModel, DynamicCache
from transformers.models.gemma.modeling_gemma import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from dexbotic.model.dexbotic_arch import (
    ActionOutputForCausalLM,
    CausalLMOutputDexbotic,
    DexboticConfig,
    DexboticForCausalLM,
    DexboticVLMModel,
)
from dexbotic.model.pi05.transformers_pi05.gemma.modeling_gemma import _gated_residual


def make_attn_mask(input_mask: torch.BoolTensor, ar_mask: torch.BoolTensor):
    ar_mask = ar_mask.broadcast_to(input_mask.shape)
    cumsum = torch.cumsum(ar_mask, dim=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    attn_mask = torch.logical_and(attn_mask, valid_mask)
    return attn_mask


def make_attn_mask_4d(attn_mask: torch.BoolTensor):
    attn_mask = torch.where(attn_mask, 0.0, -2.3819763e38)[:, None]
    return attn_mask


def posemb_sincos(
    position_ids: torch.LongTensor,
    dim: int,
    min_period: int,
    max_period: int,
):
    if dim % 2 != 0:
        raise ValueError("dim must be even for sincos position embeddings")

    fraction = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float64).to(
        position_ids.device
    )
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = position_ids[:, None].float() / period[None, :] * 2 * np.pi
    return torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1)


class Pi05Config(DexboticConfig):
    model_type = "dexbotic_pi05"
    vision_config: dict | str
    processor_config: str
    action_config: dict | str
    action_dim: Optional[int] = 32
    chunk_size: Optional[int] = 50

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        vision_config = kwargs.pop("vision_config", None)
        if isinstance(vision_config, dict):
            self.vision_config = CONFIG_MAPPING[vision_config["model_type"]](
                **vision_config
            )
        elif isinstance(vision_config, str):
            self.vision_config = AutoConfig.from_pretrained(vision_config)

        action_config = kwargs.get("action_config", None)
        if isinstance(action_config, dict):
            self.action_config = CONFIG_MAPPING[action_config["model_type"]](
                **action_config
            )
        elif isinstance(action_config, str):
            self.action_config = AutoConfig.from_pretrained(action_config)

        llm_config = kwargs.get("llm_config", None)
        if isinstance(llm_config, dict):
            self.llm_config = CONFIG_MAPPING[llm_config["model_type"]](**llm_config)
        elif isinstance(llm_config, str):
            self.llm_config = AutoConfig.from_pretrained(llm_config)


class Pi05Model(DexboticVLMModel):
    def __init__(self, config: Pi05Config):
        super().__init__(config)

        action_model_config = config.action_config
        self.action_expert = AutoModel.from_config(action_model_config)

        self.time_mlp_in = nn.Linear(
            action_model_config.width, action_model_config.width
        )
        self.time_mlp_out = nn.Linear(
            action_model_config.width, action_model_config.width
        )

        self.action_in_proj = nn.Linear(
            config.action_dim, action_model_config.hidden_size
        )
        self.action_time_activation = nn.SiLU()
        self.action_out_proj = nn.Linear(
            action_model_config.hidden_size, config.action_dim
        )
        torch.set_float32_matmul_precision("high")


class Pi05ForCausalLM(DexboticForCausalLM, ActionOutputForCausalLM):
    config_class = Pi05Config

    def _real_init(self, config: Pi05Config):
        self.model = Pi05Model(config)
        self.post_init()

    def _inner_forward_mot(
        self,
        module_list: List[nn.Module],
        input_embeds_list: List[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        update_cache: bool = True,
        adarms_cond: torch.Tensor = None,
    ):
        all_hidden_states = (input_embeds_list,) if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for layer_idx, layers in enumerate(
            zip(*[module.layers for module in module_list])
        ):
            query_list, key_list, value_list = [], [], []
            res_gate_list = []
            seq_len_list = []
            for module_idx, (layer, input_embeds) in enumerate(
                zip(layers, input_embeds_list)
            ):
                if input_embeds is None:
                    seq_len_list.append(0)
                    res_gate_list.append(None)
                else:
                    prenorm_embeds, gate = layer.input_layernorm(
                        input_embeds,
                        None
                        if module_idx == 0
                        else adarms_cond,  # LLM layers do not use adarms_cond
                    )
                    batch_size, seq_len, _ = prenorm_embeds.shape
                    seq_len_list.append(seq_len)

                    query = (
                        layer.self_attn.q_proj(prenorm_embeds)
                        .view(batch_size, seq_len, -1, layer.self_attn.head_dim)
                        .transpose(1, 2)
                    )
                    key = (
                        layer.self_attn.k_proj(prenorm_embeds)
                        .view(batch_size, seq_len, -1, layer.self_attn.head_dim)
                        .transpose(1, 2)
                    )
                    value = (
                        layer.self_attn.v_proj(prenorm_embeds)
                        .view(batch_size, seq_len, -1, layer.self_attn.head_dim)
                        .transpose(1, 2)
                    )
                    query_list.append(query)
                    key_list.append(key)
                    value_list.append(value)
                    res_gate_list.append(gate)

            assert (
                sum(seq_len_list) > 0
            ), "At least one module should have non-zero sequence length."
            query_states = torch.cat(query_list, dim=2)
            key_states = torch.cat(key_list, dim=2)
            value_states = torch.cat(value_list, dim=2)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, *position_embeddings
            )

            if past_key_values is not None:
                if update_cache:
                    key_states, value_states = past_key_values.update(
                        key_states, value_states, layer_idx
                    )
                else:
                    key_states = torch.cat(
                        [past_key_values.key_cache[layer_idx], key_states], dim=-2
                    )
                    value_states = torch.cat(
                        [past_key_values.value_cache[layer_idx], value_states], dim=-2
                    )

            attn_output, attn_weights = eager_attention_forward(
                layers[0].self_attn,
                query_states,
                key_states,
                value_states,
                mask,
                layers[0].self_attn.scaling,
            )

            if output_attentions:
                all_self_attns += (attn_weights,)

            attn_output = attn_output.view(batch_size, sum(seq_len_list), -1)
            layer_embeds_list = []
            start_idx = 0
            for module_idx, (layer, input_embeds, gate) in enumerate(
                zip(layers, input_embeds_list, res_gate_list)
            ):
                seq_len = seq_len_list[module_idx]
                if seq_len == 0:
                    layer_embeds_list.append(None)
                    continue
                attn_embeds = attn_output[:, start_idx : start_idx + seq_len, :]
                start_idx += seq_len

                attn_embeds = layer.self_attn.o_proj(attn_embeds)
                residual_attn_embeds = _gated_residual(input_embeds, attn_embeds, gate)
                postnorm_embeds, gate = layer.post_attention_layernorm(
                    residual_attn_embeds, adarms_cond
                )
                mlp_embeds = layer.mlp(postnorm_embeds)
                residual_mlp_embeds = _gated_residual(
                    residual_attn_embeds, mlp_embeds, gate
                )
                layer_embeds_list.append(residual_mlp_embeds)

            input_embeds_list = layer_embeds_list
            assert (
                len(input_embeds_list) == 2
            ), "Length of input_embeds_list should match module_list."

        decoder_embeds_list = []
        for module_idx, (module, input_embeds) in enumerate(
            zip(module_list, input_embeds_list)
        ):
            if input_embeds is not None:
                input_embeds, _ = module.norm(input_embeds, adarms_cond)
            decoder_embeds_list.append(input_embeds)

        if output_hidden_states:
            all_hidden_states += (decoder_embeds_list,)
        return decoder_embeds_list, past_key_values, all_hidden_states, all_self_attns

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        image_features = self.model.mm_vision_module(images)
        image_features = self.model.mm_projector_module(image_features)
        return image_features

    def embed_prefix(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.FloatTensor] = None,
        image_masks: Optional[torch.BoolTensor] = None,
    ):
        input_mask = []
        ar_mask = []
        tokens = []

        images = images.transpose(0, 1)
        image_masks = image_masks.transpose(0, 1)
        for image, image_mask in zip(images, image_masks):
            image_tokens = self.encode_images(image)
            tokens.append(image_tokens)
            image_mask = image_mask.unsqueeze(1).expand(
                image.shape[0], image_tokens.shape[1]
            )
            input_mask.append(image_mask)
            ar_mask += [False] * image_tokens.shape[1]

        if input_ids is not None:
            input_tokens = (
                self.model.llm.embed_tokens(input_ids)
                * self.model.config.llm_config.hidden_size**0.5
            )
            input_mask.append(attention_mask)
            ar_mask += [False] * input_tokens.shape[1]
            tokens.append(input_tokens)

        tokens = torch.cat(tokens, dim=1)
        input_mask = torch.cat(input_mask, dim=1)
        ar_mask = torch.tensor(ar_mask, device=tokens.device)
        return tokens, input_mask, ar_mask

    def embed_suffix(
        self,
        noisy_actions: Optional[torch.FloatTensor] = None,
        time: Optional[torch.FloatTensor] = None,
    ):
        input_mask = []
        ar_mask = []
        tokens = []

        time_emb = posemb_sincos(
            time,
            self.model.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
        )
        time_emb = time_emb.to(noisy_actions.dtype)

        def time_mlp_func(time_emb):
            time_emb = self.model.time_mlp_in(time_emb)
            time_emb = F.silu(time_emb)  # swish == silu
            time_emb = self.model.time_mlp_out(time_emb)
            adarms_cond = F.silu(time_emb)
            return adarms_cond

        adarms_cond = time_mlp_func(time_emb)

        action_tokens = self.model.action_in_proj(noisy_actions)
        tokens.append(action_tokens)
        input_mask.append(
            torch.ones(
                action_tokens.shape[:2],
                device=action_tokens.device,
                dtype=torch.bool,
            )
        )
        ar_mask += [True] + ([False] * (self.model.config.chunk_size - 1))
        tokens = torch.cat(tokens, dim=1)
        input_mask = torch.cat(input_mask, dim=1)
        ar_mask = torch.tensor(ar_mask, device=tokens.device)
        return tokens, input_mask, ar_mask, adarms_cond

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
        return_dict: Optional[bool] = None,
        actions: Optional[torch.FloatTensor] = None,
        states: Optional[torch.FloatTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        repeated_diffusion_steps: int = 4,
        image_masks: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> CausalLMOutputDexbotic:
        batch_shape = actions.shape[:1]
        noise = torch.normal(
            mean=torch.zeros_like(actions),
            std=torch.ones_like(actions),
        ).to(
            device=actions.device,
            dtype=actions.dtype,
        )
        time = (
            torch.distributions.Beta(1.5, 1)
            .sample(batch_shape)
            .to(device=actions.device, dtype=actions.dtype)
            * 0.999
            + 0.001
        )

        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(
            input_ids,
            attention_mask,
            images,
            image_masks,
        )
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            x_t, time
        )
        input_mask = torch.cat([prefix_mask, suffix_mask], dim=1)
        ar_mask = torch.cat([prefix_ar_mask, suffix_ar_mask], dim=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        attn_mask = make_attn_mask_4d(attn_mask)
        positions = torch.cumsum(input_mask, dim=1) - 1
        position_embeddings = self.model.llm.rotary_emb(prefix_tokens, positions)

        (
            (_, suffix_out),
            past_key_values,
            hidden_states,
            attentions,
        ) = self._inner_forward_mot(
            [self.model.llm, self.model.action_expert],
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            cache_position=positions,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            adarms_cond=adarms_cond,
        )
        v_t = self.model.action_out_proj(suffix_out[:, -self.model.config.chunk_size :])

        loss = F.mse_loss(v_t, u_t, reduction="none")
        loss = loss.mean()

        if output_hidden_states:
            hidden_states += (v_t,)

        outputs = CausalLMOutputDexbotic(
            loss=loss,
            logits=v_t,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=attentions,
        )
        return outputs

    @torch.no_grad()
    def inference_action(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        states: Optional[torch.FloatTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        image_masks: Optional[torch.BoolTensor] = None,
        diffusion_steps: int = 10,
        **kwargs,
    ):
        batch_size = states.shape[0]

        dt = -1.0 / diffusion_steps
        noise = torch.normal(
            0,
            1,
            size=(batch_size, self.model.config.chunk_size, self.config.action_dim),
            device=states.device,
            dtype=states.dtype,
        )

        time = torch.tensor(1.0, device=states.device, dtype=states.dtype)

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(
            input_ids,
            attention_mask,
            images,
            image_masks,
        )
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_attn_mask = make_attn_mask_4d(prefix_attn_mask)
        positions = torch.cumsum(prefix_mask, dim=1) - 1
        position_embeddings = self.model.llm.rotary_emb(prefix_tokens, positions)
        _, kv_cache, _, _ = self._inner_forward_mot(
            [self.model.llm, self.model.action_expert],
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            position_embeddings=position_embeddings,
            past_key_values=DynamicCache(),
            cache_position=positions,
            output_hidden_states=False,
            output_attentions=False,
        )

        def step(x_t, time):
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                x_t, time.broadcast_to(batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = prefix_mask.unsqueeze(1).repeat(
                1, suffix_tokens.shape[1], 1
            )
            full_attn_mask = torch.cat([prefix_attn_mask, suffix_attn_mask], dim=-1)
            full_attn_mask = make_attn_mask_4d(full_attn_mask)
            assert full_attn_mask.shape == (
                batch_size,
                1,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            full_positions = (
                prefix_mask.sum(axis=-1).unsqueeze(-1)
                + torch.cumsum(suffix_mask, dim=-1)
                - 1
            )
            full_position_embeddings = self.model.llm.rotary_emb(
                suffix_tokens, full_positions
            )
            (prefix_out, suffix_out), _, _, _ = self._inner_forward_mot(
                [self.model.llm, self.model.action_expert],
                [None, suffix_tokens],
                mask=full_attn_mask,
                position_embeddings=full_position_embeddings,
                past_key_values=kv_cache,
                cache_position=torch.cat(
                    [positions, torch.cumsum(suffix_mask, dim=1) - 1], dim=1
                ),
                output_hidden_states=False,
                output_attentions=False,
                update_cache=False,
                adarms_cond=adarms_cond,
            )
            assert prefix_out is None
            v_t = self.model.action_out_proj(
                suffix_out[:, -self.model.config.chunk_size :]
            )
            return x_t + v_t * dt, time + dt

        while time > -dt / 2:
            noise, time = step(noise, time)

        return noise

    def process_images(self, images):
        vision_tower = self.model.mm_vision_module
        image_processor = vision_tower.image_processor
        image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "pad")
        new_images = []
        if image_aspect_ratio == "pad":
            for image in images:
                image = self.expand2square(
                    image, tuple(int(x * 255) for x in [0, 0, 0])
                )
                image = image_processor.preprocess(image, return_tensors="pt")[
                    "pixel_values"
                ][0]
                new_images.append(image)
        else:
            return image_processor(images, return_tensors="pt")["pixel_values"]
        if all(x.shape == new_images[0].shape for x in new_images):
            new_images = torch.stack(new_images, dim=0)
        return new_images
