"""DM0 Prog Model Architecture for Dexbotic.

This module implements the DM0 Prog model based on dm0 architecture,
with progress prediction support for inference only.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    CONFIG_MAPPING,
    DynamicCache,
    Qwen3ForCausalLM,
)
from transformers.models.qwen3 import modeling_qwen3

from dexbotic.model.dexbotic_arch import (
    ActionOutputForCausalLM,
    CausalLMOutputDexbotic,
    DexboticConfig,
    DexboticForCausalLM,
    DexboticVLMModel,
)
from dexbotic.model.dm0.dm0_utils import (
    make_attn_mask_2d,
    make_attn_mask_4d,
    make_suffix_attn_mask_2d,
    posemb_sincos,
)


class DM0ProgConfig(DexboticConfig):
    """Configuration for DM0 Prog model."""

    model_type = "dexbotic_dm0_prog"
    action_config: dict | str = None
    processor_config: str = None
    action_dim: int = 32
    chunk_size: int = 50
    bf16: bool = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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


class DM0ProgModel(DexboticVLMModel):
    """DM0 Prog VLM model with action expert and progress prediction.

    This class follows pattern:
    - Inherits llm, mm_vision_tower, mm_projector from DexboticVLMModel
    - Adds action_expert as a direct submodule
    - Adds action projection and time MLP layers for flow matching
    - Adds progress projection layers for progress prediction
    """

    def __init__(self, config: DM0ProgConfig):
        # Use standard DexboticVLMModel initialization
        # This builds llm, mm_vision_tower, mm_projector as direct submodules
        super().__init__(config)

        # Build action expert (Qwen3ForCausalLM without embed_tokens)
        action_model_config = config.action_config
        self.action_expert = Qwen3ForCausalLM(action_model_config)
        self.action_expert.model.embed_tokens = None

        action_hidden = action_model_config.hidden_size

        # Action projection layers
        self.action_in_proj = nn.Linear(config.action_dim, action_hidden)
        self.action_out_proj = nn.Linear(action_hidden, config.action_dim)

        # Time MLP layers
        self.action_time_mlp_in = nn.Linear(2 * action_hidden, action_hidden)
        self.action_time_mlp_out = nn.Linear(action_hidden, action_hidden)

        # Progress projection layers
        self.progress_in_proj = nn.Linear(1, action_hidden)
        self.progress_out_proj = nn.Linear(action_hidden, 1)

        torch.set_float32_matmul_precision("high")

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image using vision tower and projector."""
        image_features = self.mm_vision_tower(
            image.to(
                device=self.mm_vision_tower.device, dtype=self.mm_vision_tower.dtype
            )
        )
        image_features = self.mm_projector(image_features)
        return image_features

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed language tokens using LLM embeddings."""
        return self.llm.embed_tokens(tokens)

    def to_bfloat16_for_selected_params(self):
        self.action_expert = self.action_expert.to(dtype=torch.bfloat16)
        self.llm = self.llm.to(dtype=torch.bfloat16)
        self.mm_vision_tower = self.mm_vision_tower.to(dtype=torch.bfloat16)
        self.mm_projector = self.mm_projector.to(dtype=torch.bfloat16)

        params_to_keep_float32 = [
            "mm_vision_tower.vision_model.conv1.weight",
            "mm_vision_tower.vision_model.conv1.bias",
            "mm_vision_tower.vision_model.positional_embedding",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                print(f"Keeping {name} in float32")
                param.data = param.data.to(dtype=torch.float32)


class DM0ProgForCausalLM(DexboticForCausalLM, ActionOutputForCausalLM):
    """DM0 Prog model for causal language modeling with action and progress prediction."""

    config_class = DM0ProgConfig

    def _real_init(self, config: DM0ProgConfig):
        self.model = DM0ProgModel(config)
        if config.bf16:
            self.model.to_bfloat16_for_selected_params()
        else:
            self.model = self.model.to(torch.float32)
        # Add lm_head for compatibility with parent class tie_weights
        self.lm_head = nn.Linear(
            config.llm_config.hidden_size, config.llm_config.vocab_size, bias=False
        )
        self.post_init()

    def _compute_merged_layer(
        self,
        layer_idx: int,
        module_list: List[nn.Module],
        input_embeds_list: List[torch.FloatTensor],
        position_ids: torch.LongTensor,
        past_key_values: DynamicCache | None,
        attention_mask: torch.Tensor,
        use_cache: bool,
    ) -> List[torch.FloatTensor]:
        """Compute a single merged attention layer across multiple modules."""
        query_list, key_list, value_list = [], [], []
        seq_len_list = []
        layers = [module.layers[layer_idx] for module in module_list]

        for module_idx, (layer, input_embeds) in enumerate(
            zip(layers, input_embeds_list)
        ):
            if input_embeds is None:
                seq_len_list.append(0)
            else:
                prenorm_embeds = layer.input_layernorm(input_embeds)
                batch_size, seq_len, _ = prenorm_embeds.shape
                seq_len_list.append(seq_len)

                if layer.self_attn.q_proj.weight.dtype == torch.bfloat16:
                    prenorm_embeds = prenorm_embeds.to(torch.bfloat16)

                query = layer.self_attn.q_norm(
                    layer.self_attn.q_proj(prenorm_embeds).view(
                        batch_size, seq_len, -1, layer.self_attn.head_dim
                    )
                ).transpose(1, 2)
                key = layer.self_attn.k_norm(
                    layer.self_attn.k_proj(prenorm_embeds).view(
                        batch_size, seq_len, -1, layer.self_attn.head_dim
                    )
                ).transpose(1, 2)
                value = (
                    layer.self_attn.v_proj(prenorm_embeds)
                    .view(batch_size, seq_len, -1, layer.self_attn.head_dim)
                    .transpose(1, 2)
                )

                if layer.self_attn.q_proj.weight.dtype == torch.bfloat16:
                    query = query.to(torch.bfloat16)
                    key = key.to(torch.bfloat16)

                query_list.append(query)
                key_list.append(key)
                value_list.append(value)

        query_states = torch.cat(query_list, dim=2)
        key_states = torch.cat(key_list, dim=2)
        value_states = torch.cat(value_list, dim=2)

        # Get rotary embeddings
        rotary_emb = self.model.llm.rotary_emb
        dummy_tensor = torch.zeros(
            query_states.shape[0],
            query_states.shape[2],
            query_states.shape[-1],
            device=query_states.device,
            dtype=query_states.dtype,
        )
        cos, sin = rotary_emb(dummy_tensor, position_ids)
        query_states, key_states = modeling_qwen3.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_values is not None:
            cache_length = (
                len(past_key_values.key_cache)
                if hasattr(past_key_values, "key_cache")
                else 0
            )
            if use_cache:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, layer_idx
                )
            elif cache_length > layer_idx:
                key_states = torch.cat(
                    [past_key_values.key_cache[layer_idx], key_states], dim=-2
                )
                value_states = torch.cat(
                    [past_key_values.value_cache[layer_idx], value_states], dim=-2
                )

        attn_output, _ = modeling_qwen3.eager_attention_forward(
            layers[0].self_attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=layers[0].self_attn.scaling,
        )

        attn_output = attn_output.view(batch_size, sum(seq_len_list), -1)
        layer_embeds_list = []
        start_idx = 0

        for module_idx, (layer, input_embeds) in enumerate(
            zip(layers, input_embeds_list)
        ):
            seq_len = seq_len_list[module_idx]
            if seq_len == 0:
                layer_embeds_list.append(None)
                continue

            attn_embeds = attn_output[:, start_idx : start_idx + seq_len, :]
            start_idx += seq_len

            attn_embeds = layer.self_attn.o_proj(attn_embeds)
            residual_attn_embeds = input_embeds + attn_embeds
            postnorm_embeds = layer.post_attention_layernorm(residual_attn_embeds)

            if layer.mlp.gate_proj.weight.dtype == torch.bfloat16:
                postnorm_embeds = postnorm_embeds.to(torch.bfloat16)

            mlp_embeds = layer.mlp(postnorm_embeds)
            residual_mlp_embeds = residual_attn_embeds + mlp_embeds
            layer_embeds_list.append(residual_mlp_embeds)

        return layer_embeds_list

    def _merged_attention_forward(
        self,
        module_list: List[nn.Module],
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values: DynamicCache | None = None,
        input_embeds_list: List[torch.FloatTensor] = None,
        use_cache: bool = True,
    ):
        """Forward pass with merged attention across VLM and action expert."""
        for layer_idx in range(len(module_list[0].layers)):
            input_embeds_list = self._compute_merged_layer(
                layer_idx,
                module_list,
                input_embeds_list,
                position_ids,
                past_key_values,
                attention_mask,
                use_cache,
            )

        # Final layer norms
        decoder_embeds_list = []
        for module, input_embeds in zip(module_list, input_embeds_list):
            if input_embeds is not None:
                input_embeds = module.norm(input_embeds)
            decoder_embeds_list.append(input_embeds)

        return decoder_embeds_list, past_key_values

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images using vision module.

        Uses the model's embed_image method which follows dm0 architecture.
        """
        return self.model.embed_image(images)

    def get_prefix_hidden_states(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.FloatTensor] = None,
        image_masks: Optional[torch.BoolTensor] = None,
    ):
        """Get prefix hidden states (images + language).

        Returns:
            Tuple of (hidden_states, padding_mask, attn_mask).
        """
        hidden_states_list = []
        padding_mask_list = []
        attn_mask_list = []

        images = images.transpose(0, 1)
        image_masks = image_masks.transpose(0, 1)

        for image, image_mask in zip(images, image_masks):
            image_hidden_states = self.encode_images(image)
            batch_size, num_img_tokens = image_hidden_states.shape[:2]

            hidden_states_list.append(image_hidden_states)
            padding_mask_list.append(
                image_mask.unsqueeze(1).expand(batch_size, num_img_tokens)
            )
            attn_mask_list += [1] * num_img_tokens

        if input_ids is not None:
            text_hidden_states = self.model.embed_language_tokens(input_ids)
            hidden_states_list.append(text_hidden_states)
            padding_mask_list.append(attention_mask)

            num_lang_tokens = text_hidden_states.shape[1]
            attn_mask_list += [1] * num_lang_tokens

        hidden_states = torch.cat(hidden_states_list, dim=1)
        padding_mask = torch.cat(padding_mask_list, dim=1)
        attn_mask = torch.tensor(
            attn_mask_list, device=hidden_states.device, dtype=torch.int32
        )

        # Expand attn_mask to batch size
        attn_mask = attn_mask.unsqueeze(0).expand(padding_mask.shape[0], -1)

        return hidden_states, padding_mask, attn_mask

    def get_suffix_hidden_states(
        self,
        noisy_actions: torch.FloatTensor,
        time: torch.FloatTensor,
        progress: Optional[torch.FloatTensor] = None,
    ):
        """Get suffix hidden states (noisy actions + time + optional progress).

        Returns:
            Tuple of (hidden_states, padding_mask, attn_mask).
        """
        # Time embedding using sinusoidal encoding
        time_embeddings = posemb_sincos(
            time,
            self.model.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
        )
        time_embeddings = time_embeddings.to(noisy_actions.dtype)

        # Action embedding
        action_hidden_states = self.model.action_in_proj(noisy_actions)

        if progress is not None:
            # Progress embedding
            progress_emb = self.model.progress_in_proj(progress)
            # Fuse action + progress
            action_hidden_states = torch.cat([progress_emb, action_hidden_states], dim=1)

        # Fuse time + action
        time_embeddings_expanded = time_embeddings[:, None, :].expand_as(
            action_hidden_states
        )
        fused_hidden_states = torch.cat(
            [action_hidden_states, time_embeddings_expanded], dim=2
        )

        # MLP fusion
        x = self.model.action_time_mlp_in(fused_hidden_states)
        x = F.silu(x)
        hidden_states = self.model.action_time_mlp_out(x)

        batch_size, action_len = hidden_states.shape[:2]

        # Padding mask for actions is all ones (valid)
        padding_mask = torch.ones(
            batch_size, action_len, device=time.device, dtype=torch.bool
        )

        # Attention mask: first token attends, rest are causal
        attn_mask_list = [1] + ([0] * (action_len - 1))
        attn_mask = torch.tensor(
            attn_mask_list, device=hidden_states.device, dtype=torch.int32
        )
        attn_mask = attn_mask.unsqueeze(0).expand(batch_size, -1)

        return hidden_states, padding_mask, attn_mask

    @torch.no_grad()
    def inference_action(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        states: Optional[torch.FloatTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        image_masks: Optional[torch.BoolTensor] = None,
        progress: Optional[torch.FloatTensor] = None,
        diffusion_steps: int = 10,
        **kwargs,
    ):
        """Inference action using Euler sampling with optional progress prediction."""
        batch_size = states.shape[0]
        device = states.device
        dtype = states.dtype

        # Preprocess progress
        begin_progress = None
        if progress is not None:
            if progress.ndim == 2:
                begin_progress = progress.unsqueeze(1)
            elif progress.ndim == 3:
                begin_progress = progress

        dt = -1.0 / diffusion_steps
        noise = torch.normal(
            0,
            1,
            size=(batch_size, self.model.config.chunk_size, self.config.action_dim),
            device=device,
            dtype=dtype,
        )

        time = torch.tensor(1.0, device=device, dtype=dtype)

        # Embed prefix and compute KV cache
        prefix_hidden_states, prefix_padding_mask, prefix_attn_mask = (
            self.get_prefix_hidden_states(
                input_ids, attention_mask, images, image_masks
            )
        )

        # Build attention mask
        prefix_attn_mask_2d = make_attn_mask_2d(
            padding_mask=prefix_padding_mask, attn_mask=prefix_attn_mask
        )
        prefix_attn_mask_4d = make_attn_mask_4d(
            prefix_attn_mask_2d, dtype=prefix_hidden_states.dtype
        )
        positions = torch.cumsum(prefix_padding_mask, dim=1) - 1

        module_list = [
            self.model.llm,
            self.model.action_expert.model,
        ]

        _, kv_cache = self._merged_attention_forward(
            module_list=module_list,
            attention_mask=prefix_attn_mask_4d,
            position_ids=positions,
            past_key_values=DynamicCache(),
            input_embeds_list=[prefix_hidden_states, None],
            use_cache=True,
        )

        # Euler sampling loop
        end_progress_list = []
        while time >= -dt / 2:
            noise, time, end_progress = self._denoise_step(
                x_t=noise,
                time=time,
                dt=dt,
                batch_size=batch_size,
                prefix_padding_mask=prefix_padding_mask,
                prefix_attn_mask=prefix_attn_mask,
                module_list=module_list,
                kv_cache=kv_cache,
                begin_progress=begin_progress,
            )
            if end_progress is not None:
                end_progress_list.append(end_progress)

        if progress is not None:
            # Return minimum progress across all steps
            end_progress, _ = torch.min(torch.stack(end_progress_list), dim=0)
            return noise, end_progress
        else:
            return noise

    def _denoise_step(
        self,
        x_t: torch.Tensor,
        time: torch.Tensor,
        dt: float,
        batch_size: int,
        prefix_padding_mask: torch.Tensor,
        prefix_attn_mask: torch.Tensor,
        module_list: List[torch.nn.Module],
        kv_cache: DynamicCache,
        begin_progress: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Single denoising step for Euler sampling with optional progress prediction.

        Args:
            x_t: Current noisy actions [B, T, D].
            time: Current time scalar.
            dt: Time step (negative).
            batch_size: Batch size.
            prefix_padding_mask: Padding mask for prefix [B, P].
            prefix_attn_mask: Attention mask for prefix [B, P].
            module_list: List of model modules [llm, action_expert].
            kv_cache: KV cache from prefix computation.
            begin_progress: Optional progress input [B, 1, 1].

        Returns:
            Tuple of (updated x_t, updated time, end_progress or None).
        """
        # Embed suffix
        suffix_hidden_states, suffix_padding_mask, suffix_attn_mask = (
            self.get_suffix_hidden_states(
                x_t, time.broadcast_to(batch_size), begin_progress
            )
        )

        # Build suffix attention mask
        suffix_attn_mask_2d = make_suffix_attn_mask_2d(
            suffix_padding_mask=suffix_padding_mask,
            suffix_attn_mask=suffix_attn_mask,
            prefix_padding_mask=prefix_padding_mask,
            prefix_attn_mask=prefix_attn_mask,
        )
        full_attn_mask_4d = make_attn_mask_4d(
            suffix_attn_mask_2d, dtype=suffix_hidden_states.dtype
        )

        # Positions
        prefix_offsets = torch.sum(prefix_padding_mask, dim=-1)[:, None]
        full_positions = prefix_offsets + torch.cumsum(suffix_padding_mask, dim=1) - 1

        (_, suffix_out), _ = self._merged_attention_forward(
            module_list=module_list,
            attention_mask=full_attn_mask_4d,
            position_ids=full_positions,
            past_key_values=kv_cache,
            input_embeds_list=[None, suffix_hidden_states],
            use_cache=False,
        )

        v_t = self.model.action_out_proj(suffix_out[:, -self.model.config.chunk_size :])

        # Progress prediction
        end_progress = None
        if begin_progress is not None:
            end_progress = self.model.progress_out_proj(
                suffix_out[:, -self.model.config.chunk_size - 1 : -self.model.config.chunk_size]
            )

        return x_t + v_t * dt, time + dt, end_progress
