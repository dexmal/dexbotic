"""DM05 utility functions for cache handling, time embeddings, and masks."""

import math
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache, CacheLayerMixin
from transformers.modeling_layers import GradientCheckpointingLayer

try:
    from transformers.utils import is_torch_flex_attn_available
except ImportError:

    def is_torch_flex_attn_available() -> bool:
        try:
            import torch.nn.attention.flex_attention  # noqa: F401
        except Exception:
            return False
        return True


from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer


def linear_fp32(x: torch.Tensor, linear: torch.nn.Module) -> torch.Tensor:
    """Run a projection in FP32 without changing the stored parameter dtype."""

    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x.float()
        if linear.__class__.__name__ == "ModulesToSaveWrapper":
            linear = (
                linear.original_module
                if linear.disable_adapters
                else linear.modules_to_save[linear.active_adapters[0]]
            )
        if type(linear) is torch.nn.Linear:
            bias = None if linear.bias is None else linear.bias.float()
            return F.linear(x, linear.weight.float(), bias)
        return linear(x)


def is_flash_attention_2_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return True


def is_flex_attention_available() -> bool:
    return is_torch_flex_attn_available()


def _config_value_for_compat(config, field: str):
    if field == "head_dim":
        return getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
    return getattr(config, field, None)


def validate_action_config_compatible(action_config, language_config) -> None:
    fields = (
        "num_hidden_layers",
        "layer_types",
        "max_position_embeddings",
        "rope_parameters",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
    )
    mismatches = []
    for field in fields:
        action_value = _config_value_for_compat(action_config, field)
        language_value = _config_value_for_compat(language_config, field)
        if action_value != language_value:
            mismatches.append(
                f"{field}: action={action_value!r}, language={language_value!r}"
            )
    if mismatches:
        raise ValueError(
            "DM05 action_config must match VLM language config for suffix KV reuse; "
            + "; ".join(mismatches)
        )


class SafeCacheDecoderLayer(GradientCheckpointingLayer):
    """Gradient-checkpointing-safe decoder layer that preserves past key values.

    The upstream ``GradientCheckpointingLayer`` clears ``past_key_values`` when
    gradient checkpointing is enabled. DM05 prefix forward still needs to write
    key-value states into the cache during checkpointed execution so they can be
    reused by suffix forward. ``OverwriteDynamicLayer`` makes repeated writes
    during recomputation safe.

    This class overrides ``__call__`` to keep ``past_key_values`` available in
    gradient-checkpointing mode.
    """

    def __call__(self, *args, **kwargs):
        if self.gradient_checkpointing and self.training:
            return self._gradient_checkpointing_func(
                partial(super(GradientCheckpointingLayer, self).__call__, **kwargs),
                *args,
            )
        return super(GradientCheckpointingLayer, self).__call__(*args, **kwargs)


def patch_decoder_layers(model):
    """Replace model decoder layers with ``SafeCacheDecoderLayer`` variants."""
    PatchedDecoderLayer = type(
        "PatchedGemma3DecoderLayer",
        (SafeCacheDecoderLayer,)
        + tuple(
            b
            for b in Gemma3DecoderLayer.__bases__
            if b is not GradientCheckpointingLayer
        ),
        dict(Gemma3DecoderLayer.__dict__),
    )

    for layer in model.language_model.layers:
        layer.__class__ = PatchedDecoderLayer


class OverwriteDynamicLayer(CacheLayerMixin):
    """Cache layer with overwrite semantics that preserves gradients."""

    is_sliding = False

    def __init__(self):
        super().__init__()
        self.keys = None
        self.values = None
        self.is_initialized = True

    def lazy_initialization(
        self, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> None:
        """No-op because initialization is already handled in ``__init__``."""
        pass

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Overwrite cached tensors while preserving gradients.
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.keys = key_states
        self.values = value_states
        return self.keys, self.values

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        if self.keys is None or self.keys.numel() == 0:
            return 0
        return self.keys.shape[-2]

    def get_max_cache_shape(self) -> int:
        return -1


class VLADynamicCache(Cache):
    """VLA-specific dynamic cache with gradient-preserving overwrite semantics."""

    def __init__(self, config=None):
        if config is not None:
            decoder_config = config.get_text_config(decoder=True)
            layer_types = getattr(decoder_config, "layer_types", None)
            if layer_types is None:
                n_layers = decoder_config.num_hidden_layers
                layer_types = ["full_attention"] * n_layers
            layers = [OverwriteDynamicLayer() for _ in layer_types]
            super().__init__(layers=layers)
        else:
            super().__init__(layer_class_to_replicate=OverwriteDynamicLayer)


def posemb_sincos(
    time: torch.Tensor,
    dim: int,
    min_period: float = 4e-3,
    max_period: float = 256.0,
) -> torch.Tensor:
    """Compute sinusoidal time positional embeddings."""
    fraction = torch.linspace(
        0.0, 1.0, dim // 2, dtype=torch.float32, device=time.device
    )
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None].to(torch.float32)
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


# DM05 represents each history image with a pooled 4x4 grid of Gemma 3 vision
# features. ``<unused0>`` reserves those positions in the language prefix and
# ``<unused1>`` marks history slots that must remain invisible to attention.
HISTORY_IMAGE_TOKEN = "<unused0>"
HISTORY_PAD_TOKEN = "<unused1>"
HISTORY_TOKENS_PER_IMAGE = 16
HISTORY_POOL_SIZE = 4
HISTORY_PAD_TOKEN_ID = 7

assert HISTORY_POOL_SIZE * HISTORY_POOL_SIZE == HISTORY_TOKENS_PER_IMAGE


def mask_history_pad_tokens_in_attention(
    input_ids: torch.LongTensor | None,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if input_ids is None:
        return attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    else:
        attention_mask = attention_mask.clone()
    return attention_mask.masked_fill(input_ids == HISTORY_PAD_TOKEN_ID, 0)


def make_suffix_attn_mask(
    input_ids: torch.Tensor,
    prefix_len: int,
    suffix_len: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    pad_token_id: int = 0,
    invisible_prefix_token_ids: tuple[int, ...] = (),
    labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the attention mask from suffix tokens to prefix and suffix tokens.

    Each suffix token can attend to:
    - Non-padding tokens in the prefix.
    - All suffix tokens with full attention.

    Returns:
        A 4D attention mask with shape
        ``[B, 1, suffix_len, prefix_len + suffix_len]``.
    """
    NEG_INF = -2.3819763e38

    prefix_mask = torch.zeros(
        batch_size, suffix_len, prefix_len, device=device, dtype=dtype
    )
    prefix_ids = input_ids[:, :prefix_len]
    pad_mask = prefix_ids == pad_token_id
    for token_id in invisible_prefix_token_ids:
        pad_mask = pad_mask | (prefix_ids == token_id)
    if labels is not None:
        # The action expert must not read the supervised answer from the KV cache.
        pad_mask = pad_mask | (labels[:, :prefix_len] != -100)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, suffix_len, -1)
    prefix_mask = prefix_mask.masked_fill(pad_mask, NEG_INF)

    suffix_mask = torch.zeros(
        batch_size, suffix_len, suffix_len, device=device, dtype=dtype
    )

    mask = torch.cat([prefix_mask, suffix_mask], dim=2)
    return mask.unsqueeze(1)
