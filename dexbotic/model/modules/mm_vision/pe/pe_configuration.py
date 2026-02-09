from dataclasses import dataclass, field
from typing import Optional

from dexbotic.model.modules.mm_vision.pe.pe_model import PerceptionEncoderWithDownsample


@dataclass
class PerceptionEncoderConfig:
    patch_size: int
    width: int
    layers: int
    heads: int
    mlp_ratio: float
    output_dim: Optional[int]

    ls_init_value: float = None
    drop_path: float = 0.0

    image_size: int = 224
    use_abs_posemb: bool = True
    use_cls_token: bool = False
    use_rope2d: bool = True

    pool_type: str = "attn"
    attn_pooler_heads: int = 8

    use_ln_pre: bool = True
    use_ln_post: bool = True

    layer_types: list[str] = field(default_factory=list)
    sliding_window_size: int = -1

    def build_model(self):
        model = PerceptionEncoderWithDownsample(
            patch_size=self.patch_size,
            width=self.width,
            layers=self.layers,
            heads=self.heads,
            mlp_ratio=self.mlp_ratio,
            use_ln_post=self.use_ln_post,
            use_ln_pre=self.use_ln_pre,
            ls_init_value=self.ls_init_value,
            drop_path=self.drop_path,
            image_size=self.image_size,
            use_abs_posemb=self.use_abs_posemb,
            use_rope2d=self.use_rope2d,
            use_cls_token=self.use_cls_token,
            output_dim=self.output_dim,
            attn_pooler_heads=self.attn_pooler_heads,
            pool_type=self.pool_type,
            layer_types=self.layer_types,
            sliding_window_size=self.sliding_window_size,
        )
        return model


@dataclass
class PE_LANG_L14_728(PerceptionEncoderConfig):
    image_size: int = 728
    patch_size: int = 14
    width: int = 1024
    layers: int = 23
    heads: int = 16
    mlp_ratio: float = 4.0
    pool_type: str = "none"
    output_dim: Optional[int] = None
    use_cls_token: bool = True
    use_ln_post: bool = False
    ls_init_value: float = 0.1


def get_config(config_name: str) -> PerceptionEncoderConfig:
    if config_name == "pe_lang_l14_728":
        return PE_LANG_L14_728()
    else:
        raise ValueError(f"Unknown configuration name: {config_name}")
