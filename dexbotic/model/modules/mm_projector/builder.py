import re

import torch
import torch.nn as nn

from dexbotic.exp.utils import require_config_keys


class DownSampleBlock(nn.Module):
    def forward(self, x):
        vit_embeds = x
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.flat_square(vit_embeds)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        return vit_embeds

    def flat_square(self, x):
        n, w, h, c = x.size()
        if w % 2 == 1:
            x = torch.concat(
                [x, torch.zeros((n, 1, h, c), dtype=x.dtype).to(x.device)], dim=1
            ).contiguous()
            n, w, h, c = x.size()
        if h % 2 == 1:
            x = torch.concat(
                [x, torch.zeros((n, w, 1, c), dtype=x.dtype).to(x.device)], dim=2
            ).contiguous()
            n, w, h, c = x.size()
        x = x.view(n, w, int(h / 2), int(c * 2))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h / 2), int(w / 2), int(c * 4))
        return x


@require_config_keys(["mm_projector_type", "mm_hidden_size", "hidden_size"])
def build_vision_projector(config):
    """
    Build the projector module for the vision tower.
    Use the config to parse parameters.
    The config should contain the following parameters:
        - mm_projector_type: the type of the projector module
        - mm_hidden_size: the hidden size of the vision tower
        - hidden_size: the hidden size of the language model
    """
    projector_type = getattr(config, 'mm_projector_type', 'mlp2x_gelu')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    elif projector_type.startswith("linear"):
        linear_nx_match = re.match(r"^linear(\d+)x$", projector_type)
        if linear_nx_match:
            multiplier = int(linear_nx_match.group(1))
            projector_bias = getattr(config, "projector_bias", False)
            return nn.Linear(
                config.mm_hidden_size * multiplier,
                config.hidden_size,
                bias=projector_bias,
            )

    elif projector_type == "mlp_downsample":
        return nn.Sequential(
            DownSampleBlock(),
            nn.LayerNorm(config.mm_hidden_size * 4),
            nn.Linear(config.mm_hidden_size * 4, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    elif projector_type.startswith("mlp"):
        mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            return nn.Sequential(*modules)

    raise ValueError(f'Unknown projector type: {projector_type}')
