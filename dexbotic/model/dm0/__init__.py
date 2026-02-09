"""DM0 Model for Dexbotic.

This module provides the DM0 model based on dm0 architecture,
using Qwen3-based VLM with a separate action expert for flow matching.
"""

from transformers import AutoConfig, AutoModel

from .dm0_arch import DM0Config, DM0ForCausalLM
from .dm0_prog_arch import DM0ProgConfig, DM0ProgForCausalLM

AutoConfig.register("dexbotic_dm0", DM0Config)
AutoModel.register(DM0Config, DM0ForCausalLM)

AutoConfig.register("dexbotic_dm0_prog", DM0ProgConfig)
AutoModel.register(DM0ProgConfig, DM0ProgForCausalLM)
