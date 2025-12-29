from transformers import AutoConfig, AutoModel

from .transformers_pi05.gemma.configuration_gemma import AdaRMSGemmaConfig
from .transformers_pi05.gemma.modeling_gemma import AdaRMSGemmaModel

AutoConfig.register("adarms_gemma", AdaRMSGemmaConfig)
AutoModel.register(AdaRMSGemmaConfig, AdaRMSGemmaModel)
