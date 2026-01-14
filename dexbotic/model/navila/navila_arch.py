import inspect
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import CONFIG_MAPPING, PretrainedConfig

from dexbotic.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from dexbotic.model.dexbotic_arch import (
    CausalLMOutputDexbotic,
    DexboticConfig,
    DexboticForCausalLM,
    DexboticVLMModel,
)
from dexbotic.model.navila.loss import soft_cross_entropy


class NaVILAConfig(DexboticConfig):
    model_type = "dexbotic_navila"
    chat_template: Optional[str] = "llama_3"
    llm_config: str | PretrainedConfig
    mm_projector_type: str = "mlp_downsample"
    mm_vision_tower: str = "google/siglip-so400m-patch14-384"


class NaVILAModel(DexboticVLMModel):
    def __init__(self, config: NaVILAConfig):
        config.llm_config = CONFIG_MAPPING[config.llm_config["model_type"]](
            **config.llm_config
        )
        super().__init__(config)

    def initialize_model(self, extra_config: dict):
        super().initialize_model(extra_config)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        image_features = self.mm_vision_module(images)
        image_features = self.mm_projector_module(image_features)
        return image_features

    def _prepare_inputs_labels_for_multimodal(
        self,
        input_ids: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        cache_position: Optional[torch.Tensor],
        images: Optional[torch.Tensor],
    ) -> tuple:
        """Override to fix index error: use batch_idx instead of cur_image_idx"""
        if input_ids.shape[1] == 1:
            return self._prepare_inputs_labels_for_multimodal_decode(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                cache_position,
                images,
            )

        vision = self.mm_vision_module
        if vision is None or images is None or input_ids.shape[1] == 1:
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
                cache_position,
            )

        image_features = self._extract_vision_features(images)
        _labels, _position_ids, _attention_mask = labels, position_ids, attention_mask

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        new_input_embeds = []
        new_labels = []

        # Fix: use batch_idx instead of cur_image_idx
        for batch_idx, (cur_input_ids, cur_labels) in enumerate(zip(input_ids, labels)):
            (
                cur_new_input_embeds,
                cur_new_labels,
                _,
            ) = self._insert_multimodal_embeds_per_batch(
                image_features, cur_input_ids, cur_labels, batch_idx
            )
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        tokenizer_model_max_length = getattr(
            self.config, "tokenizer_model_max_length", None
        )
        if tokenizer_model_max_length is not None:
            new_input_embeds = [
                x[:tokenizer_model_max_length] for x in new_input_embeds
            ]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        (
            new_input_embeds_padded,
            new_labels_padded,
            attention_mask,
            position_ids,
        ) = self._pad_multimodal_embeds_per_batch(
            new_input_embeds, new_labels, attention_mask, position_ids
        )
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        new_labels = None if _labels is None else new_labels_padded
        attention_mask = (
            None
            if _attention_mask is None
            else attention_mask.to(dtype=_attention_mask.dtype)
        )
        position_ids = None if _position_ids is None else position_ids

        cache_position = (
            None
            if (_attention_mask is None or cache_position is None)
            else torch.arange(attention_mask.shape[1], device=attention_mask.device)
        )

        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
            cache_position,
        )

    def _insert_multimodal_embeds_per_batch(
        self, image_features, cur_input_ids, cur_labels, cur_image_idx
    ):
        """Override: fix index error by using batch_idx (passed as cur_image_idx)"""
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        batch_idx = min(cur_image_idx, image_features.shape[0] - 1)  # Safety clamp
        batch_image_features = image_features[batch_idx]

        if num_images == 0:
            cur_input_embeds_1 = self.backbone.embed_tokens(cur_input_ids)
            cur_input_embeds = torch.cat(
                [cur_input_embeds_1, batch_image_features[0:0]], dim=0
            )
            return cur_input_embeds, cur_labels, cur_image_idx

        image_positions = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
        image_token_indices = [-1] + image_positions + [cur_input_ids.shape[0]]

        cur_input_ids_noim = []
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(
                cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]]
            )
            cur_labels_noim.append(
                cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]]
            )

        split_sizes = [x.shape[0] for x in cur_labels_noim]
        cur_input_embeds = self.backbone.embed_tokens(torch.cat(cur_input_ids_noim))
        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)

        cur_new_input_embeds = []
        cur_new_labels = []

        for i in range(num_images + 1):
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            cur_new_labels.append(cur_labels_noim[i])

            if i < num_images:
                if num_images == 1:
                    cur_image_features = batch_image_features
                else:
                    tokens_per_image = batch_image_features.shape[0] // num_images
                    cur_image_features = batch_image_features[
                        i * tokens_per_image : (i + 1) * tokens_per_image
                    ]

                cur_new_input_embeds.append(cur_image_features)
                cur_new_labels.append(
                    torch.full(
                        (cur_image_features.shape[0],),
                        IGNORE_INDEX,
                        device=cur_labels.device,
                        dtype=cur_labels.dtype,
                    )
                )

        return torch.cat(cur_new_input_embeds), torch.cat(cur_new_labels), cur_image_idx


class NaVILAForCausalLM(DexboticForCausalLM):
    config_class = NaVILAConfig
    _tied_weights_keys = []

    def _real_init(self, config: NaVILAConfig):
        self.model = NaVILAModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def __init__(self, config: NaVILAConfig):
        super().__init__(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        config = kwargs.get("config", None)
        if config is None:
            trust_remote = kwargs.pop("trust_remote_code", True)
            config = cls.config_class.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=trust_remote
            )
            kwargs["trust_remote_code"] = trust_remote

        if hasattr(config, "tie_word_embeddings"):
            config.tie_word_embeddings = False

        text_cfg = (
            config.get_text_config(decoder=True)
            if hasattr(config, "get_text_config")
            else None
        )
        if (
            text_cfg is not None
            and text_cfg is not config
            and hasattr(text_cfg, "tie_word_embeddings")
        ):
            text_cfg.tie_word_embeddings = False
        kwargs["config"] = config
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    def repack_multimodal_data(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        inputs_embeds,
        labels,
    ):
        """
        Repack multimodal data for sequence packing optimization.
        This is a simplified version that works without sequence parallelism.
        For full functionality with sequence parallelism, you may need to implement
        the full repack logic similar to NaVILA's implementation.
        """
        # Simplified repack: reorder sequences by length for better packing
        if inputs_embeds is None or attention_mask is None:
            return (
                None,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                None,
            )

        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        sorted_seqlens_in_batch, sorted_idx = torch.sort(
            seqlens_in_batch, descending=True
        )
        max_seqlen = inputs_embeds.shape[1]

        new_inputs_embeds = []
        new_position_ids = []
        new_labels = []

        cur_inputs_embeds = []
        cur_position_ids = []
        cur_labels = []
        cur_batch_len = 0

        for i in range(len(sorted_seqlens_in_batch)):
            cur_seqlen = sorted_seqlens_in_batch[i].item()
            if cur_seqlen + cur_batch_len <= max_seqlen:
                cur_batch_len += cur_seqlen
                # Remove padding on-the-fly
                cur_inputs_embeds.append(
                    inputs_embeds[sorted_idx[i]][attention_mask[sorted_idx[i]]]
                )
                cur_position_ids.append(
                    torch.arange(
                        cur_inputs_embeds[-1].shape[0],
                        device=cur_inputs_embeds[-1].device,
                    )
                )
                cur_labels.append(labels[sorted_idx[i]][attention_mask[sorted_idx[i]]])
            else:
                # Pack current batch and start new one
                if len(cur_inputs_embeds) > 0:
                    new_inputs_embeds.append(torch.cat(cur_inputs_embeds, 0))
                    new_position_ids.append(torch.cat(cur_position_ids, 0))
                    new_labels.append(torch.cat(cur_labels, 0))

                cur_batch_len = cur_seqlen
                cur_inputs_embeds = [
                    inputs_embeds[sorted_idx[i]][attention_mask[sorted_idx[i]]]
                ]
                cur_position_ids = [
                    torch.arange(
                        cur_inputs_embeds[-1].shape[0],
                        device=cur_inputs_embeds[-1].device,
                    )
                ]
                cur_labels = [labels[sorted_idx[i]][attention_mask[sorted_idx[i]]]]

        if len(cur_inputs_embeds) > 0:
            new_inputs_embeds.append(torch.cat(cur_inputs_embeds, 0))
            new_position_ids.append(torch.cat(cur_position_ids, 0))
            new_labels.append(torch.cat(cur_labels, 0))

        # Pad sequences
        new_inputs_embeds = torch.nn.utils.rnn.pad_sequence(
            new_inputs_embeds, batch_first=True, padding_value=0.0
        )
        new_position_ids = torch.nn.utils.rnn.pad_sequence(
            new_position_ids, batch_first=True, padding_value=-1
        )
        new_labels = torch.nn.utils.rnn.pad_sequence(
            new_labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        new_attention_mask = new_position_ids.ne(-1)

        # Sanity check
        assert new_attention_mask.sum() == attention_mask.sum()

        return (
            None,
            new_position_ids,
            new_attention_mask,
            past_key_values,
            new_inputs_embeds,
            new_labels,
            sorted_seqlens_in_batch,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        images: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        seqlens_in_batch: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        image_masks: Optional[torch.BoolTensor] = None,
    ) -> CausalLMOutputDexbotic:
        # self.freezed_module_patch()

        if inputs_embeds is None:
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

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        # TODO: output_hidden_states is not used actually
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Check if the underlying LLM supports sequence packing
        support_packing = (
            "seqlens_in_batch" in inspect.signature(self.model.llm.forward).parameters
        )

        # Repack multimodal data for training if supported
        if self.training and support_packing:
            (
                _,
                new_position_ids,
                new_attention_mask,
                _,
                new_inputs_embeds,
                new_labels,
                sorted_seqlens_in_batch,
            ) = self.repack_multimodal_data(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            )
            if sorted_seqlens_in_batch is None:
                sorted_seqlens_in_batch = seqlens_in_batch
            new_input_ids = None
            past_key_values = None
        else:
            new_attention_mask = attention_mask
            new_position_ids = position_ids
            new_inputs_embeds = inputs_embeds
            new_labels = labels
            if seqlens_in_batch is None and attention_mask is not None:
                sorted_seqlens_in_batch = attention_mask.sum(-1).int()
            else:
                sorted_seqlens_in_batch = seqlens_in_batch
            new_input_ids = input_ids

        forward_kwargs = {
            "input_ids": new_input_ids,
            "attention_mask": new_attention_mask,
            "position_ids": new_position_ids,
            "past_key_values": past_key_values,
            "inputs_embeds": new_inputs_embeds,
            "labels": new_labels,
            "use_cache": use_cache,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        }

        if support_packing:
            forward_kwargs["seqlens_in_batch"] = sorted_seqlens_in_batch

        outputs = self.model.backbone(**forward_kwargs)
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if new_labels is not None:
            # Use soft cross entropy if time_token_ids are configured, otherwise use standard loss
            if (
                self.training
                and hasattr(self.config, "time_token_ids")
                and self.config.time_token_ids
            ):
                loss = soft_cross_entropy(
                    logits,
                    new_labels,
                    soft_tokens=self.config.time_token_ids,
                    std=getattr(self.config, "soft_ce_std", 1.0),
                )
            else:
                loss = self.loss_function(
                    logits, new_labels, self.model.backbone.vocab_size
                )

        return CausalLMOutputDexbotic(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @property
    def mm_projector_prefix(self) -> str:
        return "model.mm_projector"

    @property
    def mm_vision_prefix(self) -> str:
        return "model.mm_vision_tower"
