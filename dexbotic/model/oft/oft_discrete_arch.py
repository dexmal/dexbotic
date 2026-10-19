from typing import List, Optional

import torch
import torch.nn as nn

from dexbotic.model.dexbotic_arch import CausalLMOutputDexbotic

from .oft_arch import OFTConfig, OFTForCausalLM, OFTModel


class OFTDiscreteConfig(OFTConfig):
    model_type = "dexbotic_oft_discrete"
    num_bins: Optional[int] = 256


class OFTDiscreteModel(OFTModel):
    pass


class OFTDiscreteForCausalLM(OFTForCausalLM):
    config_class = OFTDiscreteConfig

    def get_output_embeddings(self):
        return None

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
        noisy_dict: Optional[dict[str, torch.FloatTensor]] = None,
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

        if actions is not None:
            actions = actions.reshape(actions.size(0), -1, self.config.action_dim)
            actions = actions[:, : self.config.chunk_size, :]

        assert (
            "Discrete" in self.config.action_model_type
        ), "This forward method is only for OFT-discrete model."
        if labels is not None:
            # Remove action label tokens from input_ids based on attention_mask
            # Structure: [prefix tokens] + [action tokens (chunk_size * action_dim)] + [suffix token]
            # We want: [prefix tokens] + [suffix token]
            action_token_length = self.config.chunk_size * self.config.action_dim
            batch_size = input_ids.size(0)

            updated_input_ids = []
            discrete_action_labels = []
            updated_attention_mask = []

            for i in range(batch_size):
                # Find the actual non-padding length for this sample
                non_padding_length = attention_mask[i].sum().item()

                # Keep prefix tokens and last token, remove action tokens in between
                prefix_length = non_padding_length - action_token_length - 1

                # Extract tokens: prefix + suffix
                prefix_tokens = input_ids[i, :prefix_length]
                suffix_token = input_ids[i, non_padding_length - 1 :]
                new_ids = torch.cat([prefix_tokens, suffix_token], dim=0)

                updated_input_ids.append(new_ids)
                discrete_action_labels.append(
                    labels[i, prefix_length : prefix_length + action_token_length]
                )  # Labels remain unchanged

                # Update attention mask
                new_mask = torch.zeros(
                    len(new_ids),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                new_mask[: prefix_length + 1] = 1
                updated_attention_mask.append(new_mask)

            input_ids = torch.stack(updated_input_ids, dim=0)
            discrete_action_labels = torch.stack(discrete_action_labels, dim=0)
            labels = None
            attention_mask = torch.stack(updated_attention_mask, dim=0)
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

        placeholder_action_token_ids = self.model.action_head.action_query.expand(
            inputs_embeds.shape[0], -1
        ).to(self.model.device)
        action_embeds = self.model.llm.get_input_embeddings()(
            placeholder_action_token_ids.long()
        )

        if self.config.use_proprio:
            assert states is not None, "states is required when use_proprio is True"
            state_embeds = self.model.action_head.proprio_projector(states).reshape(
                states.size(0), -1, self.config.hidden_size
            )
            action_embeds = torch.cat([state_embeds, action_embeds], dim=1)

        (
            inputs_embeds,
            attention_mask,
            non_padding_lengths,
        ) = self.insert_action_embedding(inputs_embeds, attention_mask, action_embeds)

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
        action_hidden_states = self.extract_action_hidden_states(
            last_hidden_state, action_embeds.size(1), non_padding_lengths
        )

        if self.config.use_proprio:
            action_hidden_states = action_hidden_states[:, 1:, :]

        loss = None

        # Project to vocabulary logits for parallel prediction
        # Shape: (batch_size, chunk_size * action_dim, vocab_size)
        predicted_actions = self.lm_head(action_hidden_states)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            if actions is not None or labels is not None:
                # Extract action token labels from labels tensor
                # Action tokens are located at positions [non_padding_length : non_padding_length + action_token_length]
                # discrete_action_labels = []
                action_token_length = action_embeds.size(1)
                if self.config.use_proprio:
                    # Skip the first token (proprio state) when extracting action labels
                    action_token_length -= 1

                # Compute cross-entropy loss
                # predicted_actions: (batch_size, chunk_size * action_dim, vocab_size)
                # discrete_action_labels: (batch_size, chunk_size * action_dim)
                predicted_actions_flat = predicted_actions.reshape(
                    -1, predicted_actions.size(-1)
                )
                discrete_action_labels_flat = discrete_action_labels.reshape(-1)

                loss = nn.functional.cross_entropy(
                    predicted_actions_flat,
                    discrete_action_labels_flat,
                    reduction="mean",
                )

        if not return_dict:
            return (
                (loss,) + last_hidden_state if loss is not None else last_hidden_state
            )

        return CausalLMOutputDexbotic(
            loss=loss,
            logits=predicted_actions,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def inference_action(self, input_ids, image_tensor, inference_args={}, **kwargs):
        action_norms = inference_args.get("action_norms")
        states = inference_args.get("states", None)

        # For discrete actions, use parallel decoding instead of autoregressive generation
        # Forward pass to get logits at action token positions
        out_features = self.__call__(
            input_ids, images=image_tensor, use_cache=False, states=states
        )

        # predicted_actions shape: (batch_size, chunk_size * action_dim, vocab_size)
        predicted_logits = out_features.logits

        # Get discrete bin indices via argmax (parallel decoding)
        # Shape: (batch_size, chunk_size * action_dim)
        discrete_action_indices = torch.argmax(
            predicted_logits[:, :, -self.config.num_bins + 1 :], dim=-1
        )

        # Convert discrete indices to continuous actions
        # Shape: (batch_size, chunk_size, action_dim)
        predicted_actions = self.model.action_head.discrete_tokens_to_continuous(
            discrete_action_indices
        )

        actions = predicted_actions[0]

        actions = self._denorm(actions.float().cpu().numpy(), action_norms).tolist()
        return actions

    @torch.no_grad()
    def generate_action(
        self,
        input_ids,
        pixel_values,
        attention_masks,
        temperature,
        inference_args={},
        **kwargs,
    ):
        action_norms = inference_args.get("action_norms")
        states = inference_args.get("states", None)

        assert (
            "Discrete" in self.config.action_model_type
        ), "generate_action is only for discrete action model."
        # For discrete actions, use parallel decoding instead of autoregressive generation
        # Forward pass to get logits at action token positions
        out_features = self.__call__(
            input_ids,
            images=pixel_values,
            attention_mask=attention_masks,
            use_cache=False,
            states=states,
        )

        # predicted_actions shape: (batch_size, chunk_size * action_dim, vocab_size)
        predicted_logits = out_features.logits[..., -self.config.num_bins + 1 :]
        scaled_logits = predicted_logits / temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        probs_flat = probs.reshape(-1, probs.shape[-1])
        sampled_indices_flat = torch.multinomial(probs_flat, num_samples=1)
        discrete_action_indices = sampled_indices_flat.view(
            predicted_logits.shape[0], -1
        )
        reponse_ids = (
            discrete_action_indices + self.config.vocab_size - self.config.num_bins + 1
        )

        predicted_actions = self.model.action_head.discrete_tokens_to_continuous(
            discrete_action_indices
        )
        actions = self._denorm(
            predicted_actions.float().cpu().numpy(), action_norms
        ).tolist()
        return actions, reponse_ids
