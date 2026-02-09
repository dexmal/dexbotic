from typing import Dict, List, Sequence

import transformers
import numpy as np

from dexbotic.constants import DEFAULT_IMAGE_TOKEN
from dexbotic.data.dataset.tokenization import Tokenization
from dexbotic.tokenization import conversation as conversation_lib
from dexbotic.tokenization import tokenization as tokenization_lib


def _process(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False,
        chat_template: str = "dexbotic",
):
    if chat_template in ["dexbotic", "step"]:
        return tokenization_lib.tokenize_dexbotic(
            sources=sources,
            tokenizer=tokenizer,
            has_image=has_image,
            chat_template=chat_template,
        )
    else:
        raise ValueError(f"Unsupported chat template: {chat_template}")


def llava_multi_image_map_fn(conversations, mode="dexbotic"):
    messages = conversations

    for msg in messages:
        if DEFAULT_IMAGE_TOKEN in msg['value']:
            # move the image token to the beginning of the sentence
            msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
            if mode == 'step':
                msg['value'] = msg['value'] + f"<im_start>{DEFAULT_IMAGE_TOKEN}<im_end>"
            else:
                msg['value'] = DEFAULT_IMAGE_TOKEN + "\n" + msg['value']
            msg['value'] = msg['value'].strip()

    return conversations


def process_data_item(
    conversations: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    chat_template: str,
    has_image: bool
) -> Dict:
    conversations = llava_multi_image_map_fn(conversations)
    text_dict = _process(
        sources=[conversations],
        tokenizer=tokenizer,
        has_image=has_image,
        chat_template=chat_template,
    )
    data_dict = dict(input_ids=text_dict["input_ids"][0], labels=text_dict["labels"][0])
    return data_dict


class LLMTokenization(Tokenization):
    def __init__(self, tokenizer, data_args):
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __call__(self, conversations: List[Dict], has_image: bool) -> Dict:
        data_dict = process_data_item(
            conversations=conversations,
            tokenizer=self.tokenizer,
            chat_template=self.data_args.chat_template,
            has_image=has_image,
        )
        return data_dict


class NaVILATokenization(Tokenization):
    """
    Tokenization class for NaVILA dataset.

    Directly processes conversation format without using chat templates.
    Preserves all <image> tokens in their original positions.
    """

    def __init__(self, tokenizer, data_args):
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __call__(self, conversations: List[Dict], has_image: bool) -> Dict:
        from dexbotic.constants import IGNORE_INDEX
        from dexbotic.tokenization.tokenization import tokenizer_image_token

        human_msg = conversations[0]["value"]
        gpt_msg = conversations[1]["value"] if len(conversations) > 1 else ""

        # Tokenize full prompt: human_msg + gpt_msg + "\n"
        prompt = human_msg + gpt_msg + "\n"
        input_ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt")
        # Ensure 1D tensor [seq_len] for DataCollator (it will add batch dimension)
        if input_ids.dim() > 1:
            input_ids = input_ids.squeeze()

        # Create labels and mask human question part
        labels = input_ids.clone()
        human_len = len(tokenizer_image_token(human_msg, self.tokenizer))
        labels[:human_len] = IGNORE_INDEX

        # Mask padding tokens
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if pad_token_id is not None:
            labels[input_ids == pad_token_id] = IGNORE_INDEX

        return {"input_ids": input_ids, "labels": labels}


class Pi0Tokenization(Tokenization):
    def __init__(self, tokenizer: transformers.GemmaTokenizer, *args, **kwargs):
        self.tokenizer = tokenizer
        self._max_len = tokenizer.model_max_length

    def __call__(self, conversations: List[Dict], **kwargs):
        prompt = conversations[0]["value"]
        cleaned_prompt = prompt.strip().replace('\n', ' ').replace('_', ' ')
        tokens = self.tokenizer.sp_model.encode(cleaned_prompt, add_bos=True) + self.tokenizer.sp_model.encode("\n")
        tokens = tokens[: self._max_len]
        tokens += [0] * (self._max_len - len(tokens))
        return {"input_ids": np.asarray(tokens), "labels": np.asarray(tokens)}


class DM0Tokenization(Tokenization):
    """DM0 tokenization matching OpenPI's DM0Tokenizer format for SFT mode.

    Uses "step" conversation template with USER/ASSISTANT roles.
    Format: "System prompt USER: prompt ASSISTANT: <empty>"
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        chat_template: str = "step",
        *args,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self._max_len = tokenizer.model_max_length
        self.chat_template = chat_template

    def __call__(self, conversations: List[Dict], **kwargs) -> Dict:
        """Tokenize conversations in SFT format.

        Args:
            conversations: List of conversation turns, e.g. [{"from": "human", "value": "prompt"}]

        Returns:
            Dict with input_ids, labels, token_mask, ar_mask, loss_mask
        """
        # Get conversation template
        conv = conversation_lib.conv_templates[self.chat_template].copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        seps = {conv.roles[0]: conv.sep, conv.roles[1]: conv.sep2}

        # Build system prompt
        system_prompt = f"{conv.system}{conv.sep}"
        tokens = list(self.tokenizer.encode(system_prompt, add_special_tokens=False))
        token_mask = [True] * len(tokens)
        ar_mask = [1] * len(tokens)  # Causal attention for all
        loss_mask = [False] * len(tokens)  # No loss on system prompt

        # Remove empty trailing assistant turn if present (requested for OpenPI alignment)
        conversations = list(conversations)
        if (
            conversations
            and conversations[-1].get("from") == "gpt"
            and not conversations[-1].get("value")
        ):
            conversations.pop()

        # Process each conversation turn
        for i, msg in enumerate(conversations):
            role_key = msg.get("from", "human")
            if role_key not in roles:
                continue
            role = roles[role_key]
            text = msg.get("value", "")
            if text is None:
                text = ""
            text = text.strip().replace("\n", " ")
            sep = seps[role]

            # Role token
            role_str = f"{role}: "
            role_tokens = list(
                self.tokenizer.encode(role_str, add_special_tokens=False)
            )
            tokens.extend(role_tokens)
            token_mask.extend([True] * len(role_tokens))
            ar_mask.extend([1] * len(role_tokens))
            loss_mask.extend([False] * len(role_tokens))

            # Content + separator
            if text:
                content_str = f"{text}{sep}"
            else:
                content_str = ""  # Empty response for assistant in SFT
            content_tokens = list(
                self.tokenizer.encode(content_str, add_special_tokens=False)
            )
            tokens.extend(content_tokens)
            token_mask.extend([True] * len(content_tokens))
            ar_mask.extend([1] * len(content_tokens))

            # Loss only on assistant responses
            if role == roles["gpt"]:
                loss_mask.extend([True] * len(content_tokens))
            else:
                loss_mask.extend([False] * len(content_tokens))

        # Pad or truncate to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [self.tokenizer.pad_token_id] * (self._max_len - tokens_len)
            pad_mask = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + pad_mask
            ar_mask = ar_mask + pad_mask
            loss_mask = loss_mask + pad_mask
        else:
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        # Create labels (same as input_ids, with IGNORE_INDEX where loss_mask is False)
        from dexbotic.constants import IGNORE_INDEX

        input_ids = np.asarray(tokens)
        labels = np.where(np.asarray(loss_mask), input_ids, IGNORE_INDEX)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "token_mask": np.asarray(token_mask),
            "ar_mask": np.asarray(ar_mask),
            "loss_mask": np.asarray(loss_mask),
        }
