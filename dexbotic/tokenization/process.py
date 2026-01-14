from typing import Dict, List, Sequence

import transformers
import numpy as np

from dexbotic.constants import DEFAULT_IMAGE_TOKEN
from dexbotic.data.dataset.tokenization import Tokenization
from dexbotic.tokenization import tokenization as tokenization_lib


def _process(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False,
        chat_template: str = "dexbotic",
):
    if chat_template == "dexbotic":
        return tokenization_lib.tokenize_dexbotic(
            sources=sources,
            tokenizer=tokenizer,
            has_image=has_image,
            chat_template=chat_template,
        )
    else:
        raise ValueError(f"Unsupported chat template: {chat_template}")


def llava_multi_image_map_fn(conversations):
    messages = conversations

    for msg in messages:
        if DEFAULT_IMAGE_TOKEN in msg['value']:
            # move the image token to the beginning of the sentence
            msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
            msg['value'] = DEFAULT_IMAGE_TOKEN + '\n' + msg['value']
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
