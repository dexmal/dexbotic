import copy
import math
import random
import re
from typing import Callable, Dict, List

import numpy as np
import torch

from dexbotic.constants import DEFAULT_IMAGE_TOKEN
from dexbotic.data.dataset.dex_dataset import DexDataset, load_jsonl
from dexbotic.data.dataset.transform.common import ToTensor


class DexNavilaDataset(DexDataset):
    default_keys = ["input_ids", "labels", "image", "image_masks"]

    def __init__(
        self,
        data_args,
        tokenization_func: Callable[[List[Dict], bool], dict[str, torch.Tensor]],
        action_process_func=None,
        image_process_func=None,
        **kwargs,
    ):
        super().__init__(
            data_args,
            tokenization_func,
            action_process_func,
            image_process_func,
            **kwargs,
        )
        self.data_args = data_args

    def _build_dataset_index(self):
        total_samples = 0
        data_indices = []
        global_index = []
        file_name_map = {}
        dataset_map = {}
        file_id = 0
        dataset_id = 0
        for dataset_info in self.datasets_info:
            data_path = dataset_info["annotations"]
            data_path_prefix = dataset_info.get("data_path_prefix", "")
            frequency = dataset_info["frequency"]
            meta_data = dataset_info["meta_data"]

            if data_path not in dataset_map:
                dataset_map[data_path] = {
                    "id": dataset_id,
                    "meta_data": meta_data,
                    "data_path_prefix": data_path_prefix,
                }
                dataset_id += 1
            dataset_index = dataset_map[data_path]["id"]

            data_index = self._get_index_cache(data_path)["data"]
            data_index = list(data_index.items())
            random.shuffle(data_index)

            sampled_data_index = []
            while frequency > 0:
                if frequency >= 1:
                    sampled_data_index.extend(copy.deepcopy(data_index))
                else:
                    sampled_data_index.extend(
                        copy.deepcopy(
                            data_index[: math.ceil(len(data_index) * frequency)]
                        )
                    )
                frequency -= 1

            for jsonl_file, num_samples in sampled_data_index:
                if jsonl_file not in file_name_map:
                    file_name_map[jsonl_file] = file_id
                    file_id += 1
                file_index = file_name_map[jsonl_file]
                global_index.append((dataset_index, file_index, num_samples - 1))
            total_samples += len(sampled_data_index)
            data_indices.extend(sampled_data_index)

        self.global_index = global_index
        self.file_name_map = {v: k for k, v in file_name_map.items()}
        self.dataset_map = {
            v["id"]: {
                "data_path": k,
                "meta_data": v["meta_data"],
                "data_path_prefix": v["data_path_prefix"],
            }
            for k, v in dataset_map.items()
        }
        self.total_samples = total_samples

    @staticmethod
    def _sanitize_instruction(instruction: str) -> str:
        instruction = instruction.replace("\r\n", " ").replace("\n", " ")
        instruction = re.sub(
            r"(?<=\.\s)([a-z])", lambda x: x.group().upper(), instruction.capitalize()
        )
        instruction = re.sub(r"\s+\.", ".", instruction)
        return instruction

    def _build_conversation(
        self, instruction: str, answer: str, num_frames: int
    ) -> List[Dict]:
        image_placeholders = ""
        if num_frames > 1:
            image_placeholders = (DEFAULT_IMAGE_TOKEN + "\n") * (num_frames - 1)

        question = (
            "Imagine you are a robot programmed for navigation tasks. You have been given a video "
            f"of historical observations {image_placeholders}, and current observation {DEFAULT_IMAGE_TOKEN}\n. "
            f'Your assigned task is: "{instruction}" '
            "Analyze this series of images to decide your next action, which could be turning left or right by a specific "
            "degree, moving forward a certain distance, or stop if the task is completed."
        )

        conversation = [
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer},
        ]
        return conversation

    def unsafe_getitem(self, idx) -> dict:
        dataset_index, file_index, frame_index = self.global_index[idx]
        jsonl_file = self.file_name_map[file_index]
        dataset_info = self.dataset_map[dataset_index]
        dataset = dataset_info["data_path"]
        meta_data = dataset_info["meta_data"]
        data_path_prefix = dataset_info["data_path_prefix"]
        episode_data_list = load_jsonl(jsonl_file, parse=True)

        episode_length = len(episode_data_list)
        assert episode_length > 0, f"Episode length is 0 for {jsonl_file}"
        if episode_length < self.num_images:
            frame_indices = list(range(episode_length))
        else:
            latest_index = episode_length - 1
            sampled_indices = np.linspace(
                0, latest_index, num=self.num_images - 1, endpoint=False, dtype=int
            ).tolist()
            frame_indices = sampled_indices + [latest_index]

        meta_data.update(
            dict(
                fram_indicies=frame_indices,
                jsonl_file=jsonl_file,
                dataset=dataset,
                num_images=self.num_images,
                images_keys=self.images_keys,
                depths_keys=self.depths_keys,
                load_depth=self.load_depth,
                data_path_prefix=data_path_prefix,
            )
        )

        # process the episode data
        episode_data_list = self.action_process_func(
            episode_data_list, meta_data=meta_data
        )

        # Get data for all frames
        all_rgb_data = []
        for fidx in frame_indices:
            frame_data = copy.deepcopy(episode_data_list[fidx])
            rgb_data = frame_data.get("rgb_data", [])
            all_rgb_data.append(rgb_data[0] if len(rgb_data) > 0 else None)
        if len(all_rgb_data) < self.num_images:
            all_rgb_data = [None] * (self.num_images - len(all_rgb_data)) + all_rgb_data

        data = episode_data_list[-1]
        data.update({"meta_data": meta_data})
        return_dict = {}

        # Process all images (history + current)
        pixel_values = []
        for image_process_func, rgb_data in zip(
            self.image_process_func, all_rgb_data, strict=True
        ):
            pixel_values.append(image_process_func(rgb_data))

        return_dict["image"] = torch.stack(pixel_values, dim=0)

        if "conversations" not in data:
            instruction = self._sanitize_instruction(data["prompt"])
            answer = data["answer"]
            data["conversations"] = self._build_conversation(
                instruction, answer, len(all_rgb_data)
            )
        conversations = data["conversations"]

        tokenized_dict = self.tokenization_func(
            conversations=conversations,
            has_image=True,
        )
        return_dict["input_ids"] = tokenized_dict["input_ids"]
        return_dict["labels"] = tokenized_dict["labels"]

        if "label_masks" in tokenized_dict:
            return_dict["label_masks"] = tokenized_dict["label_masks"]

        # 5. extract other data and convert to tensor (use current frame's data)
        other_keys = [_ for _ in self.data_keys if _ not in return_dict]
        return_dict.update(self.key_extract_func(data, other_keys))

        return_dict.update(
            {"image_masks": torch.ones(self.num_images, dtype=torch.bool)}
        )
        return_dict = ToTensor()(return_dict)

        return return_dict
