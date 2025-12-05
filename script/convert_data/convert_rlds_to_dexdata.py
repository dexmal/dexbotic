
"""
Example script for converting RLDS dataset format to DexData format.

This script demonstrates how to convert RLDS datasets (specifically Libero dataset)
to the DexData format.

NOTE: As different RLDS datasets have different observation keys, this example script
should only be treated as a reference for converting RLDS datasets to DexData format.
The directly usage of this script is not recommended.
"""

import dlimp as dl
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Any, Optional, Union
import numpy as np
import json
import os
import imageio.v2 as iio
import argparse

DATASET_CONFIG = {
    # LIBERO datasets - robotic manipulation with RGB images
    "libero_10_no_noops": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
    },
    "libero_10": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
    },
    "libero_spatial": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
    },
    "libero_object": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
    },
    "libero_goal": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
    }
}

def create_video_writers(episode, episode_key, dataset_name, video_output_path, frame_rate=10):
    """
    Create video writers for all image and depth streams in the episode.

    Args:
        episode: Episode data
        episode_key: Episode index
        dataset_name: Dataset name
        video_output_path: Video output path
        frame_rate: Video frame rate, default is 10

    Returns:
        tuple: (video_writers dict, video_relative_paths dict, image_keys list, depth_keys list)
    """
    video_writers = {}
    video_relative_paths = {}

    # Get image keys, filter out None observations
    image_keys = [key for key in episode['observation'].keys()
                  if key.startswith('image_') and episode['observation'][key] is not None]

    # Get depth keys, filter out None observations
    depth_keys = [key for key in episode['observation'].keys()
                  if key.startswith('depth_') and episode['observation'][key] is not None]

    # Process images
    for image_key in image_keys:
        video_name = f"episode{str(episode_key)}_{image_key}.mp4"
        video_save_relative_path = os.path.join(dataset_name, video_name)
        video_full_path = os.path.join(video_output_path, video_save_relative_path)

        # Ensure directory exists
        os.makedirs(os.path.dirname(video_full_path), exist_ok=True)

        video_writers[image_key] = iio.get_writer(
            video_full_path,
            format='ffmpeg',
            mode='I',
            fps=frame_rate,
            codec='libx264',
            pixelformat='yuv420p'
        )
        video_relative_paths[image_key] = video_save_relative_path

    # Process depth images
    for depth_key in depth_keys:
        video_name = f"episode{str(episode_key)}_{depth_key}.mp4"
        video_save_relative_path = os.path.join(dataset_name, video_name)
        video_full_path = os.path.join(video_output_path, video_save_relative_path)

        # Ensure directory exists
        os.makedirs(os.path.dirname(video_full_path), exist_ok=True)

        video_writers[depth_key] = iio.get_writer(
            video_full_path,
            format='ffmpeg',
            mode='I',
            fps=frame_rate,
            codec='libx264',
            pixelformat='yuv420p'
        )
        video_relative_paths[depth_key] = video_save_relative_path

    return video_writers, video_relative_paths, image_keys, depth_keys

def process_step(step, idx, image_keys, depth_keys, video_writers, video_relative_paths,
                 current_position, current_orientation, task_instruction):
    """
    Process a single step, update video and build JSON data.

    Args:
        step: Current step data
        idx: Step index
        image_keys: List of image keys
        depth_keys: List of depth keys
        video_writers: Dictionary of video writers
        video_relative_paths: Dictionary of video relative paths
        current_position: Current position (will be updated)
        current_orientation: Current orientation (will be updated)
        task_instruction: Task instruction

    Returns:
        dict: Constructed JSON data
    """
    action_unnormalized = step["action_unnormalized"]

    # Process image data
    json_images = {}
    image_count = 0

    # Process image data
    for image_key in image_keys:
        if image_key in step['observation'] and step['observation'][image_key] is not None:
            # Decode image data
            if isinstance(step['observation'][image_key], tf.Tensor):
                if step['observation'][image_key].dtype == tf.string:
                    if tf.strings.length(step['observation'][image_key]) > 0:
                        image_data = tf.io.decode_image(step['observation'][image_key], expand_animations=False, dtype=tf.uint8)
                        image_array = image_data.numpy()
                else:
                    image_array = step['observation'][image_key].numpy()
            else:
                image_array = np.array(step['observation'][image_key])

            # Add to video
            video_writers[image_key].append_data(image_array)

            image_count += 1
            json_images[f"images_{image_count}"] = {
                "type": "video",
                "url": video_relative_paths[image_key],
                "frame_idx": idx
            }

    # Process depth data
    for depth_key in depth_keys:
        if depth_key in step['observation'] and step['observation'][depth_key] is not None:
            # Decode depth image data
            if isinstance(step['observation'][depth_key], tf.Tensor):
                if step['observation'][depth_key].dtype == tf.string:
                    if tf.strings.length(step['observation'][depth_key]) > 0:
                        depth_data = tf.io.decode_image(step['observation'][depth_key], expand_animations=False, dtype=tf.uint8)
                        depth_array = depth_data.numpy()
                else:
                    depth_array = step['observation'][depth_key].numpy()
            else:
                depth_array = np.array(step['observation'][depth_key])

            # Add to video
            video_writers[depth_key].append_data(depth_array)

            image_count += 1
            json_images[f"images_{image_count}"] = {
                "type": "video",
                "url": video_relative_paths[depth_key],
                "frame_idx": idx
            }

    # Ensure action data is native Python type
    if isinstance(action_unnormalized, tf.Tensor):
        action_last = float(action_unnormalized[-1].numpy())
    else:
        action_last = float(action_unnormalized[-1])

    # Build JSON data
    json_data = {
        **json_images,
        "prompt": task_instruction,
        "is_robot": True,
        "state": current_position + current_orientation + [action_last]
    }

    # Update position and orientation (accumulate actions)
    if len(action_unnormalized) >= 6:
        # Ensure action data is native Python type
        if isinstance(action_unnormalized, tf.Tensor):
            action_pos = action_unnormalized[0:3].numpy()
            action_ori = action_unnormalized[3:6].numpy()
        else:
            action_pos = action_unnormalized[0:3]
            action_ori = action_unnormalized[3:6]

        current_position[:] = [a + float(b) for a, b in zip(current_position, action_pos)]
        current_orientation[:] = [a + float(b) for a, b in zip(current_orientation, action_ori)]

    return json_data

def process_all_steps(episode, image_keys, depth_keys, video_writers, video_relative_paths, task_instruction):
    """
    Process all steps in the episode and generate JSON data list.

    Args:
        episode: Episode data
        image_keys: List of image keys
        depth_keys: List of depth keys
        video_writers: Dictionary of video writers
        video_relative_paths: Dictionary of video relative paths
        task_instruction: Task instruction

    Returns:
        list: List of JSON data
    """
    current_position = [0.0, 0.0, 0.0]  # Initialize position
    current_orientation = [0.0, 0.0, 0.0]  # Initialize orientation
    jsons_tmp = []

    # Get episode length, ensure action_unnormalized is not None
    if episode['action_unnormalized'] is None:
        print("Warning: action_unnormalized is None, skipping this episode")
        return []

    episode_length = len(episode['action_unnormalized'])

    for idx in range(episode_length):
        # Build current step data, filter out None observations
        step = {
            'observation': {key: value for key, value in
                           {key: episode['observation'][key][idx] for key in episode['observation'].keys()}.items()
                           if value is not None},
            'action_unnormalized': episode['action_unnormalized'][idx]
        }

        json_data = process_step(
            step, idx, image_keys, depth_keys, video_writers, video_relative_paths,
            current_position, current_orientation, task_instruction
        )
        jsons_tmp.append(json_data)

    return jsons_tmp

def save_json_file(json_data_list, json_full_path):
    """
    Save JSON data list to file.

    Args:
        json_data_list: List of JSON data to save
        json_full_path: Full path of the JSON file
    """
    os.makedirs(os.path.dirname(json_full_path), exist_ok=True)

    with open(json_full_path, 'w') as outfile:
        for json_obj in json_data_list:
            json_line = json.dumps(json_obj)
            outfile.write(json_line + '\n')

def load_rlds_dataset(
    dataset_name: str,
    split: str = 'train',
    data_dir: str = None,
    output_dir: str = "./output",
    frame_rate: int = 10,
    verbose: bool = False,
    **kwargs: Any
) -> int:
    """
    Load RLDS dataset and convert to DexData format with video and JSON outputs.

    This function loads RLDS datasets from TensorFlow Datasets, processes episodes,
    and converts them to DexData format with separate video files and JSON metadata.

    Args:
        dataset_name (str): Dataset name, e.g., 'libero_10_no_noops'
        split (str): Data split, options include 'train', 'validation', 'test', default is 'train'
        data_dir (str): Data storage directory, uses default if None
        output_dir (str): Root directory for output files (videos and JSONs), default is './output'
        frame_rate (int): Frame rate for output videos, default is 10
        verbose (bool): Enable verbose output, default is False
    """
    if dataset_name in DATASET_CONFIG:
        image_obs_keys = DATASET_CONFIG[dataset_name]["image_obs_keys"]
        depth_obs_keys = DATASET_CONFIG[dataset_name]["depth_obs_keys"]
    else:
        raise ValueError(f"Dataset {dataset_name} not found in configuration.")

    def restructure(traj):
        # extracts images, depth images and proprio from the "observation" dict
        traj_len = tf.shape(traj["action"])[0]
        old_obs = traj["observation"]
        new_obs = {}

        # Only process non-None image_obs_keys configurations
        for new, old in image_obs_keys.items():
            if old is not None:
                new_obs[f"image_{new}"] = old_obs[old]

        # Only process non-None depth_obs_keys configurations
        for new, old in depth_obs_keys.items():
            if old is not None:
                new_obs[f"depth_{new}"] = old_obs[old]

        # Add timestep info
        new_obs["timestep"] = tf.range(traj_len)

        # Extract language instruction into the "task" dict
        task = {}
        task["language_instruction"] = traj.pop("language_instruction")

        traj = {
            "observation": new_obs,
            "task": task,
            "action_unnormalized": tf.cast(traj["action"], tf.float32),
            "dataset_name": tf.repeat(dataset_name, traj_len),
        }

        return traj

    # Build loading parameters
    load_kwargs = {
        'data_dir': data_dir,
        **kwargs
    }
    # Load dataset builder
    builder = tfds.builder(dataset_name, **load_kwargs)

    # Load specified split dataset
    dataset = dl.DLataset.from_rlds(builder, split=split)

    dataset = dataset.traj_map(restructure, num_parallel_calls=10)

    video_output_path = os.path.join(output_dir, "videos")
    json_output_path = os.path.join(output_dir, "jsons")
    os.makedirs(video_output_path, exist_ok=True)
    os.makedirs(json_output_path, exist_ok=True)
    # Process episodes and convert to DexData format
    processed_episodes = 0

    for episode_index, episode in enumerate(dataset):
        try:
            # Get task instruction (take first one, as all steps have the same instruction)
            raw_task_instruction = episode['task']['language_instruction']
            if isinstance(raw_task_instruction, tf.Tensor):
                if raw_task_instruction.dtype == tf.string:
                    task_instruction = raw_task_instruction[0].numpy().decode('utf-8')
                else:
                    task_instruction = str(raw_task_instruction[0].numpy())
            else:
                task_instruction = str(raw_task_instruction[0])

            # Create video writers
            video_writers, video_relative_paths, image_keys, depth_keys = create_video_writers(
                episode, episode_index, dataset_name, video_output_path, frame_rate
            )

            total_video_streams = len(image_keys) + len(depth_keys)
            if verbose:
                print(f'Processing {dataset_name}:episode{episode_index}, contains {total_video_streams} video streams ({len(image_keys)} images + {len(depth_keys)} depth)')
            else:
                # Show progress for non-verbose mode
                if processed_episodes % 10 == 0:
                    print(f"Processed {processed_episodes} episodes...")

            # Process all steps
            jsons_tmp = process_all_steps(episode, image_keys, depth_keys, video_writers, video_relative_paths, task_instruction)

            # Close all video writers
            for writer in video_writers.values():
                writer.close()

            # Save JSON file
            json_save_relative_path = os.path.join(dataset_name, f"episode{str(episode_index)}.jsonl")
            json_full_path = os.path.join(json_output_path, json_save_relative_path)
            save_json_file(jsons_tmp, json_full_path)

            processed_episodes += 1
            if verbose:
                print(f"Processed {processed_episodes} episodes, current episode length: {len(jsons_tmp)}")

        except Exception as e:
            if verbose:
                print(f"Error occurred while processing episode {episode_index}: {str(e)}")
            continue

    # Print processing summary
    if verbose:
        print(f"\n=== Episode Processing Complete ===")
        print(f"Successfully processed {processed_episodes} episodes")
        print(f"Video files saved to: {video_output_path}")
        print(f"JSON files saved to: {json_output_path}")
    else:
        print(f"\nProcessing complete! Successfully processed {processed_episodes} episodes.")

    return processed_episodes

def parse_arguments():
    """
    Parse command line arguments for the RLDS to DexData converter.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Convert RLDS datasets to DexData format with video and JSON outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the RLDS dataset to convert (e.g., libero_10_no_noops)"
    )

    # Optional arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory where the dataset is stored (uses default if not specified)"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to process"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Root directory for output files (videos and JSONs)"
    )

    parser.add_argument(
        "--frame_rate",
        type=int,
        default=10,
        help="Frame rate for output videos"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output"
    )

    return parser.parse_args()


def main():
    """
    Main function to run the RLDS to DexData conversion.
    """
    args = parse_arguments()

    if args.verbose:
        print("Starting RLDS to DexData conversion with arguments:")
        print(f"  Dataset: {args.dataset_name}")
        print(f"  Data directory: {args.data_dir}")
        print(f"  Split: {args.split}")
        print(f"  Output directory: {args.output_dir}")
        print(f"  Frame rate: {args.frame_rate}")
        print()

    # Call the conversion function with parsed arguments
    processed_episodes = load_rlds_dataset(
        dataset_name=args.dataset_name,
        split=args.split,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        frame_rate=args.frame_rate,
        verbose=args.verbose
    )

    print(f"\nConversion completed! Processed {processed_episodes} episodes.")


# Usage example
if __name__ == "__main__":
    main()