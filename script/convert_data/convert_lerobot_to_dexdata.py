"""
Example script for converting LeRobot dataset format to DexData format.

This script demonstrates how to convert LeRobot datasets (specifically galaxea_open_world_dataset)
to the DexData format.
"""

import os
import json

import pyarrow.parquet as pq
import pandas as pd
import numpy as np 
from loguru import logger
from tqdm import tqdm
import click


def get_task_list(task_root): 
    """
    Load task list from tasks.jsonl file in the given task root directory.
    Args:
        task_root: Path to the task root directory
    Returns:
        list: List of task strings
    """
    task_file = os.path.join(task_root, "meta", "tasks.jsonl")
    assert os.path.isfile(task_file), f"task file not found: {task_file}"
    tasks = []
    with open(task_file, "r") as f:
        for i, line in enumerate(f): 
            info = json.loads(line)
            task_index = info["task_index"]
            sub_task = info["task"]
            assert task_index == i, f"task index mismatch: {task_index} vs {i}"
            tasks.append(sub_task)
    return tasks


def get_latest_episode_idx(task_dir): 
    """
    Get the latest episode index in the given task directory
    by counting the number of existing episode files.
    Args:
        task_dir: Path to the task directory
    Returns:
        int: Latest episode index
    """
    fnames = os.listdir(task_dir)
    return len(fnames)


def parse_one_episode(
    df: pd.DataFrame,
    task_list,
    head_video_rel_path,
    left_wrist_video_rel_path,
    right_wrist_video_rel_path,
): 
    """
    Parse one episode dataframe into a list of data dictionaries.
    Args:
        df: DataFrame of the episode
        task_list: List of task strings
        head_video_rel_path: Relative path to the head video
        left_wrist_video_rel_path: Relative path to the left wrist video
        right_wrist_video_rel_path: Relative path to the right wrist video
    Returns:
        list: List of data dictionaries for the episode
    """
    data_list = []
    for row_id, row in df.iterrows(): 
        try: 
            # NOTE: parse all data in one row
            left_arm = np.array(row["observation.state.left_arm"])  # (6,) 6dof
            left_arm_vel = np.array(row["observation.state.left_arm.velocities"])  # (6,) 6dof
            right_arm = np.array(row["observation.state.right_arm"])  # (6,) 6dof
            right_arm_vel = np.array(row["observation.state.right_arm.velocities"])  # (6,) 6dof
            chassis = np.array(row["observation.state.chassis"])  # (10,) imu: quat(4) + rot_vel(3) + lin_acc(3)
            torso = np.array(row["observation.state.torso"])  # (4,) 4dof(pad 0 in last)
            torso_vel = np.array(row["observation.state.torso.velocities"])  # (4,) 4dof(pad 0 in last)
            left_gripper = row["observation.state.left_gripper"]  # () mm
            right_gripper = row["observation.state.right_gripper"]  # () mm
            left_ee_pose = np.array(row["observation.state.left_ee_pose"])  # (7,) ee: pos(3) + quat(4)
            right_ee_pose = np.array(row["observation.state.right_ee_pose"])  # (7,) ee: pos(3) + quat(4)

            action_left_gripper = row["action.left_gripper"]  # () 0~100
            action_right_gripper = row["action.right_gripper"]  # () 0~100
            action_chassis_vel = np.array(row["action.chassis.velocities"])  # (3,) lin_x, lin_y, ang_z
            action_torso_vel = np.array(row["action.torso.velocities"])  # (6,) torso velocities: aug(3) + lin(3) 
            action_left_arm = np.array(row["action.left_arm"])  # (6,) 6dof
            action_right_arm = np.array(row["action.right_arm"])  # (6,) 6dof

            timestamp = row["timestamp"]
            frame_index = row["frame_index"]
            episode_index = row["episode_index"]
            index = row["index"]
            coarse_task_index = row["coarse_task_index"]
            task_index = row["task_index"]
            coarse_quality_index = row["coarse_quality_index"]
            quality_index = row["quality_index"]
        
        except: 
            logger.error(f"Error parsing row {row_id}, skip this episode")
            data_list = None
            break

        state = np.concatenate(
            [
                left_arm,
                left_arm_vel,
                right_arm,
                right_arm_vel,
                chassis,
                torso,
                torso_vel,
                np.array([left_gripper]),
                np.array([right_gripper]),
                left_ee_pose,
                right_ee_pose,
            ]
        )
        action = np.concatenate(
            [
                np.array([action_left_gripper]),
                np.array([action_right_gripper]),
                action_chassis_vel,
                action_torso_vel,
                action_left_arm,
                action_right_arm,
            ]
        )

        prompt = task_list[coarse_task_index]
        # NOTE: sometimes, the task is None
        if task_list[task_index] is None: 
            print(f"Invalid task format: {task_list[task_index]}")
            data_list = None
            break
        # NOTE: subtask schema is Chinese@English
        if len(task_list[task_index].split("@")) != 2: 
            print(f"Invalid task format: {task_list[task_index]}")
            data_list = None
            break
        subtask = task_list[task_index].split("@")[1]
        data_list.append(
            {
                "images_1": {
                    "type": "video",
                    "url": head_video_rel_path,
                    "frame_idx": frame_index,
                },
                "images_2": {
                    "type": "video",
                    "url": left_wrist_video_rel_path,
                    "frame_idx": frame_index,
                },
                "images_3": {
                    "type": "video",
                    "url": right_wrist_video_rel_path,
                    "frame_idx": frame_index,
                },
                "prompt": prompt,
                "state": state.tolist(),
                "action": action.tolist(),
                "is_robot": True,
                "extra": {
                    "subtask": subtask,
                    "timestamp": timestamp,
                    "episode_index": episode_index,
                    "index": index,
                    "coarse_quality_index": coarse_quality_index,
                    "quality_index": quality_index,
                }
            }
        )

    return data_list


def save_jsonl(data_list, jsonl_path):
    """
    Save data list to a jsonl file.
    Args:
        data_list: List of data dictionaries
        jsonl_path: Path to the output jsonl file
    Returns:
        None
    """
    with open(jsonl_path, "w") as f:
        for data in data_list:
            line = json.dumps(data, ensure_ascii=False)
            f.write(line + "\n")


@click.command()
@click.option("-i", "--lerobot_dir", type=str, required=True, help="Path to the lerobot dataset root directory.")
@click.option("-o", "--output_dir", type=str, required=True, help="Path to the output dexdata directory.")
def main(lerobot_dir, output_dir): 
    """
    Convert lerobot dataset to dexdata format.
    Args:
        lerobot_dir: Path to the lerobot dataset root directory.
        output_dir: Path to the output dexdata directory.
    Returns:
        None
    """
    for task_name in os.listdir(lerobot_dir): 
        task_root = os.path.join(lerobot_dir, task_name)
        if not os.path.isdir(task_root):
            continue 
        output_jsonl_dir = os.path.join(output_dir, "jsonl", task_name)
        output_video_dir = os.path.join(output_dir, "videos", task_name)
        os.makedirs(output_jsonl_dir, exist_ok=True)
        os.makedirs(output_video_dir, exist_ok=True)
        task_list = get_task_list(task_root)
        
        task_data_dir = os.path.join(task_root, "data")
        for chunk_name in os.listdir(task_data_dir): 
            chunk_path = os.path.join(task_data_dir, chunk_name)
            head_video_dir = os.path.join(task_root, "videos", chunk_name, "observation.images.head_rgb")
            left_wrist_video_dir = os.path.join(task_root, "videos", chunk_name, "observation.images.left_wrist_rgb")
            right_wrist_video_dir = os.path.join(task_root, "videos", chunk_name, "observation.images.right_wrist_rgb")

            for episode_name in tqdm(os.listdir(chunk_path), desc=f"Processing {task_name}/{chunk_name}"): 
                episode_path = os.path.join(chunk_path, episode_name)
                if not episode_name.endswith(".parquet"):
                    continue 
                df = pq.read_table(episode_path).to_pandas()

                episode_id = get_latest_episode_idx(output_jsonl_dir)
                head_video_rel_path = os.path.join(task_name, f"episode_{episode_id:05d}_head.mp4")
                left_wrist_video_rel_path = os.path.join(task_name, f"episode_{episode_id:05d}_left_wrist.mp4")
                right_wrist_video_rel_path = os.path.join(task_name, f"episode_{episode_id:05d}_right_wrist.mp4")
                episode_data_list = parse_one_episode(
                    df,
                    task_list,
                    head_video_rel_path,
                    left_wrist_video_rel_path,
                    right_wrist_video_rel_path,
                )
                if episode_data_list is None: 
                    continue 
                episode_jsonl_path = os.path.join(output_jsonl_dir, f"episode_{episode_id:05d}.jsonl")
                save_jsonl(episode_data_list, episode_jsonl_path)

                # NOTE: Copy video files to output directory
                video_name = episode_name.replace(".parquet", ".mp4")
                head_video_path = os.path.join(head_video_dir, video_name)
                left_wrist_video_path = os.path.join(left_wrist_video_dir, video_name)
                right_wrist_video_path = os.path.join(right_wrist_video_dir, video_name)
                output_head_video_path = os.path.join(output_video_dir, f"episode_{episode_id:05d}_head.mp4")
                cmd = f"cp {head_video_path} {output_head_video_path}"
                os.system(cmd)
                output_left_wrist_video_path = os.path.join(output_video_dir, f"episode_{episode_id:05d}_left_wrist.mp4")
                cmd = f"cp {left_wrist_video_path} {output_left_wrist_video_path}"
                os.system(cmd)
                output_right_wrist_video_path = os.path.join(output_video_dir, f"episode_{episode_id:05d}_right_wrist.mp4")
                cmd = f"cp {right_wrist_video_path} {output_right_wrist_video_path}"
                os.system(cmd)
        

if __name__ == "__main__":
    main()