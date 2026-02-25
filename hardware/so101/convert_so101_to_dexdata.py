import os
import json
import shutil
import glob
import pandas as pd
import numpy as np 
from loguru import logger
from tqdm import tqdm
import click

try:
    import pyarrow.parquet as pq
except ImportError:
    logger.error("Missing pyarrow library. Please run: pip install pyarrow")
    exit(1)

def get_task_list(meta_dir): 
    parquet_path = os.path.join(meta_dir, "tasks.parquet")
    if os.path.exists(parquet_path):
        try:
            df = pd.read_parquet(parquet_path)
            # First, check standard columns
            for col in ["task", "instruction", "language_instruction", "desc", "description"]:
                if col in df.columns:
                    return df[col].astype(str).tolist()
            
            # Then check other string/object columns
            for col in df.columns:
                if df[col].dtype == object or df[col].dtype == str:
                    return df[col].astype(str).tolist()
            
            # If no suitable column found, check if index contains task strings
            # (Some LeRobot datasets store prompts in the index)
            if df.index.dtype == object or (len(df) > 0 and isinstance(df.index[0], str)):
                task_list = df.index.astype(str).tolist()
                # Filter out numeric indices, keep only string prompts
                if task_list and not all(t.replace('.', '').replace('-', '').isdigit() for t in task_list):
                    return task_list

            # Last resort: return first column
            return df.iloc[:, 0].astype(str).tolist()
        except Exception as e:
            logger.error(f"Failed to read parquet: {e}")
            return []

    jsonl_path = os.path.join(meta_dir, "tasks.jsonl")
    tasks = []
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r") as f:
            for line in f:
                try:
                    info = json.loads(line)
                    tasks.append(info.get("task", info.get("instruction", "")))
                except Exception:
                    continue
    return tasks

def get_latest_episode_idx(output_dir): 
    if not os.path.exists(output_dir):
        return 0
    return len(glob.glob(os.path.join(output_dir, "*.jsonl")))

def parse_one_episode(df: pd.DataFrame, task_list, camera_map): 
    data_list = []
    cols = df.columns
    
    sorted_cams = sorted(camera_map.items()) 

    for row_id, row in df.iterrows(): 
        try: 
            if "observation.state" in cols:
                state_raw = row["observation.state"]
                state = state_raw if isinstance(state_raw, np.ndarray) else np.array(state_raw)
            elif "observation.state.left_arm" in cols:
                left_arm = np.array(row["observation.state.left_arm"])
                state = np.concatenate([left_arm, np.zeros(10)]) 
            else:
                state = np.zeros(16)

            action = np.array(row["action"]) if "action" in cols else np.zeros(6)
            
            timestamp = row.get("timestamp", 0.0)
            frame_index = row.get("frame_index", 0)
            episode_index = row.get("episode_index", 0)
            
            task_index = int(row.get("task_index", 0))
            if task_list and task_index < len(task_list):
                prompt = str(task_list[task_index])
            else:
                prompt = "unknown task"

            if "@" in prompt: subtask = prompt.split("@")[1]
            else: subtask = prompt

            entry = {
                "prompt": prompt,
                "state": state.tolist(),
                "action": action.tolist(),
                "is_robot": True,
                "extra": {
                    "subtask": subtask,
                    "timestamp": timestamp,
                    "episode_index": episode_index,
                }
            }

            for idx, (cam_name, url) in enumerate(sorted_cams):
                key = f"images_{idx + 1}"
                entry[key] = {
                    "type": "video",
                    "url": url,
                    "frame_idx": frame_index,
                    "_camera_name": cam_name 
                }

            data_list.append(entry)

        except Exception as e: 
            logger.error(f"Error parsing row {row_id}: {e}")
            break
    return data_list

def save_jsonl(data_list, jsonl_path):
    with open(jsonl_path, "w") as f:
        for data in data_list:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

@click.command()
@click.option("-i", "--lerobot_dir", type=str, required=True, help="Input directory")
@click.option("-o", "--output_dir", type=str, required=True, help="Output directory")
def main(lerobot_dir, output_dir): 
    TARGET_SPLITS = ["train", "test", "val"] 

    all_items = os.listdir(lerobot_dir)
    task_dirs = [d for d in all_items if os.path.isdir(os.path.join(lerobot_dir, d))]

    for task_name in task_dirs: 
        logger.info(f"=== Processing Task: {task_name} ===")
        
        for SPLIT in TARGET_SPLITS:
            task_split_dir = os.path.join(lerobot_dir, task_name, SPLIT)
            if not os.path.isdir(task_split_dir): continue 
            
            logger.info(f"--- Processing Split: {SPLIT} ---")

            meta_dir = os.path.join(task_split_dir, "meta")
            if not os.path.exists(meta_dir): meta_dir = os.path.join(lerobot_dir, task_name, "meta")
            task_list = get_task_list(meta_dir)

            if not task_list or (len(task_list) > 0 and task_list[0].isdigit()):
                # Fallback: use task name as prompt if no prompt found in dataset
                fixed_prompt = task_name.replace("_", " ")
                task_list = [fixed_prompt] * 1000 
                logger.warning(f"No prompt found in dataset, using task name as fallback: '{fixed_prompt}'")

            data_root = os.path.join(task_split_dir, "data")
            video_base = os.path.join(task_split_dir, "videos")
            
            if not os.path.exists(data_root): 
                logger.warning(f"Data root not found: {data_root}")
                continue
            
            if os.path.exists(video_base):
                all_video_dirs = [d for d in os.listdir(video_base) if os.path.isdir(os.path.join(video_base, d))]
                camera_folders = sorted([d for d in all_video_dirs if "images" in d])
                
                if not camera_folders:
                    camera_folders = sorted(all_video_dirs)
                
                logger.info(f"Discovered cameras: {camera_folders}")
            else:
                logger.warning(f"Video base not found: {video_base}")
                camera_folders = []

            output_jsonl_dir = os.path.join(output_dir, "jsonl", task_name)
            output_video_dir = os.path.join(output_dir, "videos", task_name)
            os.makedirs(output_jsonl_dir, exist_ok=True)
            os.makedirs(output_video_dir, exist_ok=True)

            for chunk_name in os.listdir(data_root): 
                chunk_path = os.path.join(data_root, chunk_name)
                if not os.path.isdir(chunk_path): continue

                parquet_files = sorted([f for f in os.listdir(chunk_path) if f.endswith(".parquet")])
                
                for episode_file in tqdm(parquet_files, desc=f"{task_name}/{SPLIT}/{chunk_name}"):
                    try:
                        df = pq.read_table(os.path.join(chunk_path, episode_file)).to_pandas()
                    except Exception as e:
                        logger.error(f"Bad parquet {episode_file}: {e}")
                        continue

                    eid = get_latest_episode_idx(output_jsonl_dir)
                    
                    current_episode_cam_map = {} 
                    
                    for cam_folder in camera_folders:
                        clean_cam_name = cam_folder.split(".")[-1]
                        out_vid_name = f"episode_{eid:05d}_{clean_cam_name}.mp4"
                        current_episode_cam_map[clean_cam_name] = os.path.join(task_name, out_vid_name)

                    data = parse_one_episode(df, task_list, current_episode_cam_map)
                    if not data: continue
                    
                    save_jsonl(data, os.path.join(output_jsonl_dir, f"episode_{eid:05d}.jsonl"))

                    src_vid_name = episode_file.replace(".parquet", ".mp4") 
                    
                    for cam_folder in camera_folders:
                        clean_cam_name = cam_folder.split(".")[-1]
                        
                        src_vid_path = os.path.join(video_base, cam_folder, chunk_name, src_vid_name)
                        
                        target_name = os.path.basename(current_episode_cam_map[clean_cam_name])
                        dst_vid_path = os.path.join(output_video_dir, target_name)
                        
                        if os.path.exists(src_vid_path):
                            shutil.copy2(src_vid_path, dst_vid_path)

if __name__ == "__main__":
    main()