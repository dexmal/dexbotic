# Data Usage Guide

This guide provides detailed instructions on how to use data for model training in the Dexbotic framework.

## Table of Contents

- [Using Provided Data](#using-provided-data)
- [Data Format](#data-format)
- [Using Custom Data](#using-custom-data)


## Using Provided Data

We provide simulation data that has already been processed and formatted for direct use.

| Dataset | Link  |
|---------|-------|
| Libero  | [🤗 Hugging Face](https://huggingface.co/datasets/Dexmal/libero) |
| CALVIN  | [🤗 Hugging Face](https://huggingface.co/datasets/Dexmal/calvin) |
| Simpler-Env  | [🤗 Hugging Face](https://huggingface.co/datasets/Dexmal/simpler) |
| RoboTwin 2.0 (CogACT) | [🤗 Hugging Face](https://huggingface.co/datasets/Dexmal/robotwin) |
| RoboTwin 2.0 (DM05 SFT) | [🤗 Hugging Face](https://huggingface.co/datasets/Dexmal/robotwin2-full) |
| ManiSkill2 | [🤗 Hugging Face](https://huggingface.co/datasets/Dexmal/maniskill2) |

Please organize the data according to the following directory structure:

```bash
[Your Code Path]
├── dexbotic
├── docs
├── data
│   ├── libero
│   │   ├── libero_10
│   │   │   ├── video
│   │   │   └── jsonl
│   │   ├── libero_goal
│   │   ├── libero_object
│   │   └── libero_spatial
│   ├── calvin
│   │   └── task_ABC_D
│   │       ├── video
│   │       └── jsonl
│   ├── robotwin
│   │   └── video
│   │   └── jsonl
│   ├── maniskill2
│   │   └── video
│   │   └── jsonl
│   └── simpler
│       ├── video
│       └── jsonl
└── ...

```

### RoboTwin2 Data Layouts

The `data/robotwin` layout above is used by the RoboTwin2 **CogACT** recipe:
annotations are under `data/robotwin/jsonl/<task>/`, and media paths use
`data/robotwin/video` as their prefix.

The **DM05 SFT** recipe uses the official
[Dexmal/robotwin2-full](https://huggingface.co/datasets/Dexmal/robotwin2-full)
archive, registered by
[`robotwin2_sft.py`](../dexbotic/data/data_source/robotwin2_sft.py). The archive
extracts as follows:

```text
data/robotwin2.0/
├── jsonl/
│   ├── adjust_bottle/
│   │   ├── clean/
│   │   └── randomized/
│   └── ...
└── video/
    ├── adjust_bottle/
    │   ├── clean/
    │   └── randomized/
    └── ...
```

JSONL files may also be nested below each setting directory. The default
dataset selection includes all 50 tasks in both settings (100 datasets), named
`robotwin2_example_<task>_<clean|randomized>`. Use `--dataset-name` to select a
subset; join multiple registered names with `+`.

See [DM05 RoboTwin2 training](DM05.md#robotwin2) for the state,
action-target, and normalization-statistics requirements.


## Data Format

We designed the Dexdata format to store robotic datasets in a unified and efficient way.

### Dataset Directory Structure

A Dexdata dataset is organized according to the following structure:

```bash
dataset_1
    index_cache.json   # Global index of dataset_1
    episode1.jsonl     # Data for the first episode
    episode2.jsonl     # Data for the second episode
    ...

```

- Each `.jsonl` file contains the data for a single robot episode.

- The `index_cache.json` file stores metadata for all episodes and is automatically generated for fast access.

> **Note:** Users do not need to manually manage the `index_cache.json` file—it is automatically created and maintained during dataset usage.

### Episode Data Format

Each line in a `.jsonl` file corresponds to one frame of robot data.

An example structure is shown below:

```json
{
    "images_1": {"type": "video", "url": "url1", "frame_idx": 21}, 
    "images_2": {"type": "video", "url": "url2", "frame_idx": 21},
    "images_3": {"type": "video", "url": "url3", "frame_idx": 21},
    "state": [0.1, 0.2],
    "prompt": "open the door",
    "is_robot": true,

    // Optional fields
    "answer": "answer text",
    "action": [0.12, 0.24],
    "conversations": [{"from": "human", "value": "<image>\nWhat are the colors of the box in the image?"},
                      {"from": "gpt", "value": "The box in the image is red."}]
}

```

> **Note:** Although the example above is formatted across multiple lines for readability, each `.jsonl` entry must be stored in a single line.

Field Specifications:

+ RGB data
    + Stored under keys like `images_*`.
    + Multiple views can be added (`images_1`, `images_2`, …). The usage order can be specified in the data configuration (DataConfig) `data_keys`, and you can also specify to use only a subset of the images.
    + We recommend using the Main View in `images_1`, the Left Hand View in `images_2`, and the Right Hand View in `images_3`.
    + Data can be video format, represented as `{"type": "video", "url": "video_url", "frame_idx": xx}`.
    + Data can also be image format, represented as `{"type": "image", "url": "image_url"}`.

+ Robot state

    + Stored under the `state` key.
    + Typically 7-dimensional: 3D position + 3D rotation + 1 gripper
    + By default, actions are constructed online using built-in dataset transforms.
    + Pre-processed actions can also be stored explicitly under the `action` key.
    + The RoboTwin2 DM05 SFT recipe requires finite 14-dimensional states and
      reconstructs absolute actions from future states, overwriting any stored
      `action`. See its [action-target contract](DM05.md#prepare-provided-data).

+ Text data
    + Prompts are stored in the `prompt` key.
    + Responses can be specified in two ways:
        + Directly: via the answer key.
        + Indirectly: leave answer empty, and Dexdata will use `ActionNormAnd2String` to convert actions into discretized textual responses.
    + **Multi-turn dialogue format (`conversations`)**: Use the `conversations` field for general multimodal dialogue data to support multi-turn conversations (LLaVA-compatible format). This is equivalent to `prompt`/`answer` but supports multiple dialogue turns.
        + If `conversations` exists, it is used directly; otherwise, `prompt` and `answer` are automatically converted to `conversations` format.
        + Format: A list of dialogue turns, each containing:
            + `{"from": "human", "value": "..."}` - user message (use `<image>` to reference images)
            + `{"from": "gpt", "value": "..."}` - assistant response
        + Single-turn example: `[{"from": "human", "value": "<image>\nWhat color is the box?"}, {"from": "gpt", "value": "The box is red."}]`
        + Multi-turn example: `[{"from": "human", "value": "<image>\nWhat's in the image?"}, {"from": "gpt", "value": "A red box."}, {"from": "human", "value": "What color?"}, {"from": "gpt", "value": "Red."}]`

+ Robot vs. general conversation data [**Important**]
    + The is_robot flag distinguishes robot data (true) from general data (false).

### Data Source Configuration

Data source configuration files are used to define metadata and configuration information for datasets. These files should be placed in the `data/data_source/` directory.

Example Configuration File: `dexbotic/data/data_source/libero_official.py`

```python
from dexbotic.data.data_source.register import register_dataset

# LIBERO dataset configuration
LIBERO_DATASET = {
    "goal": {
        "data_path_prefix": "./data/libero/libero_goal/video",
        "annotations": './data/libero/libero_goal',
        "frequency": 1,
    },
    "10": {
        "data_path_prefix": "./data/libero/libero_10/video",
        "annotations": './data/libero/libero_10',
        "frequency": 1,
    },
    "spatial": {
        "data_path_prefix": "./data/libero/libero_spatial/video",
        "annotations": './data/libero/libero_spatial',
        "frequency": 1,
    },
    "object": {
        "data_path_prefix": "./data/libero/libero_object/video",
        "annotations": './data/libero/libero_object',
        "frequency": 1,
    },
}

meta_data = {
    'non_delta_mask': [-1],  # Non-delta dim index, e.g. gripper
    'periodic_mask': None,  # Indices of periodic action dimensions (e.g., rotation), used for handling wrapping
    'periodic_range': None  # periodic range
}

# Register the dataset
register_dataset(LIBERO_DATASET, meta_data=meta_data, prefix='libero')
```

Configuration File Structure:

+ **Dataset Configuration**: The `LIBERO_DATASET` dictionary defines configuration information for multiple sub-datasets under a main dataset
    + Each key (e.g., `"goal"`, `"10"`, `"spatial"`, `"object"`) represents a sub-dataset within the main dataset
    + `data_path_prefix`: Specifies the storage path prefix for multimodal data (e.g. rgb images)
    + `annotations`: Specifies the full path to the annotation files for each sub-dataset
    + `frequency`: Data sampling frequency for each sub-dataset
+ **Dataset Registration**: Use the `register_dataset` function to register the entire dataset collection with a prefix. After registration, each sub-dataset can be accessed using names like `libero_goal`, `libero_10`, etc.

+ **Metadata Configuration**: The `meta_data` dictionary defines important properties for action processing and normalization:
    + `non_delta_mask`: Specifies the indices of action dimensions that should not be treated as delta values in delta computation. For example,  the gripper in the last dimension (index -1).
    + `periodic_mask`: Indices of action dimensions that have periodic properties (e.g., rotation angles). These dimensions require special handling for wrapping around their periodic range (e.g., 0° and 360° are equivalent).
    + `periodic_range`: The range value for periodic dimensions. For rotation angles, this is typically `2 * math.pi` (360 degrees in radians). When `periodic_mask` is `None`, this field is also `None`.


## Using Custom Data


### 1. Data Collection

Collect your robot data, ensuring it includes:
- Image data
- Robot state information (state field)
- Corresponding text instructions (prompt field)


### 2. Data Conversion

Convert your raw data to Dexdata format:

```python
import json
import os

def convert_to_dexdata_format(episode_data, output_dir):
    """
    Convert raw data to Dexdata format
    
    Args:
        episode_data: List containing episode data
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, episode in enumerate(episode_data):
        episode_file = os.path.join(output_dir, f"episode{i+1}.jsonl")
        
        with open(episode_file, 'w', encoding='utf-8') as f:
            for frame in episode:
                # Convert each frame of data
                dexdata_frame = {
                    "images_1": {
                        "type": "image", 
                        "url": frame['image_path']
                    },
                    "state": frame['robot_state'],
                    "prompt": frame['instruction'],
                    "is_robot": True
                }
                
                # Write to jsonl file (one JSON object per line)
                f.write(json.dumps(dexdata_frame, ensure_ascii=False) + '\n')
```


### 3. Use Data

#### 3.1 Create Data Source File

Create your data source file in the `dexbotic/data/data_source/` directory:

```python
# dexbotic/data/data_source/my_custom_dataset.py

from dexbotic.data.data_source.register import register_dataset
import math

# Define your dataset
MY_CUSTOM_DATASET = {
    "my_robot_data": {
        "data_path_prefix": "", # Image path prefix
        "annotations": '/path/to/your/custom_dataset/',  # Dataset path
        "frequency": 1,  # Data sampling frequency
    },
}

# Define metadata
meta_data = {
    'non_delta_mask': [6],  # Non-delta dim index, e.g. gripper
    'periodic_mask': [3, 4, 5],  # Indices of periodic action dimensions (e.g., rotation), used for handling wrapping
    'periodic_range': 2 * math.pi  # periodic range
}

# Register dataset
register_dataset(MY_CUSTOM_DATASET, meta_data=meta_data, prefix='my_custom')
```


#### 3.2 Use Data in Config

Set the `dataset_name` key in config file, for example

```python
@dataclass
class MyCustomDataConfig(CogACTDataConfig):
    """Data configuration"""
    dataset_name: str = field(default='my_custom_my_robot_data')  # Dataset name
    num_images: int = field(default=1)  # Number of images
    images_keys: list[str] = field(default_factory=lambda: ['images_1'])  # Image fields
```
