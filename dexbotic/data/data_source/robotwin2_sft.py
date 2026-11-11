"""RoboTwin2 DM05's 50 tasks x clean/randomized Dexdata registrations."""

from dexbotic.data.data_source.register import register_dataset
from dexbotic.data.data_source.robotwin2_official import ROBOTWIN2_DATASET

ROBOTWIN2_TASKS = tuple(ROBOTWIN2_DATASET)
ROBOTWIN2_SETTINGS = ("clean", "randomized")
ROBOTWIN2_META = {"robot_type": "Aloha RoboTwin2", "fps": 30}
ROBOTWIN2_ANNOTATIONS = "./data/robotwin2.0/jsonl"
ROBOTWIN2_MEDIA_PREFIX = "./data/robotwin2.0/video"


def register_robotwin2_sft(
    root=ROBOTWIN2_ANNOTATIONS,
    media_prefix=ROBOTWIN2_MEDIA_PREFIX,
):
    """Root layout: <root>/<task>/<clean|randomized>/**/*.jsonl."""
    register_dataset(
        {
            f"{task}_{setting}": {
                "annotations": f"{root.rstrip('/')}/{task}/{setting}",
                "data_path_prefix": media_prefix,
                "frequency": 1,
            }
            for task in ROBOTWIN2_TASKS
            for setting in ROBOTWIN2_SETTINGS
        },
        prefix="robotwin2_example",
        meta_data=ROBOTWIN2_META,
    )


register_robotwin2_sft()
