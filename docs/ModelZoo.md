## General Pretrained Models

| Model                 | Description                                                                 | Input Images                                  | Action Dim | Model Size | Link |
| -                     | -                                                                           | -                                             |-           | -          | - |
| Dexbotic-Base         | Discrete vision-language action model (similar to OpenVLA)                  | Single View                                   | NA         | 7B         |[🤗 Hugging Face](https://huggingface.co/Dexmal/Dexbotic-Base) |
| Dexbotic-CogACT-SArm       | Single-arm CogACT model                                                     | Single View                                   | 7D         | 7B         |[🤗 Hugging Face](https://huggingface.co/Dexmal/Dexbotic-CogACT-SArm) |
| Dexbotic-CogACT-HArm  | Dual-arm CogACT model with multiple views input                             | Main View + Left Hand-View + Right Hand-View  | 16D        | 7B         |[🤗 Hugging Face](https://huggingface.co/Dexmal/Dexbotic-CogACT-HArm) |
| DM0-base | DM0 base model with Flow Matching action generation | Up to 3 Views | 32D | 2.4B | [🤗 Hugging Face](https://huggingface.co/Dexmal/DM0-base) |

It is recommended to download the pretrained models into the following folders.

```bash
mkdir checkpoints
cd checkpoints
git clone https://huggingface.co/Dexmal/Dexbotic-Base Dexbotic-Base
git clone https://huggingface.co/Dexmal/Dexbotic-CogACT-SArm Dexbotic-CogACT-SArm
git clone https://huggingface.co/Dexmal/Dexbotic-CogACT-HArm Dexbotic-CogACT-HArm
git clone https://huggingface.co/Dexmal/DM0-base DM0-base
```

## Action Dimension Description

Users need to map their data to the action dimensions of the pretrained models. If the data dimension is smaller than the pretrained model dimension, padding will be conducted automatically.

We recommend using the following data formats to fully utilize the pretrained models:

1. **Single-arm end-effector pose**: Organize 7D action data as `[xyz + rpy + gripper]`
2. **Single-arm joint angles**: Organize 8D action data as `[joints + gripper]`
3. **Dual-arm end-effector pose**: Organize 14D action data as `[left_arm_xyz + left_arm_rpy + left_arm_gripper + right_arm_xyz + right_arm_rpy + right_arm_gripper]`
4. **Dual-arm joint angles**: Organize 16D action data as `[left_arm_joints + left_arm_gripper + right_arm_joints + right_arm_gripper]`

## Other Dexbotic Models

| Model | Link |
| -     | -    |
| Dexbotic-π0 | [🤗 HF](https://huggingface.co/Dexmal/Dexbotic-PI0) |
| Dexbotic-π05 | [🤗 HF](https://huggingface.co/Dexmal/Dexbotic-PI05) |
| Dexbotic-NaVILA | [🤗 HF](https://huggingface.co/Dexmal/Dexbotic-NaVILA) |
| Dexbotic-RL-Base | [🤗 HF](https://huggingface.co/Dexmal/Dexbotic-RL-Base) |


## Benchmark Results

### Libero

| Model     | Libero-Spatial | Libero-Object | Libero-Goal | Libero-10 | Average | Config | Checkpoint  Link |
| -         | -              | -             | -           | -         | -       | -      | -                |
| CogACT    | 97.2 | 98.0 | 90.2 | 88.8 | 93.6 | - | - |
| DB-CogACT | 93.8 | 97.8 | 96.2 | 91.8 | 94.9 | [libero_cogact.py](playground/benchmarks/libero/libero_cogact.py) | [🤗 HF](https://huggingface.co/Dexmal/libero-db-cogact) |
| π0 | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 | - | - |
| DB-π0 | 97 | 98.2 | 94 | 86.4 | 93.9 | [libero_pi0.py](playground/benchmarks/libero/libero_pi0.py) | [🤗 HF](https://huggingface.co/Dexmal/libero-db-pi0) |
| MemVLA | 98.4 | 98.4 | 96.4 | 93.4 |96.7 | - |
| DB-MemVLA | 97.2 | 99.2 | 98.4 | 93.2 | 97.0 | [libero_memvla.py](https://github.com/Dexmal/dexbotic/blob/main/playground/benchmarks/libero/libero_memvla.py) | [🤗 HF](https://huggingface.co/Dexmal/libero-db-memvla) | [🤗 HF](https://huggingface.co/Dexmal/libero-db-memvla) |

### CALVIN

> Our training and evaluation are conducted under the ABC->D setting.

| Model | 1 | 2 | 3 | 4 | 5 | Average Length | Config | Checkpoint  Link |
| -         | -      | - | -             | -           | -         | -       | -      | -                |
| CogACT | 83.8 | 72.9 | 64 | 55.9 | 48 | 3.246 | - | - |
| DB-CogACT | 93.5 | 86.7 | 80.3 | 76 | 69.8 | 4.063 | [calvin_cogact.py](playground/benchmarks/calvin/calvin_cogact.py) | [🤗 HF](https://huggingface.co/Dexmal/calvin-db-cogact) |
| OFT | 89.1 | 79.4 | 67.4 | 59.8 | 51.5 | 3.472 | - | - |
| DB-OFT | 92.8 | 80.7 | 69.2 | 60.2 | 51.1 | 3.540 | [calvin_oft.py](playground/benchmarks/calvin/calvin_oft.py) |  [🤗 HF](https://huggingface.co/Dexmal/calvin-db-oft) |

### SimplerEnv

> Our training uses the Bridge dataset and is tested on the WidowX environment.

| Model | Put Spoon on Towel | Put Carrot on Plate | Stack Green Block on Yellow Block |Put Eggplant in Yellow Basket | Average | Config | Checkpoint  Link |
| -         | -              | -             | -           | -         | -       | -      | -                |
| CogACT    | 71.7 | 50.8 | 15 |67.5 | 51.25 | - | - |
| DB-CogACT | 87.5 | 65.28 | 29.17 | 95.83 | 69.45 | [simpler_cogact.py](playground/benchmarks/simpler/simpler_cogact.py) | [🤗 HF](https://huggingface.co/Dexmal/simpler-db-cogact) |
| OFT | 12.5 | 4.2 | 4.2 | 100 | 30.23 | - | - |
| DB-OFT | 91.67 | 76.39 | 43.06 | 94.44 | 76.39 | [simpler_oft.py](playground/benchmarks/simpler/simpler_oft.py) | [🤗 HF](https://huggingface.co/Dexmal/simpler-db-oft) |
| MemVLA | 75.0 | 75.0 | 37.5 | 100.0 | 71.9 | - | - |
| DB-MemVLA | 100.0 | 66.7 | 70.8 | 100.0 | 84.4 | [simpler_memvla.py](playground/benchmarks/simpler/simpler_memvla.py) | [🤗 HF](https://huggingface.co/Dexmal/simpler-db-memvla) |

### ManiSkill2

| Model | PickCube | StackCube | PickSingleYCB | PickSingleEGAD | PickClutterYCB | Average | Config | Checkpoint  Link |
| -         | -              | -             | -           | -         | -       | -      | -      | -                |
| CogACT    | 55 | 70 | 30 | 25 | 20 | 40 | - | - |
| DB-CogACT | 90 | 65 | 65 | 40 | 30 | 58 | [maniskill2_cogact.py](playground/benchmarks/maniskill2/maniskill2_cogact.py) | [🤗 HF](https://huggingface.co/Dexmal/maniskill2-db-cogact) |
| OFT | 40 | 45 | 5 | 5 | 0 | 21 | - | - |
| DB-OFT | 90 | 75 | 55 | 65 | 30 | 63 | [maniskill2_oft.py](playground/benchmarks/maniskill2/maniskill2_oft.py) | [🤗 HF](https://huggingface.co/Dexmal/maniskill2-db-oft) |
| π0 | 95 | 85 | 55 | 85 | 10 | 66 | - | - |
| DB-π0 | 95 | 85 | 65 | 50 | 30 | 65 | [maniskill2_pi0.py](playground/benchmarks/maniskill2/maniskill2_pi0.py) | [🤗 HF](https://huggingface.co/Dexmal/maniskill2-db-pi0) |

### RoboTwin2.0

> Our training uses the RoboTwin2.0 demo_clean dataset and is tested on the Aloha-AgileX demo_clean environment.

| Model | Adjust Bottle | Grab Roller | Place Empty Cup |Place Phone Stand | Average | Config | Checkpoint  Link |
| -         | -              | -             | -           | -         | -       | -      | -                |
| CogACT   | 87 | 72 | 11 |5 | 43.8 | - | - |
| DB-CogACT | 99 | 89 | 28 | 18 | 58.5 | [robotwin2_cogact.py](playground/benchmarks/robotwin2/robotwin2_cogact.py) | [🤗 HF](https://huggingface.co/Dexmal/robotwin-db-cogact) |