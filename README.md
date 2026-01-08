<!-- # Dexbotic -->

<p align="center">
  <img src="resources/logo.png" alt="logo" width="280"/>
</p>

## Introduction

Dexbotic aims to provide a one-stop VLA research service for professionals in embodied intelligence field. It offers a codebase that supports multiple mainstream VLA policies simultaneously, allowing users to reproduce various mainstream VLA methods with just a single environment setup based on the pretrained models we provide. Additionally, Dexbotic will continuously update to include more of the latest pre-trained foundation models and cutting-edge VLA models in the industry.

![intro](resources/intro.jpeg)

<details open>
<summary>Main features</summary>

+ **Unified Modular VLA Framework**

  Dexbotic centers around VLA models and is compatible with open-source interfaces of mainstream large language models. It integrates embodied manipulation and navigation, supporting multiple leading embodied manipulation and navigation policies, while also incorporating interfaces for future whole-body control.

+ **Powerful Pre-trained Foundation Models**
  
  For mainstream VLA policies such as Pi0 and CogACT, Dexbotic open-sources several more powerful pre-trained foundation models. These models bring significant performance improvements across various mainstream simulators (like SimplerEnv and CALVIN) as well as real-world robotic tasks.

+ **Experiment-Centric Development Framework**

  The experimental framework of Dexbotic adopts a "layered configuration + factory registration + entry dispatch" approach. Users can easily meet various needs such as modifying configurations, changing models, or adding tasks by simply altering the experimental Exp script. This design aligns with the Open-Closed Principle, allowing for flexibility and extensibility while maintaining stability.

+ **Cloud and Local Training Capabilities**
  
  Dexbotic fully addresses the training needs of users from different universities and enterprises. It supports large-scale cloud-based training platforms such as Alibaba Cloud and Volcano Engine. Additionally, it accommodates local training with consumer-grade GPUs, like RTX 4090 cards.

+ **Diverse Robot Support for Training and Deployment**
  For various mainstream robots, such as UR5, Franka and ALOHA, Dexbotic offers a unified data format for training. It also provides open-source, general-purpose deployment scripts, allowing users to customize their deployments. In the future, Dexbotic will continue to support additional mainstream robotic platforms.

</details>

## 🔥News!

+ [2026-1-8] Dexbotic adds [co-training](dexbotic/exp/hybrid_cogact_exp.py) capability, enabling joint training of action experts and LLMs for the CogACT model. Dexbotic now also provides a compatible image for Blackwell GPUs.
+ [2025-12-29] Dexbotic supports [OFT](playground/benchmarks/libero/libero_oft.py) and [Pi0.5](playground/benchmarks/libero/libero_pi05.py).
+ [2025-10-20] Dexbotic has been released. Checkout the [paper](docs/Dexbotic_Tech_Report.pdf) and [document](https://dexbotic.com/docs/) for details.

## Open-Source Plan


| Category            | Model/Policy             | Status |
|----------------------|-------------------------|--------|
| **Pretraining Model** | Dexbotic-Base          | ✔️     |
|                      | Dexbotic-CogACT         | ✔️     |
|                      | ├─ Dexbotic-CogACT-SArm | ✔️     |
|                      | └─ Dexbotic-CogACT-HArm | ✔️     |
|                      | Dexbotic-Pi0            | ✔️     |
|                      | Dexbotic-OFT            | ✖️     |
| **Manipulation Policy** | Pi0                  | ✔️     |
|                      | OFT                     | ✔️     |
|                      | CogACT                  | ✔️     |
|                      | MemoryVLA               | ✔️     |
|                      | Pi0.5                   | ✔️     |
| **Navigation Policy**  | MUVLA                 | ✔️     |
|                      | NaVid                   | ✖️     |
|                      | NaVILA                  | ✖️     |
|                      | StreamVLN               | ✖️     |



## Installation

### 🐳 Docker (Recommended)


We strongly recommend using the docker as a unified, consistent, and reproducible environment for training and deployment. This approach not only ensures reliability across workflows but also minimizes potential issues arising from CUDA version differences and Python dependency conflicts.

> Please see the [`Dockerfile`](Dockerfile) for details about the image contents.

0. Prerequisites

+ Ubuntu 20.04 or 22.04

+ NVIDIA GPU: RTX 4090 / RTX 5090 / A100 / H100 (8 GPUs recommended for training; 1 GPU for deployment)

+ NVIDIA Docker installed

1. Step 1: Clone the Repository

```bash
git clone https://github.com/Dexmal/dexbotic.git
```

2. Step 2: Start Docker

```bash
docker run -it --rm --gpus all --network host \
  -v /path/to/dexbotic:/dexbotic \
  dexmal/dexbotic \
  bash
```

3. Step 3: Activate Dexbotic Environment

```bash
cd /dexbotic
conda activate dexbotic
pip install -e .
```

<details>
<summary>Using on Blackwell GPUs</summary>

For users with Blackwell GPUs (e.g., B100, RTX 5090), please use the specialized Docker image `dexmal/dexbotic:c130t28`.

**Step 1: Start Docker with Blackwell Image**

```bash
docker run -it --rm --gpus all --network host \
  -v /path/to/dexbotic:/dexbotic \
  dexmal/dexbotic:c130t28 \
  bash
```

**Step 2: Activate Environment**

```bash
cd /dexbotic
pip install -e .
```

</details>

### Conda Installation

0. Prerequisites

+ Ubuntu 20.04 or 22.04

+ NVIDIA GPU: RTX 4090 / A100 / H100 (8 GPUs recommended for training; 1 GPU for deployment)

+ CUDA 11.8 (tested; other versions may also work)

+ Anaconda

1. Step 1: Clone the Repository

```bash
git clone https://github.com/Dexmal/dexbotic.git
```

2. Step 2: Install Dependencies

```bash
conda create -n dexbotic python=3.10 -y
conda activate dexbotic

pip install torch==2.2.2 torchvision==0.17.2 xformers --index-url https://download.pytorch.org/whl/cu118
cd dexbotic
pip install -e .
pip install transformers=4.51.0

# Install FlashAttention
pip install ninja packaging
pip install flash-attn --no-build-isolation
```


## Evaluation

We provide pre-trained models for both simulation benchmarks and real-robot settings.
Here we use the Libero pre-trained model as an example.

First, you should download the pre-trained models and put it in the `checkpoints` folder.

```bash
mkdir -p checkpoints/libero
cd checkpoints/libero
git clone https://huggingface.co/Dexmal/libero-db-cogact libero_cogact
```

We will demonstrate two ways to evaluate the model. The first is to directly infer one sample, which is the quick way to experience the model. The other is to first deploy the model server and then use a client to get the results, which is more practical in real-world deployment.

### Inference One Sample

```bash
CUDA_VISIBLE_DEVICES=0 python playground/benchmarks/libero/libero_cogact.py --task inference_single --image_path test_data/libero_test.png --prompt 'What action should the robot take to put both moka pots on the stove?'
```

You will expect the model to output a set of actions.

### Deploy Mode

1. Start Inference Server

```bash
CUDA_VISIBLE_DEVICES=0 python playground/benchmarks/libero/libero_cogact.py --task inference
```

2. Test Model Inference Results

```bash
curl -X POST \
  -F "text=What action should the robot take to put both moka pots on the stove?" \
  -F "image=@test_data/libero_test.png" \
  http://localhost:7891/process_frame

```

3. Test Libero Benchmark with Dexbotic-Benchmark

Set up the [dexbotic-benchmark](https://github.com/Dexmal/dexbotic-benchmark.git) following its instructions and test the deployed model in the LIBERO-GOAL environment.

```bash
cd dexbotic-benchmark
docker run --gpus all --network host -v $(pwd):/workspace \
  dexmal/dexbotic_benchmark \
  bash /workspace/scripts/env_sh/libero.sh /workspace/evaluation/configs/libero/example_libero.yaml
```

> dexbotic-benchmark also works without docker, see its documentation for further support

## Training

Before starting training, please follow the instructions in [ModelZoo.md](docs/ModelZoo.md) to set up the pre-trained models, and download the Libero dataset as described in [docs/Data.md](docs/Data.md).

### Training a Model with Provided Data

We use Libero as an example to demonstrate how to train a model with Dexbotic.
The experiment configuration file for this example is located at: [`playground/benchmarks/libero/libero_cogact.py`](playground/benchmarks/libero/libero_cogact.py)

1. Experiment Configuration

```python
# LiberoCogActTrainerConfig
output_dir = [Path to save checkpoints]

```

2. Launch Training

```bash
torchrun --nproc_per_node=8 playground/benchmarks/libero/libero_cogact.py
```
> We recommend using 8 × NVIDIA A100/H100 GPUs for training.
> If you are using 8 × RTX 4090, please use the configuration file
> `scripts/deepspeed/zero3_offload.json` to reduce GPU memory utilization.

### Training a Model with Your Own Data

1. Prepare Your Own Data

Refer to  [docs/Data.md](docs/Data.md) for detailed instructions on data preparation.
Once created, register your dataset under `dexbotic/data/data_source`.

2. Experiment Configuration

Create a new experiment configuration file (based on [`playground/example_exp.py`](playground/example_exp.py)) and set the required keys:

```python
# CogActTrainerConfig
output_dir = [Path to save checkpoints]

# CogActDataConfig
dataset_name = [Name of your registered dataset]

```

3. Launch Training

```bash
torchrun --nproc_per_node=8 playground/benchmarks/example_exp.py
```

After training, please refer to the [Evaluation](#evaluation) section above to evaluate your model. Update the `model_name_or_path` in the inference config to your trained checkpoint, and run inference or start the inference server as described.


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

# FAQ

1. Failed to install Flash-Attention: 

For detailed installation instructions and troubleshooting, please refer to the official documentation at https://github.com/Dao-AILab/flash-attention.

# Citaion

If you find this useful in your research, please consider citing:

```bibtex
@article{dexbotic,
  title={Dexbotic: Open-Source Vision-Language-Action Toolbox},
  author={Dexbotic Contributors},
  journal={arXiv preprint arXiv:2510.23511},
  year={2025}
}
```

# Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Dexmal/dexbotic&type=date&legend=top-left)](https://www.star-history.com/#Dexmal/dexbotic&type=date&legend=top-left)