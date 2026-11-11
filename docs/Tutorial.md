## Installation

### 🐳 Docker (Recommended)


We strongly recommend using the docker as a unified, consistent, and reproducible environment for training and deployment. This approach not only ensures reliability across workflows but also minimizes potential issues arising from CUDA version differences and Python dependency conflicts.

> Please see the [`Dockerfile`](../Dockerfile) for details about the image contents.

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

The image built from this repo's `Dockerfile` ships with `torch==2.6.0` and `transformers==4.57.6`.

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

pip install torch==2.6.0 torchvision==0.21.0 xformers --index-url https://download.pytorch.org/whl/cu118
cd dexbotic
pip install -e .
pip install transformers==4.57.6

# FlashAttention kernels (e.g. cross-entropy used in RL training) are fetched
# on demand from the Hugging Face Hub via the `kernels` library, which is
# installed as a core dependency above. No local flash-attn build is required.
#
# Optionally, to use a locally compiled flash-attn (e.g. for the
# `flash_attention_2` HF attention implementation), install it explicitly:
# pip install ninja packaging
# pip install flash-attn --no-build-isolation
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

Before starting training, please follow the instructions in [ModelZoo.md](ModelZoo.md) to set up the pre-trained models, and download the Libero dataset as described in [docs/Data.md](Data.md).

### Training a Model with Provided Data

We use Libero as an example to demonstrate how to train a model with Dexbotic.
The experiment configuration file for this example is located at: [`playground/benchmarks/libero/libero_cogact.py`](../playground/benchmarks/libero/libero_cogact.py)

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
> For FSDP2 support, see [FSDP2.md](FSDP2.md).

### Training a Model with Your Own Data

1. Prepare Your Own Data

Refer to  [docs/Data.md](Data.md) for detailed instructions on data preparation.
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

After training, please refer to the [Evaluation](ModelZoo.md#benchmark-results) section above to evaluate your model. Update the `model_name_or_path` in the inference config to your trained checkpoint, and run inference or start the inference server as described.