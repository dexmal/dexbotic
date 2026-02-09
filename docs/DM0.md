# Dexbotic Model 0 (DM0) Tutorial

DM0 is a vision-language-action model built on a dual-expert architecture with merged attention and Flow Matching for continuous action generation. Unlike the CogACT/OFT models, DM0 generates action trajectories through a diffusion-based approach, producing a chunk of future actions in one forward pass.

![](../resources/dm0_arch.png)

> This tutorial follows the same workflow as the main [Tutorial](Tutorial.md) but focuses on DM0-specific configurations. Please ensure you have completed the [Installation](Tutorial.md#installation) steps before proceeding.

## Pretrained Model

| Model | Description | Input Images | Action Dim | Model Size | Link |
| - | - | - | - | - | - |
| DM0-base | DM0 base model with Flow Matching action generation | Up to 3 Views | 32D | 2.4B | [🤗 Hugging Face](https://huggingface.co/Dexmal/DM0-base) |

Download the pretrained DM0 model into the `checkpoints` folder:

```bash
mkdir -p checkpoints
cd checkpoints
git clone https://huggingface.co/Dexmal/DM0-base DM0-base
```

## Training

Before starting training, please follow the instructions in [ModelZoo.md](ModelZoo.md) to download the pretrained DM0 model, and download the Libero dataset as described in [Data.md](Data.md).

### Training a Model with Provided Data

We use Libero as an example to demonstrate how to train a DM0 model.
The experiment configuration file for this example is located at: [`playground/benchmarks/libero/libero_dm0.py`](../playground/benchmarks/libero/libero_dm0.py)

1. Launch Training

```bash
torchrun --nproc_per_node=8 playground/benchmarks/libero/libero_dm0.py
```

> We recommend using 8 × NVIDIA A100/H100 GPUs for training.
> If you are using 8 × RTX 4090, please use the configuration file
> `scripts/deepspeed/zero3_offload.json` to reduce GPU memory utilization.
> Normalization statistics are automatically computed before the first training run if not already cached.

### Training a Model with Your Own Data

1. Prepare Your Own Data

Refer to [Data.md](Data.md) for detailed instructions on data preparation.
Once created, register your dataset under `dexbotic/data/data_source`.

2. Experiment Configuration

Create a new experiment configuration file based on [`playground/benchmarks/libero/libero_dm0.py`](../playground/benchmarks/libero/libero_dm0.py) and customize the following:

```python
# DM0TrainerConfig
output_dir = [Path to save checkpoints]

# DM0DataConfig
dataset_name = [Name of your registered dataset]
num_images = [Number of camera views in your dataset]

# DM0InferenceConfig
model_name_or_path = [Path to your trained checkpoint]
action_dim = [Your action dimension]
non_delta_mask = [Indices of non-delta dimensions, e.g., gripper]
```

3. Launch Training

```bash
torchrun --nproc_per_node=8 path/to/your_dm0_exp.py
```

## Evaluation

We provide pre-trained models for the Libero simulation benchmark.
Here we use the Libero pre-trained DM0 model as an example.

First, you should download the pre-trained models and put it in the `checkpoints` folder.

```bash
mkdir -p checkpoints/libero
cd checkpoints/libero
git clone https://huggingface.co/Dexmal/DM0-libero DM0-libero
```

### Deploy Mode

1. Start Inference Server

```bash
CUDA_VISIBLE_DEVICES=0 python playground/benchmarks/libero/libero_dm0.py --task inference
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

### Real-Robot Evaluation with RoboChallenge

You can evaluate DM0 models on real robots through the [RoboChallenge](https://robochallenge.ai) platform using the [Dexbotic-RoboChallengeInference](https://github.com/dexmal/Dexbotic-RoboChallengeInference) framework.

1. **Installation**: Install this project (`dexbotic`) first, then clone and install the inference framework:

```bash
git clone https://github.com/dexmal/Dexbotic-RoboChallengeInference.git
cd Dexbotic-RoboChallengeInference
pip install -r requirements.txt
```

2. **Download Checkpoints**: Download task-specific DM0 checkpoints from the [DM0-table30-specialist](https://huggingface.co/collections/Dexmal/dm0-table30-specialist) collection:

```bash
huggingface-cli download Dexmal/DM0-table30_put_cup_on_coaster --local-dir ./checkpoints/DM0-table30_put_cup_on_coaster
```

3. **Submit Evaluation**: Log in to [RoboChallenge](https://robochallenge.ai), submit an evaluation request, and wait for task assignment.

4. **Run Inference**:

```bash
# Online mode (with robot, during assigned evaluation period)
python execute.py --config-name=specialist/put_cup_on_coaster user_id=YOUR_USER_ID
```

> For full details on configuration and advanced usage, see the [Dexbotic-RoboChallengeInference README](https://github.com/dexmal/Dexbotic-RoboChallengeInference).

After training, please refer to the [Evaluation](#evaluation) section above to evaluate your model. Update the `model_name_or_path` in the inference config to your trained checkpoint, and run inference or start the inference server as described.

## Benchmark Results
### Libero
| Model | Spatial | Object | Goal | Long | Average |
|-------|---------|--------|------|------|--------|
| DM0 | 98.2 | 98.8 | 96.6 | 82.6 | 94.1 |

### RoboChallenge

| # | Task Name | DM0 SR/Score | DM0_gen SR/Score | pi0 SR/Score | pi0.5 SR/Score |
|---|-----------|-------------|-----------------|-------------|---------------|
| 1 | sort_books | 20% / 44.50 | 0% / 8.50 | 0% / 24.50 | 0% / 60.00 |
| 2 | make_vegetarian_sandwich | 0% / 7.00 | 0% / 15.00 | 0% / 17.50 | 0% / 29.50 |
| 3 | fold_dishcloth | 20% / 44.00 | 10% / 10.50 | 0% / 32.00 | 20% / 24.00 |
| 4 | arrange_paper_cups | 30% / 73.00 | 10% / 54.00 | 0% / 41.50 | 0% / 48.00 |
| 5 | stack_color_blocks | 100% / 100.00 | 100% / 100.00 | 70% / 72.25 | 100% / 99.00 |
| 6 | pour_fries_into_plate | 40% / 51.00 | 0% / 6.00 | 40% / 56.00 | 30% / 38.00 |
| 7 | place_shoes_on_rack | 100% / 100.00 | 100% / 98.50 | 80% / 77.00 | 90% / 90.50 |
| 8 | put_opener_in_drawer | 30% / 28.00 | 10% / 10.00 | 50% / 71.50 | 80% / 77.50 |
| 9 | search_green_boxes | 100% / 98.50 | 100% / 95.50 | 70% / 74.00 | 80% / 80.00 |
| 10 | open_the_drawer | 100% / 98.00 | 90% / 95.00 | 0% / 50.00 | 40% / 60.50 |
| 11 | turn_on_faucet | 100% / 100.00 | 70% / 84.50 | 20% / 23.00 | 100% / 99.00 |
| 12 | turn_on_light_switch | 80% / 84.00 | 70% / 70.50 | 10% / 40.00 | 40% / 61.00 |
| 13 | scan_QR_code | 0% / 7.00 | 0% / 0.00 | 30% / 30.50 | 50% / 55.00 |
| 14 | put_cup_on_coaster | 100% / 97.50 | 100% / 100.00 | 60% / 71.00 | 90% / 96.00 |
| 15 | stack_bowls | 100% / 100.00 | 70% / 71.00 | 100% / 98.50 | 100% / 99.50 |
| 16 | wipe_the_table | 0% / 72.00 | 0% / 47.50 | 0% / 35.00 | 0% / 46.00 |
| 17 | hang_toothbrush_cup | 80% / 84.00 | 90% / 95.00 | 50% / 70.00 | 50% / 71.00 |
| 18 | press_three_buttons | 90% / 96.00 | 0% / 0.00 | 0% / 0.00 | 0% / 0.00 |
| 19 | plug_in_network_cable | 80% / 84.00 | 20% / 45.50 | 20% / 45.00 | 20% / 65.00 |
| 20 | arrange_flowers | 70% / 82.50 | 20% / 49.00 | 50% / 67.50 | 50% / 69.50 |
| 21 | stick_tape_to_box | 40% / 68.00 | 0% / 14.00 | 10% / 28.00 | 10% / 29.00 |
| 22 | move_objects_into_box | 0% / 20.50 | 0% / 12.00 | 0% / 33.50 | 10% / 58.50 |
| 23 | arrange_fruits_in_basket | 100% / 99.50 | 70% / 87.00 | 20% / 22.50 | 40% / 70.50 |
| 24 | clean_dining_table | 100% / 97.00 | 50% / 64.50 | 50% / 66.00 | 50% / 63.50 |
| 25 | sweep_the_rubbish | 80% / 82.00 | 30% / 40.00 | 10% / 27.00 | 20% / 46.00 |
| 26 | sort_electronic_products | 0% / 20.88 | 0% / 18.38 | 0% / 31.12 | 50% / 68.62 |
| 27 | shred_scrap_paper | 30% / 39.00 | 30% / 45.00 | 30% / 59.00 | 0% / 36.00 |
| 28 | put_pen_into_pencil_case | 90% / 96.00 | 20% / 40.00 | 70% / 88.00 | 80% / 89.50 |
| 29 | water_potted_plant | 80% / 94.00 | 0% / 33.50 | 0% / 6.00 | 0% / 36.50 |
| 30 | set_the_plates | 100% / 99.50 | 60% / 62.00 | 10% / 34.50 | 80% / 88.00 |
| | Average | 62% / 72.25 | 37% / 49.08 | 28% / 46.41 | 43% / 61.84 |

### ObjectNav

| Method         | HM3D SR ↑ | HM3D SPL ↑ | MP3D SR ↑ | MP3D SPL ↑ |
|----------------|-----------|------------|-----------|------------|
| VLFM           | 52.5      | 30.4       | 36.4      | 17.5       |
| L3MVN          | 54.2      | 25.5       | -         | -          |
| UniGoal        | 54.5      | 25.1       | 41.0      | 16.4       |
| OVRL           | 62.0      | 26.8       | 28.6      | 7.4        |
| PirlNav        | 70.4      | 34.1       | -         | -          |
| Uni-NaVid      | 73.7    | 37.1       | -         | -          |
| DM0        | 73.5  | 25.7   | 45.3  | 12.9   |