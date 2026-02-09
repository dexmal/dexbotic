# RL Post-Training for Dexbotic-π0 using RLinf

## Overview
We are pleased to announce a strategic collaboration with [RLinf](https://rlinf.readthedocs.io/en/latest/). This document describes how to apply RL post-training with Dexbotic-π0 using RLinf on LIBERO.

---

## Environment Setup

Set up the [RLinf](https://github.com/RLinf/RLinf) environment first.

```bash
git clone https://github.com/RLinf/RLinf.git
cd RLinf
bash requirements/install.sh embodied --venv dexbotic --model dexbotic --env maniskill_libero
source .venv/bin/activate
```

---

## Step 1. Apply Supervised Fine-tuning (SFT) with Dexbotic-π0
You can directly download our released checkpoints from [Hugging Face](https://huggingface.co/Dexmal/libero-db-pi0), which were trained on LIBERO 4 suites jointly, or fine-tune by yourself with the command below:

```bash
python playground/benchmarks/libero/libero_pi0.py --task train
```

---

## Step 2. Apply Post-Training with RLinf

Follow the RLinf training workflow and use the fine-tuned checkpoint from Step 1 in the RLinf config files.

### Configuration Files

* `libero_10_ppo_dexbotic_pi0.yaml`
* `libero_goal_ppo_dexbotic_pi0.yaml`
* `libero_spatial_ppo_dexbotic_pi0.yaml`
* `libero_object_ppo_dexbotic_pi0.yaml`

### Running the Training

Before running, set the checkpoint path in the chosen config to your converted checkpoint from Step 1, then launch training:

```bash
bash examples/embodiment/run_embodiment.sh CHOSEN_CONFIG
```

Replace `CHOSEN_CONFIG` with one of the four configs above.

---

## Step 3. Evaluate with RLinf

Evaluation follows the RLinf OpenPI evaluation guide:

[https://github.com/RLinf/RLinf/blob/main/toolkits/eval_scripts_openpi/README.md](https://github.com/RLinf/RLinf/blob/main/toolkits/eval_scripts_openpi/README.md)

Use your trained checkpoint and run the corresponding evaluation command from that guide.

---

## Results

| Model Setting | Libero-Spatial | Libero-Object | Libero-Goal | Libero-10 | Average |
| ------------- | -------------: | ------------: | ----------: | --------: | ------: |
| DB-π0 (SFT)   |           97.6 |          97.6 |        94.8 |      85.0 |    93.8 |
| + RLinf-PPO    |           99.2 |          99.8 |        97.2 |      95.6 |   97.95 |
| Δ Improvement |           +1.6 |          +2.2 |        +2.4 |     +10.6 |   +4.15 |

---

## References

- RLinf pi0 tutorial: https://rlinf.readthedocs.io/en/latest/rst_source/examples/pi0.html
- RLinf LIBERO tutorial: https://rlinf.readthedocs.io/en/latest/rst_source/examples/libero.html
- RLinf OpenPI evaluation scripts: https://github.com/RLinf/RLinf/blob/main/toolkits/eval_scripts_openpi/README.md
