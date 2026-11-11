# FSDP2 Training Backend

Dexbotic supports training `pi0`, `pi05`, `dm0`, and `cogact` style models with the PyTorch FSDP2 backend. Compared with the default DeepSpeed training stack, FSDP2 is lighter, supports reliable checkpoint resume, and provides better training throughput.

In our training setup, FSDP2 provides more than 20% speedup over DeepSpeed. After switching to FSDP2, LIBERO simulation evaluation remains at the same level.

## Requirements

Use the following dependency range for FSDP2 training:

```toml
torch>=2.6.0
torchvision>=0.21.0
transformers>=4.57.6
accelerate>=1.10.0
pydantic==2.10.6
```

After pulling the repo, `pip install -e .` installs these pins. The default `Dockerfile` also pre-installs `torch==2.6.0` and `transformers==4.57.6` in the `dexbotic` conda environment.

`transformers<5.0.0` uses the explicit Accelerate FSDP2 plugin path. `transformers>=5.0.0` is supported and will use the native Trainer FSDP2 path when available.

## Performance

FSDP2 improves training speed by more than 20% compared with DeepSpeed in our tested configuration. On LIBERO, simulation evaluation remains stable after switching the training backend to FSDP2:

| Model | Libero-Spatial | Libero-Object | Libero-Goal | Libero-10 | Average |
| --- | ---: | ---: | ---: | ---: | ---: |
| DB-CogACT | 93.8 | 97.8 | 96.2 | 91.8 | 94.9 |
| DB-CogACT-FSDP2 | 93.2 | 97.8 | 97.2 | 91.2 | 94.9 |
| DB-PI0 | 97.0 | 98.2 | 94.0 | 86.4 | 93.9 |
| DB-PI0-FSDP2 | 96.0 | 97.6 | 95.0 | 86.4 | 93.75 |
| DB-PI05 | 95.8 | 98.6 | 95.8 | 84.8 | 93.75 |
| DB-PI05-FSDP2 | 95.4 | 98.0 | 96.2 | 96.2 | 96.45 |
| DM0 | 98.2 | 98.8 | 96.6 | 82.6 | 94.1 |
| DM0-FSDP2 | 97.2 | 99.0 | 95.8 | 82.2 | 93.55 |

## Use FSDP2

For supported benchmark entrypoints, select FSDP2 with `--train-backend fsdp2`:

```bash
torchrun --nproc_per_node=8 playground/benchmarks/libero/libero_pi0.py \
  --task train \
  --train-backend fsdp2
```

The same backend flag is available for:

```bash
playground/benchmarks/libero/libero_pi0.py
playground/benchmarks/libero/libero_pi05.py
playground/benchmarks/libero/libero_dm0.py
playground/benchmarks/libero/libero_cogact.py
```

During startup, check the logs for one of these resolved modes:

- `resolved_mode=accelerate_fsdp2_plugin`
- `resolved_mode=trainer_fsdp2_native`


## Add FSDP2 Support for a New Model

### Required Steps

1. Add an FSDP profile to the model's `TrainerConfig`.

Start with the default root-only wrapping profile. This is a coarse but simple implementation, and is the recommended first step for adding FSDP2 support to a new model:

```python
from dexbotic.exp.base_exp import FSDPProfile

fsdp_profile: FSDPProfile = field(
    default_factory=lambda: FSDPProfile(
        enabled=True,
    )
)
```

Root-only wrapping lets FSDP/FSDP2 wrap the model at the top level. It avoids common issues in custom forward implementations that access transformer layer internals directly and bypass FSDP unshard hooks. After the model trains successfully, you can refine the wrap policy for better memory and performance.

2. Expose the backend flag in the training entrypoint.

```python
parser.add_argument(
    "--train-backend",
    type=str,
    default=None,
    choices=["deepspeed", "fsdp", "fsdp2", "ddp"],
)

if args.train_backend is not None:
    exp.trainer_config.train_backend = args.train_backend
```

3. Run a minimal validation.

- Start a short multi-GPU training run with `--train-backend fsdp2`.
- Confirm the log shows `accelerate_fsdp2_plugin` or `trainer_fsdp2_native`.
- Train until at least one checkpoint is saved.
- Confirm the checkpoint contains `pytorch_model_fsdp_*`.
- Resume once from that checkpoint.

### Optional Tuning

1. Use a finer wrap selector after root-only works.

If the model's forward path safely calls transformer block `forward()` methods, configure a wrap selector:

```python
fsdp_profile: FSDPProfile = field(
    default_factory=lambda: FSDPProfile(
        enabled=True,
        transformer_layer_cls_to_wrap=("YourDecoderLayer",),
    )
)
```

Examples:

- `dm0`: `("Qwen3MLP",)`
- `cogact`: `("Qwen2DecoderLayer",)`
- `pi0` / `pi05`: use root-only, because MoT forward directly accesses layer internals.

2. Set `cpu_ram_efficient_loading` only when needed.

```python
fsdp_profile: FSDPProfile = field(
    default_factory=lambda: FSDPProfile(
        enabled=True,
        cpu_ram_efficient_loading=False,
    )
)
```

Use `False` when the model initialization is not meta-device friendly, for example when `__init__` creates real tensors, precomputes diffusion schedules, or builds runtime buffers. Use `True` only after verifying the model can initialize safely in the FSDP loading path.

3. Cast the whole model to `bfloat16` only for backends that need it.

```python
fsdp_profile: FSDPProfile = field(
    default_factory=lambda: FSDPProfile(
        enabled=True,
        cast_model_to_bf16_backends=("fsdp",),
    )
)
```

This is mainly useful when FSDP1 flattening hits mixed-dtype parameters inside the same wrap unit. Do not add this by default for new models. Existing examples:

- `dm0`: casts for `("fsdp", "fsdp2")`.
- `cogact`: casts for `("fsdp", "fsdp2")`.

### Troubleshooting Checklist

If FSDP2 does not start cleanly, check these areas:

- `root_only`: keep the default root-only profile if your forward function directly reads layer internals such as layernorm or q/k/v/o projection modules.
- dtype/device: make sure manual `F.linear`, projector, vision tower, and action head calls receive tensors on the same dtype/device as their weights.
- initialization: avoid persistent large buffers for values that can be rebuilt at runtime; use runtime caches for large, reproducible caches.
- cache API: support `past_key_values[layer_idx]` or `past_key_values.update(...)` instead of relying only on `key_cache` / `value_cache` attributes.
- output embeddings: if loading, saving, or initialization reports `lm_head`, tied weights, or output embedding errors, check the Hugging Face model interface. Models that already work with `from_pretrained()` and `save_pretrained()` usually do not need extra changes for FSDP2.

For output embedding issues:

- Standard causal language models should expose `lm_head` via `get_output_embeddings()` and define tied weight keys only when the head should share weights with token embeddings.
- Action-only models without `lm_head` should make `get_output_embeddings()` return `None` and keep tied weight keys empty.
