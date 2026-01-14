import os
import random
from typing import Iterator, List, Optional

import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from dexbotic.exp.trainer import DexboticTrainer


def _get_rank_world():
    """
    Return (RANK, WORLD_SIZE) using common env vars.

    Priority:
      - RANK (set by torch.distributed/torchrun/accelerate)
      - fallback to LOCAL_RANK if RANK is absent
      - WORLD_SIZE defaults to 1 when not set

    NOTE:
    - We intentionally avoid importing torch.distributed here to keep this
      utility callable before the process group is initialized.
    """
    try:
        r = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    except Exception:
        r = 0
    try:
        w = int(os.environ.get("WORLD_SIZE", "1"))
    except Exception:
        w = 1
    return r, w


class CollatePassThrough:
    """
    Collate wrapper that preserves sample ordering metadata (indexes) across workers.

    Behavior:
      - Each sample may carry a 'indexes' tuple of (dataset_idx, file_idx, frame_idx)
      - We pop 'indexes' off samples and attach a consolidated list at batch-level
        as 'out["indexes"]' so the trainer can log exact per-sample ordering.

    NOTE:
    - The actual collate is delegated to 'base_collate'.
    - Do not forward 'indexes' to the model; trainers should pop it before compute.
    """

    def __init__(self, base_collate):
        self.base = base_collate

    def __call__(self, batch):
        tlist = []
        for s in batch:
            t = s.pop("indexes", None)
            if t is not None:
                tlist.append(tuple(int(x) for x in t))
        out = self.base(batch)
        if tlist:
            out["indexes"] = tlist
        return out


class LongVILADistributedSampler(DistributedSampler):
    """
    Distributed sampler (LongVILA style).

    Key features:
      - Block-based subsampling: Each rank gets a contiguous block of data,
        preserving intra-batch sample order (crucial for video/temporal data).
      - Interleaved multi-dataset sampling: Uniformly spreads samples from
        different datasets across the training epoch.
      - Batch-level shuffle: Maintains order within each batch.
    """

    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        batch_size: Optional[int] = None,
        sample_len_list: Optional[List[int]] = None,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )

        # Initialize attributes
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = True  # Always True for this implementation
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        if self.batch_size is None:
            raise ValueError(
                "batch_size must be provided for LongVILADistributedSampler"
            )

        # Dataset length management
        if sample_len_list is None:
            self.org_sample_len_list = [len(dataset)]
        else:
            self.org_sample_len_list = sample_len_list
            assert sum(sample_len_list) == len(
                self.dataset
            ), f"sum(sample_len_list) {sum(sample_len_list)} must match dataset size {len(self.dataset)}"

        # Calculate samples per replica (per rank) for each dataset
        # Each Rank gets a contiguous block that is a multiple of batch_size
        self.per_replica_samples = [
            (length // (self.num_replicas * self.batch_size)) * self.batch_size
            for length in self.org_sample_len_list
        ]
        self.num_samples = sum(self.per_replica_samples)

        # Total samples across all replicas that will be used (after drop_last)
        self.total_samples = [
            samples * self.num_replicas for samples in self.per_replica_samples
        ]

    def batch_shuffle(self, indices: List[int]) -> List[int]:
        """Shuffle batches while maintaining order within each batch."""
        if not indices or len(indices) < self.batch_size:
            return indices

        # Group indices into batches
        batches = [
            indices[i : i + self.batch_size]
            for i in range(0, len(indices), self.batch_size)
        ]
        random.shuffle(batches)

        # Flatten back
        return [idx for batch in batches for idx in batch]

    def __iter__(self) -> Iterator[int]:
        # 1. Full indices
        indices = list(range(len(self.dataset)))

        # 2. Split into dataset chunks and apply Block Slicing
        indices_list = []
        curr_offset = 0
        for i, org_len in enumerate(self.org_sample_len_list):
            dataset_total_used = self.total_samples[i]
            ds_indices = indices[curr_offset : curr_offset + dataset_total_used]

            # Subsample for this rank (Block Slicing)
            start = self.rank * self.per_replica_samples[i]
            end = (self.rank + 1) * self.per_replica_samples[i]
            rank_ds_indices = ds_indices[start:end]

            # Shuffle each dataset chunk at batch-level
            random.seed(self.seed + self.epoch + i)
            if self.shuffle:
                rank_ds_indices = self.batch_shuffle(rank_ds_indices)

            indices_list.append(rank_ds_indices)
            curr_offset += org_len

        # 3. Interleaved Mapping (spread datasets uniformly)
        all_indices = [-1] * self.num_samples
        indices_available = list(range(self.num_samples))

        # Sort datasets by length descending for better interleaving
        indices_list_sorted = sorted(
            [(i, indices_list[i]) for i in range(len(indices_list))],
            key=lambda x: -len(x[1]),
        )

        for _, ds_indices in indices_list_sorted:
            if not ds_indices:
                continue

            n = len(ds_indices)
            m = len(indices_available)

            # Uniform mapping logic
            mapped_pos = [i * m // n for i in range(n)]
            for i, pos_idx in enumerate(mapped_pos):
                slot_idx = indices_available[pos_idx]
                all_indices[slot_idx] = ds_indices[i]

            # Remove used slots
            for p in reversed(mapped_pos):
                del indices_available[p]

        assert -1 not in all_indices
        return iter(all_indices)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class DexboticNaVILATrainer(DexboticTrainer):
    """
    Trainer override that uses LongVILADistributedSampler for batch-level shuffle.

    This trainer applies batch shuffle strategy: maintains order within batches,
    but shuffles the order of batches. This is useful for long sequence training
    where maintaining batch coherence is important.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self):
        """
        Build a DataLoader that:
          - uses LongVILADistributedSampler for batch-level shuffle
          - wraps the collator with CollatePassThrough to propagate 'indexes'
        """
        # Get rank and world_size
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            num_replicas = dist.get_world_size()
        else:
            rank, num_replicas = _get_rank_world()

        # Try to extract sample_len_list from dataset (for multi-dataset interleaving)
        sample_len_list = getattr(self.train_dataset, "sample_len_list", None)
        if sample_len_list is None and hasattr(self.train_dataset, "cumulative_sizes"):
            # Handle standard ConcatDataset
            sizes = self.train_dataset.cumulative_sizes
            sample_len_list = [sizes[0]] + [
                sizes[i] - sizes[i - 1] for i in range(1, len(sizes))
            ]

        sampler = LongVILADistributedSampler(
            self.train_dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=True,
            seed=int(getattr(self.args, "seed", 42) or 42),
            # drop_last=self.args.dataloader_drop_last if hasattr(self.args, "dataloader_drop_last") else True,
            drop_last=True,
            batch_size=self.args.per_device_train_batch_size,
            sample_len_list=sample_len_list,
        )

        return DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=CollatePassThrough(self.data_collator),
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=(self.args.dataloader_num_workers > 0),
        )
