"""Dataset-balanced sampling; Accelerate owns distributed batch sharding."""

import numpy as np
from torch.utils.data import Sampler


class WeightedMultiDatasetSampler(Sampler):
    """Choose a dataset by weight, then choose a frame uniformly within it.

    The random stream and default epoch length match the RoboTwin2 reference.
    Weights are per dataset, not per frame, so episode counts do not change the
    clean/randomized or task balance. Do not shard this sampler by rank.
    """

    def __init__(self, datasets, weights, num_samples=None, seed=42):
        self.dataset_sizes = np.asarray(
            [len(dataset) for dataset in datasets], dtype=np.int64
        )
        weights = np.asarray(weights, dtype=np.float64)
        if (
            weights.shape != self.dataset_sizes.shape
            or not len(weights)
            or not np.isfinite(weights).all()
            or (weights <= 0).any()
            or (self.dataset_sizes <= 0).any()
        ):
            raise ValueError(
                "Sampling requires nonempty datasets and one finite positive weight per dataset"
            )
        self.weights = weights / weights.sum()
        self.cumulative_sizes = np.concatenate(([0], np.cumsum(self.dataset_sizes)))
        self.total_samples = (
            int(num_samples)
            if num_samples is not None
            else int(self.dataset_sizes[0] / self.weights[0])
        )
        self.seed, self.epoch = seed, 0
        self.chunk_size = 1_000_000

    def __len__(self):
        return self.total_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        remaining = self.total_samples
        while remaining > 0:
            count = min(self.chunk_size, remaining)
            dataset_ids = rng.choice(
                len(self.dataset_sizes), size=count, p=self.weights
            )
            local_ids = np.empty(count, dtype=np.int64)
            for index, size in enumerate(self.dataset_sizes):
                positions = np.nonzero(dataset_ids == index)[0]
                if positions.size:
                    local_ids[positions] = rng.integers(
                        0, int(size), size=positions.size
                    )
            yield from (self.cumulative_sizes[dataset_ids] + local_ids).tolist()
            remaining -= count
