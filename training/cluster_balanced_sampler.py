import random
from collections import defaultdict

from torch.utils.data import Sampler


class ClusterBalancedSampler(Sampler):
    """Sample targets, then clusters, then members with equal probabilities."""

    def __init__(self, records, num_samples=None, seed=42):
        self.num_samples = len(records) if num_samples is None else num_samples
        if self.num_samples < 1:
            raise ValueError("num_samples must be positive")
        self.seed = seed
        self.epoch = 0
        grouped = defaultdict(lambda: defaultdict(list))
        for index, record in enumerate(records):
            target = record.get("target")
            cluster_id = record.get("cluster_id")
            if not isinstance(target, str) or not target:
                raise ValueError(f"Record {index} has no target")
            if not isinstance(cluster_id, str) or not cluster_id:
                raise ValueError(f"Record {index} has no cluster_id")
            grouped[target][cluster_id].append(index)
        self.grouped = {
            target: {cluster: indices for cluster, indices in sorted(clusters.items())}
            for target, clusters in sorted(grouped.items())
        }
        self.targets = tuple(self.grouped)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.num_samples):
            target = rng.choice(self.targets)
            clusters = self.grouped[target]
            cluster_id = rng.choice(tuple(clusters))
            yield rng.choice(clusters[cluster_id])

    def __len__(self):
        return self.num_samples
