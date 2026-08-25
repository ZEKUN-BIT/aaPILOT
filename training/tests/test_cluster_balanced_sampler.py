import unittest

from cluster_balanced_sampler import ClusterBalancedSampler


class ClusterBalancedSamplerTests(unittest.TestCase):
    def test_large_cluster_does_not_dominate_sampling(self):
        records = [
            {"target": "thrA_AK", "cluster_id": "large"} for _ in range(100)
        ] + [{"target": "thrA_AK", "cluster_id": "small"}]
        sampler = ClusterBalancedSampler(records, num_samples=2000, seed=3)
        selected = list(sampler)
        small_fraction = sum(index == 100 for index in selected) / len(selected)
        self.assertGreater(small_fraction, 0.45)
        self.assertLess(small_fraction, 0.55)

    def test_epoch_changes_are_reproducible(self):
        records = [
            {"target": "thrA_AK", "cluster_id": "a"},
            {"target": "thrB", "cluster_id": "b"},
        ]
        sampler = ClusterBalancedSampler(records, num_samples=10, seed=9)
        first = list(sampler)
        sampler.set_epoch(1)
        second = list(sampler)
        sampler.set_epoch(0)
        self.assertEqual(list(sampler), first)
        self.assertNotEqual(second, first)

    def test_required_metadata_is_validated(self):
        with self.assertRaisesRegex(ValueError, "cluster_id"):
            ClusterBalancedSampler([{"target": "thrC"}])


if __name__ == "__main__":
    unittest.main()
