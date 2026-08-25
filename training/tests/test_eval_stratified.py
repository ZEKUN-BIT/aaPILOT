import unittest
from unittest.mock import patch

import numpy as np
import torch

from eval_stratified import bootstrap_delta, evaluate_pair


class RecordingModel:
    def __init__(self):
        self.orders = []

    def eval(self):
        return self

    def __call__(self, X, S, mask, chain_M, residue_idx, chain_encoding_all,
                 decoding_order, **kwargs):
        self.orders.append(decoding_order.clone())
        logits = torch.zeros((*S.shape, 21), device=S.device)
        return torch.log_softmax(logits, dim=-1)


class StratifiedEvaluationTests(unittest.TestCase):
    def test_models_share_identical_decoding_order(self):
        feat = {
            "X": torch.zeros((1, 2, 4, 3)),
            "S": torch.zeros((1, 2), dtype=torch.long),
            "mask": torch.ones((1, 2)),
            "chain_M": torch.ones((1, 2)),
            "residue_idx": torch.arange(2).reshape(1, 2),
            "chain_encoding_all": torch.ones((1, 2), dtype=torch.long),
            "Y": torch.zeros((1, 2, 1, 3)),
            "Y_t": torch.zeros((1, 2, 1), dtype=torch.long),
            "Y_m": torch.zeros((1, 2, 1)),
            "is_binding_site": torch.zeros((1, 2), dtype=torch.bool),
        }
        baseline, finetuned = RecordingModel(), RecordingModel()
        row = {"name": "x", "target": "thrB", "cluster_id": "c1",
               "structure_state": "predicted_no_ligand"}
        with patch("eval_stratified.featurize", return_value=feat):
            result = evaluate_pair(baseline, finetuned, [[row]], torch.device("cpu"), 1, 42)
        self.assertEqual(len(result), 1)
        self.assertTrue(torch.equal(baseline.orders[0], finetuned.orders[0]))

    def test_cluster_bootstrap_uses_cluster_means(self):
        delta, _, _, n = bootstrap_delta(
            [0.0, 0.0, 0.0], [1.0, 1.0, 0.0], np.random.default_rng(42),
            samples=100, groups=["large", "large", "small"])
        self.assertEqual(n, 2)
        self.assertEqual(delta, 0.5)


if __name__ == "__main__":
    unittest.main()
