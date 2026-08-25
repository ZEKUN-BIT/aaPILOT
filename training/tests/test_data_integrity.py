import json
import tempfile
import unittest
from pathlib import Path

from data_integrity import load_jsonl_metadata, validate_disjoint_jsonl


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


class DataIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_disjoint_splits_pass(self):
        train = self.root / "train.jsonl"
        val = self.root / "val.jsonl"
        write_jsonl(train, [{"name": "train-a", "seq": "AAAA", "ligand_coords": []}])
        write_jsonl(val, [{"name": "val-a", "seq": "CCCC", "ligand_coords": [[0, 0, 0]]}])
        metadata = validate_disjoint_jsonl({"train": train, "val": val})
        self.assertEqual(metadata["train"]["rows"], 1)
        self.assertEqual(metadata["val"]["ligand_rows"], 1)

    def test_shared_sequence_is_rejected(self):
        train = self.root / "train.jsonl"
        test = self.root / "test.jsonl"
        write_jsonl(train, [{"name": "one", "seq": "ACDE"}])
        write_jsonl(test, [{"name": "two", "seq": "ACDE"}])
        with self.assertRaisesRegex(ValueError, "1 shared sequences"):
            validate_disjoint_jsonl({"train": train, "test": test})

    def test_duplicate_name_within_split_is_rejected(self):
        dataset = self.root / "data.jsonl"
        write_jsonl(dataset, [
            {"name": "duplicate", "seq": "AAAA"},
            {"name": "duplicate", "seq": "CCCC"},
        ])
        with self.assertRaisesRegex(ValueError, "Duplicate sample name"):
            load_jsonl_metadata(dataset)


if __name__ == "__main__":
    unittest.main()
