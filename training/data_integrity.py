import hashlib
import json
import random

import numpy as np
import torch


def sequence_key(sequence):
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def load_jsonl_metadata(path):
    names = set()
    sequences = set()
    rows = 0
    ligand_rows = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc

            name = record.get("name")
            sequence = record.get("seq")
            if not isinstance(name, str) or not name:
                raise ValueError(f"Missing sample name in {path}:{line_number}")
            if not isinstance(sequence, str) or not sequence:
                raise ValueError(f"Missing sequence in {path}:{line_number}")
            if name in names:
                raise ValueError(f"Duplicate sample name {name!r} within {path}")

            names.add(name)
            sequences.add(sequence_key(sequence))
            rows += 1
            ligand_rows += bool(record.get("ligand_coords"))

    if rows == 0:
        raise ValueError(f"Dataset is empty: {path}")
    return {
        "path": str(path),
        "rows": rows,
        "names": names,
        "sequences": sequences,
        "ligand_rows": ligand_rows,
    }


def validate_disjoint_jsonl(splits):
    metadata = {name: load_jsonl_metadata(path) for name, path in splits.items()}
    split_names = list(metadata)
    errors = []
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            shared_names = metadata[left]["names"] & metadata[right]["names"]
            shared_sequences = metadata[left]["sequences"] & metadata[right]["sequences"]
            if shared_names or shared_sequences:
                errors.append(
                    f"{left}/{right}: {len(shared_names)} shared names, "
                    f"{len(shared_sequences)} shared sequences"
                )
    if errors:
        raise ValueError("Dataset leakage detected: " + "; ".join(errors))
    return metadata


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

