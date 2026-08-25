"""Shared LigandMPNN JSONL batch featurization."""

import numpy as np
import torch


ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def featurize_ligand_mpnn(batch, device, atom_context_num=25):
    batch_size = len(batch)
    lengths = [len(record["seq"]) for record in batch]
    max_length = max(lengths)

    x = np.zeros((batch_size, max_length, 4, 3))
    sequence = np.zeros((batch_size, max_length), dtype=np.int32)
    mask = np.zeros((batch_size, max_length), dtype=np.float32)
    chain_mask = np.zeros((batch_size, max_length), dtype=np.float32)
    residue_idx = -100 * np.ones((batch_size, max_length), dtype=np.int32)
    chain_encoding = np.ones((batch_size, max_length), dtype=np.int32)
    ligand_xyz = np.zeros((batch_size, max_length, atom_context_num, 3))
    ligand_types = np.zeros((batch_size, max_length, atom_context_num), dtype=np.int32)
    ligand_mask = np.zeros((batch_size, max_length, atom_context_num), dtype=np.float32)

    for batch_index, record in enumerate(batch):
        length = len(record["seq"])
        mask[batch_index, :length] = 1.0
        chain_mask[batch_index, :length] = 1.0
        chain_ids = [key.removeprefix("seq_chain_") for key in record
                     if key.startswith("seq_chain_")]
        offset = 0
        all_ca = []
        for chain_index, chain_id in enumerate(chain_ids):
            chain_sequence = record[f"seq_chain_{chain_id}"]
            chain_length = len(chain_sequence)
            for residue_index, amino_acid in enumerate(chain_sequence):
                sequence[batch_index, offset + residue_index] = (
                    ALPHABET.index(amino_acid) if amino_acid in ALPHABET else 20
                )
            chain_encoding[batch_index, offset:offset + chain_length] = chain_index + 1
            residue_idx[batch_index, offset:offset + chain_length] = (
                100 * chain_index + np.arange(chain_length)
            )
            coordinates = record[f"coords_chain_{chain_id}"]
            for atom_index, atom_name in enumerate(("N", "CA", "C", "O")):
                x[batch_index, offset:offset + chain_length, atom_index] = (
                    coordinates[f"{atom_name}_chain_{chain_id}"]
                )
            all_ca.extend(coordinates[f"CA_chain_{chain_id}"])
            offset += chain_length

        if record.get("ligand_coords"):
            coordinates = np.asarray(record["ligand_coords"])
            atom_types = np.asarray(record["ligand_types"])
            distances = np.linalg.norm(
                np.asarray(all_ca)[:, None, :] - coordinates[None, :, :], axis=-1
            )
            for residue_index in range(length):
                count = min(len(coordinates), atom_context_num)
                nearest = np.argsort(distances[residue_index])[:count]
                ligand_xyz[batch_index, residue_index, :count] = coordinates[nearest]
                ligand_types[batch_index, residue_index, :count] = atom_types[nearest]
                ligand_mask[batch_index, residue_index, :count] = 1.0

    return {
        "X": torch.as_tensor(x, dtype=torch.float32, device=device),
        "S": torch.as_tensor(sequence, dtype=torch.long, device=device),
        "mask": torch.as_tensor(mask, dtype=torch.float32, device=device),
        "chain_M": torch.as_tensor(chain_mask, dtype=torch.float32, device=device),
        "residue_idx": torch.as_tensor(residue_idx, dtype=torch.long, device=device),
        "chain_encoding_all": torch.as_tensor(chain_encoding, dtype=torch.long, device=device),
        "Y": torch.as_tensor(ligand_xyz, dtype=torch.float32, device=device),
        "Y_t": torch.as_tensor(ligand_types, dtype=torch.long, device=device),
        "Y_m": torch.as_tensor(ligand_mask, dtype=torch.float32, device=device),
    }
