import glob
from pathlib import Path
import subprocess
import sys


BATCH_SIZE = 5
NUMBER_OF_BATCHES = 2
TEMPERATURES = [0.05, 0.1, 0.15, 0.2, 0.25]
OUT_BASE_DIR = Path("./seq_new")

MODELS = {
    "005": Path("./training/finetuned_ligand_models/ligandmpnn_v_32_005_25_finetuned.pt"),
    "010": Path("./training/finetuned_ligand_models/ligandmpnn_v_32_010_25_finetuned.pt"),
    "020": Path("./training/finetuned_ligand_models/ligandmpnn_v_32_020_25_finetuned.pt"),
    "030": Path("./training/finetuned_ligand_models/ligandmpnn_v_32_030_25_finetuned.pt"),
}

BASE_ARGS = [
    "--model_type", "ligand_mpnn",
    "--batch_size", str(BATCH_SIZE),
    "--number_of_batches", str(NUMBER_OF_BATCHES),
    "--fixed_residues", "",
    "--bias_AA", "A:0.8,Q:0.8,T:0.8,S:0.8,E:1.39,K:1.39,R:1.39",
    "--ligand_mpnn_use_side_chain_context", "1",
    "--save_stats", "1",
]


def require_inputs():
    if not Path("run.py").exists():
        raise FileNotFoundError(
            "run.py was not found. Run this script from the cloned LigandMPNN directory."
        )

    pdb_files = sorted(Path(path) for path in glob.glob("./inputs/*.pdb"))
    if not pdb_files:
        raise FileNotFoundError("No input PDB files found under ./inputs/*.pdb")

    missing_models = [str(path) for path in MODELS.values() if not path.exists()]
    if missing_models:
        raise FileNotFoundError(
            "Missing fine-tuned model checkpoint(s):\n" + "\n".join(missing_models)
        )

    return pdb_files


def run_generation(pdb_files):
    OUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    completed = 0

    for pdb_path in pdb_files:
        pdb_name = pdb_path.stem
        for model_name, model_path in MODELS.items():
            for temp in TEMPERATURES:
                out_folder = OUT_BASE_DIR / pdb_name / f"model_{model_name}_temp_{temp:.2f}"
                cmd = [
                    sys.executable, "run.py",
                    *BASE_ARGS,
                    "--pdb_path", str(pdb_path),
                    "--checkpoint_ligand_mpnn", str(model_path),
                    "--out_folder", str(out_folder),
                    "--temperature", str(temp),
                ]

                label = f"{pdb_name} | model {model_name} @ temp {temp:.2f}"
                print(f"Running {label}... ", end="", flush=True)
                process = subprocess.run(cmd, capture_output=True, text=True)

                if process.returncode == 0:
                    completed += 1
                    print("OK")
                else:
                    print("FAILED")
                    failures.append((label, process.stderr.strip()))

    return completed, failures


def main():
    pdb_files = require_inputs()
    completed, failures = run_generation(pdb_files)

    sequences_per_run = BATCH_SIZE * NUMBER_OF_BATCHES
    print(
        f"\nCompleted {completed} generation runs "
        f"({completed * sequences_per_run} requested sequences)."
    )

    if failures:
        print("\nFailures:")
        for label, stderr in failures:
            print(f"- {label}")
            if stderr:
                print(stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
