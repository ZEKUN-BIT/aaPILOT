import os
import subprocess

base_command =[
    "python", "run.py",
    "--model_type", "ligand_mpnn",
    "--pdb_path", "./inputs/*.pdb", #
    "--batch_size", "5",
    "--number_of_batches", "2",
    "--fixed_residues", "", #
    "--bias_AA", "A:0.8,Q:0.8,T:0.8,S:0.8,E:1.39,K:1.39,R:1.39",
    "--ligand_mpnn_use_side_chain_context", "1",
    "--save_stats", "1"
]

models = {
    "005": "./training/finetuned_ligand_models/ligandmpnn_v_32_005_25_finetuned.pt",
    "010": "./training/finetuned_ligand_models/ligandmpnn_v_32_010_25_finetuned.pt",
    "020": "./training/finetuned_ligand_models/ligandmpnn_v_32_020_25_finetuned.pt",
    "030": "./training/finetuned_ligand_models/ligandmpnn_v_32_030_25_finetuned.pt"
}

temperatures =[0.05, 0.1, 0.15, 0.2, 0.25]
out_base_dir = "./seq_new"

os.makedirs(out_base_dir, exist_ok=True)

# Run model and temperature combinations
for model_name, model_path in models.items():
    for temp in temperatures:
        out_folder = f"{out_base_dir}/model_{model_name}_temp_{temp:.2f}"
        
        cmd = base_command +[
            "--checkpoint_ligand_mpnn", model_path,
            "--out_folder", out_folder,
            "--temperature", str(temp)
        ]
        
        print(f"Running model {model_name} @ temp {temp:.2f}... ", end="", flush=True)
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode == 0:
            print("OK")
        else:
            print(f"FAILED\n{process.stderr}")

print("\nAll batches completed (Total: 200 sequences).")
