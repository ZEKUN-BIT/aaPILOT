An end-to-end pipeline designed to filter, process, fine-tune, and generate sequences using **LigandMPNN** for specific enzyme families. 

This repository provides tools to automatically clean experimental PDBs and AlphaFold structures, parse 3D ligand-protein coordinates, fine-tune pre-trained LigandMPNN models (preventing catastrophic forgetting), and run automated batch inference across multiple model checkpoints and temperatures.

## Features

- **Automated Structure Filtering (`training/filter.py`)**: 
  - Filters PDB structures by resolution (< 3.5Å).
  - Truncates AlphaFold predictions to exact Pfam domain boundaries via EBI UniProt API.
  - Trims flexible AlphaFold regions based on pLDDT scores (> 80.0 avg, > 50.0 per residue).
  - Removes redundant sequences using **MMseqs2** clustering (80% identity, 80% coverage).
- **Efficient Data Parsing (`training/parse_cif.py`)**: Converts `.cif` structural files into lightweight `.jsonl` formats, extracting protein backbones (N, CA, C, O) and ligand atom coordinates/types.
- **Robust Fine-Tuning (`training/train.py`)**: 
  - Uses `MixedStructureDataset` to mix domain-specific data with general protein data, preventing catastrophic forgetting.
  - Supports synchronized Gaussian noise injection during training.
  - Employs Automatic Mixed Precision (AMP) and Gradient Accumulation for memory-efficient training.
- **Automated Sequence Generation (`generate.py`)**: A wrapper script designed to perform batch sequence generation on input PDBs, iterating through fine-tuned checkpoints and a temperature sweep, while applying customized amino acid biases.



## Installation & Requirements

1. **Python Environment**: Python 3.8+ and PyTorch (CUDA supported).
2. **Python Packages**:
   ```bash
   pip install torch numpy biopython requests tqdm
   ```
3. **External Tools**: 
   - [MMseqs2](https://github.com/soedinglab/MMseqs2) is required for sequence clustering. Ensure `mmseqs` is installed and accessible in your system's PATH.

## Pipeline Usage

### Step 0: Download Raw Data & Pre-trained Weights

**1. Raw Enzyme Dataset:**  
The raw PDB and AlphaFold `.cif` files required for this pipeline are hosted on Zenodo. Please download the dataset and extract it to the project root directory.
- **Zenodo Download Link**: [Insert Zenodo URL Here]() 

**2. General Training Data (`ligand_mpnn_train_data.jsonl`):**  
To prevent catastrophic forgetting during domain-specific fine-tuning, the script mixes in general protein data. **This file is simply the data from the `train.json` provided in the original LigandMPNN repository.** Please make sure this file is placed in your `training/` directory before running the training script.

**3. Pre-trained LigandMPNN Weights:**  
Ensure the pre-trained weights are located in `model_params/`.

---

### Step 1: Filter and Split Dataset

Once the Zenodo data is extracted, ensure your raw `.cif` files are in the `training/enzyme_dataset/<PFAM_ID>/` directory.

```bash
cd training
python filter.py
```
*Output: Cleaned, truncated, and clustered structures split into train/val/test sets under `training/enzyme_dataset_processed_70/`.*

### Step 2: Parse to JSON Lines Format

Convert the processed `.cif` files into the dictionary format required for fine-tuning.

```bash
python parse_cif.py
```
*Output: `train.jsonl`, `val.jsonl`, `test.jsonl` in the `training/mpnn_finetune_data/` directory.*

### Step 3: Model Fine-Tuning

Run the training script to begin fine-tuning:

```bash
python train.py
```
*Output: Fine-tuned model checkpoints will be saved in the `training/finetuned_ligand_models/` directory.*

---

### Step 4: Sequence Generation (`generate.py`)

The generation script (`generate.py`) acts as a wrapper that calls the `run.py` script from the original LigandMPNN repository. To use it, you must clone the original LigandMPNN codebase.

**1. Clone the original LigandMPNN repository:**
```bash
git clone https://github.com/dauparas/LigandMPNN.git
```

**2. Setup the generation environment:**
Move or copy `generate.py`, your fine-tuned models, and input PDBs into the cloned LigandMPNN directory:
```bash
cp generate.py LigandMPNN/
cp -r training/finetuned_ligand_models LigandMPNN/training/
mkdir -p LigandMPNN/inputs
# (Place your target .pdb files in LigandMPNN/inputs/)
cd LigandMPNN
```

**3. Run the batch generation:**
```bash
python generate.py
```

**What `generate.py` does:**
- Iterates over 4 different fine-tuned models (noise levels: 0.05, 0.10, 0.20, 0.30) and 5 sampling temperatures (0.05 to 0.25).
- Processes inputs in batches, outputting a total of 200 designed sequences.
- Automatically applies an amino acid bias (e.g., boosting `E`, `K`, `R` frequencies) and enforces ligand side-chain context (`--ligand_mpnn_use_side_chain_context 1`).
- Saves all generated sequences and metrics in the `./seq_new/` directory, categorized by model and temperature.

##  Acknowledgments

This codebase builds upon the architecture and principles of [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) and [LigandMPNN](https://github.com/dauparas/LigandMPNN) by the Baker Lab.
