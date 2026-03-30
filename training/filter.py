import os
import requests
import subprocess
import random
import shutil
from pathlib import Path
from tqdm import tqdm
from Bio.PDB import MMCIFParser, MMCIF2Dict, Select
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.Polypeptide import protein_letters_3to1

# Configuration
PFAM_IDS =["PF00696", "PF00288", "PF00202", "PF14821", "PF08544"]
BASE_SAVE_DIR = "enzyme_dataset"
PROCESSED_DIR = "enzyme_dataset_processed_70"

# Filtering thresholds
MAX_PDB_RESOLUTION = 3.5
MIN_AF_PLDDT = 80.0
FLEXIBLE_PLDDT_THRESHOLD = 50.0

# MMseqs2 clustering thresholds
MMSEQS_IDENTITY = 0.8
MMSEQS_COVERAGE = 0.8

# Dataset split ratio
SPLIT_RATIO = {"train": 0.9, "val": 0.05, "test": 0.05}


class ResidueRangeSelect(Select):
    """Select residues within specified boundaries."""
    def __init__(self, start, end, chain_id=None):
        self.start = start
        self.end = end
        self.chain_id = chain_id

    def accept_residue(self, residue):
        resseq = residue.id[1]
        chain = residue.get_parent().id
        if self.chain_id and chain != self.chain_id:
            return 0
        if self.start <= resseq <= self.end:
            return 1
        return 0


def filter_pdb_structure(cif_path):
    """Filter PDB structures by resolution."""
    try:
        cif_dict = MMCIF2Dict.MMCIF2Dict(cif_path)
        res_str = cif_dict.get('_refine.ls_d_res_high', [''])[0]
        if res_str and res_str != '?':
            resolution = float(res_str)
            if resolution > MAX_PDB_RESOLUTION:
                return False, f"Resolution too low: {resolution}A"
            return True, "Pass"
        return False, "No resolution info"
    except Exception as e:
        return False, f"PDB parse error: {e}"


def get_pfam_boundaries_uniprot(uniprot_id, pfam_id):
    """Fetch Pfam domain boundaries via EBI UniProt API."""
    url = f"https://www.ebi.ac.uk/proteins/api/features/{uniprot_id}"
    try:
        response = requests.get(url, headers={"Accept": "application/json"}, timeout=10)
        if response.status_code == 200:
            data = response.json()
            for feature in data.get('features',[]):
                if feature.get('type') == 'DOMAIN' or feature.get('category') == 'DOMAINS_AND_SITES':
                    for ev in feature.get('evidences',[]):
                        if ev.get('source', '') == 'Pfam' and ev.get('id') == pfam_id:
                            return int(feature['begin']), int(feature['end'])
    except Exception:
        pass
    return None, None


def filter_and_truncate_af(cif_path, output_path, uniprot_id, pfam_id):
    """Truncate AlphaFold structure to Pfam domain and filter by pLDDT."""
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure("AF", cif_path)
        model = structure[0]
        chain = list(model.get_chains())[0]

        residues = list(chain.get_residues())
        plddt_scores = [(res.id[1], res['CA'].get_bfactor()) for res in residues if 'CA' in res]

        if not plddt_scores:
            return False, "No CA atoms found"

        # Try fetching exact boundaries via API
        domain_start, domain_end = get_pfam_boundaries_uniprot(uniprot_id, pfam_id)

        if domain_start and domain_end:
            core_plddt_scores =[score for score in plddt_scores if domain_start <= score[0] <= domain_end]
            if not core_plddt_scores:
                return False, "Boundary mismatch"
            real_start, real_end = domain_start, domain_end
        else:
            # Fallback: trim flexible ends
            start_idx, end_idx = 0, len(plddt_scores) - 1
            while start_idx < len(plddt_scores) and plddt_scores[start_idx][1] < FLEXIBLE_PLDDT_THRESHOLD:
                start_idx += 1
            while end_idx >= 0 and plddt_scores[end_idx][1] < FLEXIBLE_PLDDT_THRESHOLD:
                end_idx -= 1

            if start_idx >= end_idx:
                return False, "Valid structure too short"

            core_plddt_scores = plddt_scores[start_idx:end_idx + 1]
            real_start, real_end = plddt_scores[start_idx][0], plddt_scores[end_idx][0]

        avg_core_plddt = sum(score for _, score in core_plddt_scores) / len(core_plddt_scores)
        if avg_core_plddt < MIN_AF_PLDDT:
            return False, f"Low pLDDT: {avg_core_plddt:.1f}"

        io = MMCIFIO()
        io.set_structure(structure)
        io.save(output_path, select=ResidueRangeSelect(real_start, real_end))
        return True, "Pass"
    except Exception as e:
        return False, f"AF parse error: {e}"


def run_mmseqs(fasta_in, prefix_out, identity, coverage):
    """Run MMseqs2 for sequence clustering."""
    print(f"Running MMseqs2 (id={identity}, cov={coverage})...")
    tmp_dir = f"{prefix_out}_tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    cmd =[
        "mmseqs", "easy-cluster", fasta_in, prefix_out, tmp_dir,
        "--min-seq-id", str(identity),
        "-c", str(coverage),
        "--cov-mode", "1",
        "-v", "0"
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    retained_ids =[]
    rep_fasta = f"{prefix_out}_rep_seq.fasta"
    
    if os.path.exists(rep_fasta):
        with open(rep_fasta, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    retained_ids.append(line.strip()[1:])

    shutil.rmtree(tmp_dir, ignore_errors=True)
    for ext in ["_all_seqs.fasta", "_cluster.tsv"]:
        extraneous_file = f"{prefix_out}{ext}"
        if os.path.exists(extraneous_file):
            os.remove(extraneous_file)

    return retained_ids


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    all_retained_files =[]

    for pfam_id in PFAM_IDS:
        print(f"\n[{pfam_id}] Processing structures...")
        pfam_in_dir = os.path.join(BASE_SAVE_DIR, pfam_id)
        pfam_out_dir = os.path.join(PROCESSED_DIR, pfam_id, "cleaned_structures")
        os.makedirs(pfam_out_dir, exist_ok=True)

        fasta_records =[]

        # Process AlphaFold structures
        af_in_dir = os.path.join(pfam_in_dir, "AlphaFold_predicted")
        if os.path.exists(af_in_dir):
            af_files =[f for f in os.listdir(af_in_dir) if f.endswith('.cif')]
            for f_name in tqdm(af_files, desc="Process AF"):
                uid = f_name.split('.')[0]
                in_path = os.path.join(af_in_dir, f_name)
                out_path = os.path.join(pfam_out_dir, f_name)

                success, msg = filter_and_truncate_af(in_path, out_path, uid, pfam_id)
                if not success:
                    continue

                parser = MMCIFParser(QUIET=True)
                struct = parser.get_structure(uid, out_path)
                seq = ""
                for residue in struct.get_residues():
                    resname = residue.get_resname()
                    if residue.id[0] == ' ' and resname in protein_letters_3to1:
                        seq += protein_letters_3to1[resname]
                if seq:
                    fasta_records.append(f">{uid}_AF\n{seq}\n")

        # Process PDB structures
        pdb_in_dir = os.path.join(pfam_in_dir, "PDB_experimental")
        if os.path.exists(pdb_in_dir):
            pdb_files =[f for f in os.listdir(pdb_in_dir) if f.endswith('.cif')]
            for f_name in tqdm(pdb_files, desc="Process PDB"):
                pid = f_name.split('.')[0]
                in_path = os.path.join(pdb_in_dir, f_name)

                success, msg = filter_pdb_structure(in_path)
                if success:
                    out_path = os.path.join(pfam_out_dir, f_name)
                    shutil.copy2(in_path, out_path)

                    parser = MMCIFParser(QUIET=True)
                    struct = parser.get_structure(pid, out_path)
                    seq = "".join([
                        protein_letters_3to1[r.get_resname()] 
                        for r in struct.get_residues() 
                        if r.id[0] == ' ' and r.get_resname() in protein_letters_3to1
                    ])
                    if seq:
                        fasta_records.append(f">{pid}_PDB\n{seq}\n")

        # Run clustering
        if not fasta_records:
            print(f"[{pfam_id}] No valid structures passed filtering, skipping.")
            continue

        fasta_path = os.path.join(PROCESSED_DIR, pfam_id, "all_sequences.fasta")
        mmseqs_prefix = os.path.join(PROCESSED_DIR, pfam_id, "mmseqs_cluster")

        with open(fasta_path, "w") as f:
            f.writelines(fasta_records)

        retained_ids = run_mmseqs(fasta_path, mmseqs_prefix, MMSEQS_IDENTITY, MMSEQS_COVERAGE)
        print(f"[{pfam_id}] Clustering done: retained {len(retained_ids)} / {len(fasta_records)}.")

        for rid in retained_ids:
            uid_or_pid = rid.split('_')[0]
            ext = ".cif"
            all_retained_files.append(os.path.join(pfam_out_dir, uid_or_pid + ext))

    # Train / Val / Test Split
    if all_retained_files:
        print("\nSplitting dataset...")
        random.seed(42)
        random.shuffle(all_retained_files)

        total = len(all_retained_files)
        train_end = int(total * SPLIT_RATIO["train"])
        val_end = train_end + int(total * SPLIT_RATIO["val"])

        splits = {
            "train": all_retained_files[:train_end],
            "val": all_retained_files[train_end:val_end],
            "test": all_retained_files[val_end:]
        }

        for split_name, files in splits.items():
            split_dir = os.path.join(PROCESSED_DIR, f"dataset_{split_name}")
            os.makedirs(split_dir, exist_ok=True)
            for file_path in files:
                if os.path.exists(file_path):
                    shutil.copy2(file_path, os.path.join(split_dir, os.path.basename(file_path)))
            print(f"{split_name.capitalize()} set: {len(files)} structures.")

        print(f"\nFinished! Dataset saved to: {os.path.abspath(PROCESSED_DIR)}")


if __name__ == "__main__":
    main()
