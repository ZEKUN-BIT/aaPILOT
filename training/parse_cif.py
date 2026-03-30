import os
import json
from tqdm import tqdm
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import protein_letters_3to1

PROCESSED_DIR = "enzyme_dataset_processed_70"
OUTPUT_DIR = "mpnn_finetune_data"

# Map elements to atomic numbers (defaulting to C=6)
ELEMENT_MAP = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 
    'MG': 12, 'P': 15, 'S': 16, 'CL': 17, 'ZN': 30
}


def get_atomic_number(atom):
    element = atom.element.upper()
    return ELEMENT_MAP.get(element, 6)


def parse_cif_to_mpnn_dict(cif_path, name):
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(name, cif_path)
    except Exception as e:
        print(f"Failed to parse {name}: {e}")
        return None

    model = structure[0]
    protein_dict = {"name": name, "num_alignments": 1, "seq": ""}
    
    ligand_coords = []
    ligand_types =[]
    full_seq = ""

    for chain in model:
        chain_id = chain.id
        chain_seq = ""
        n_coords, ca_coords, c_coords, o_coords = [], [], [],[]

        for residue in chain:
            res_id = residue.get_id()
            
            # Process protein residues
            if res_id[0] == ' ':
                resname = residue.get_resname()
                if resname not in protein_letters_3to1:
                    continue
                
                if all(atom in residue for atom in ['N', 'CA', 'C', 'O']):
                    chain_seq += protein_letters_3to1[resname]
                    n_coords.append(residue['N'].get_coord().tolist())
                    ca_coords.append(residue['CA'].get_coord().tolist())
                    c_coords.append(residue['C'].get_coord().tolist())
                    o_coords.append(residue['O'].get_coord().tolist())
            
            # Process ligands (excluding water)
            elif res_id[0].startswith('H_'):
                if residue.get_resname() == 'HOH':
                    continue
                
                for atom in residue:
                    ligand_coords.append(atom.get_coord().tolist())
                    ligand_types.append(get_atomic_number(atom))

        if chain_seq:
            full_seq += chain_seq
            protein_dict[f"seq_chain_{chain_id}"] = chain_seq
            protein_dict[f"coords_chain_{chain_id}"] = {
                f"N_chain_{chain_id}": n_coords,
                f"CA_chain_{chain_id}": ca_coords,
                f"C_chain_{chain_id}": c_coords,
                f"O_chain_{chain_id}": o_coords
            }

    if not full_seq:
        return None

    protein_dict["seq"] = full_seq
    protein_dict["ligand_coords"] = ligand_coords
    protein_dict["ligand_types"] = ligand_types
    
    return protein_dict


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    splits = ["train", "val", "test"]

    for split in splits:
        input_dir = os.path.join(PROCESSED_DIR, f"dataset_{split}")
        output_file = os.path.join(OUTPUT_DIR, f"{split}.jsonl")

        if not os.path.exists(input_dir):
            print(f"Directory not found: {input_dir}, skipping.")
            continue

        cif_files =[f for f in os.listdir(input_dir) if f.endswith('.cif')]
        print(f"\nProcessing {split} set ({len(cif_files)} structures)...")

        valid_records = 0
        with open(output_file, 'w') as f_out:
            for cif_name in tqdm(cif_files):
                cif_path = os.path.join(input_dir, cif_name)
                pdb_id = os.path.splitext(cif_name)[0]

                mpnn_dict = parse_cif_to_mpnn_dict(cif_path, pdb_id)
                if mpnn_dict:
                    f_out.write(json.dumps(mpnn_dict) + '\n')
                    valid_records += 1

        print(f"{split.capitalize()} set completed: {valid_records}/{len(cif_files)} records saved to {output_file}")


if __name__ == "__main__":
    main()
