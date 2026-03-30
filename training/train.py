import os
import json
import random
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

from model_utils import ProteinMPNN, loss_nll, loss_smoothed

# ==========================================
# Configuration
# ==========================================
DATA_DIR = "mpnn_finetune_data"
OUTPUT_DIR = "finetuned_ligand_models"

# Define the models you want to iterate through
MODELS_TO_FINETUNE = [
    "ligandmpnn_v_32_005_25",
    "ligandmpnn_v_32_010_25",
    "ligandmpnn_v_32_020_25",
    "ligandmpnn_v_32_030_25"
]

EPOCHS = 50
BATCH_SIZE = 1
ACCUMULATION_STEPS = 8
LEARNING_RATE = 5e-5
ATOM_CONTEXT_NUM = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Dataset Loader
# ==========================================
class MixedStructureDataset(Dataset):
    def __init__(self, specific_jsonl, general_jsonl, mix_ratio=0.05, max_length=1200, seed=42):
        """
        Mixed dataset loader for domain-specific fine-tuning to prevent catastrophic forgetting.
        """
        self.data = []
        specific_data = self._load_jsonl(specific_jsonl, max_length)
        num_specific = len(specific_data)

        if mix_ratio > 0 and general_jsonl and os.path.exists(general_jsonl):
            general_data = self._load_jsonl(general_jsonl, max_length)
            num_general_needed = int(num_specific * mix_ratio / (1 - mix_ratio))
            
            random.seed(seed)
            if num_general_needed <= len(general_data):
                sampled_general = random.sample(general_data, num_general_needed)
            else:
                sampled_general = random.choices(general_data, k=num_general_needed)
        else:
            sampled_general = []
            
        self.data = specific_data + sampled_general
        random.shuffle(self.data)
        print(f"Loaded {num_specific} specific and {len(sampled_general)} general structures. Total: {len(self.data)}")

    def _load_jsonl(self, file_path, max_length):
        loaded = []
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                if len(item['seq']) <= max_length:
                    loaded.append(item)
        return loaded

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    return batch

# ==========================================
# Featurization
# ==========================================
def featurize_ligand_mpnn(batch, device, atom_context_num=32):
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    B = len(batch)
    lengths = [len(b['seq']) for b in batch]
    L_max = max(lengths)

    X = np.zeros([B, L_max, 4, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    mask = np.zeros([B, L_max], dtype=np.float32)
    chain_M = np.zeros([B, L_max], dtype=np.float32)
    residue_idx = -100 * np.ones([B, L_max], dtype=np.int32)
    chain_encoding_all = np.ones([B, L_max], dtype=np.int32)

    Y = np.zeros([B, L_max, atom_context_num, 3])
    Y_t = np.zeros([B, L_max, atom_context_num], dtype=np.int32)
    Y_m = np.zeros([B, L_max, atom_context_num], dtype=np.float32)

    for i, b in enumerate(batch):
        seq = b['seq']
        l_seq = len(seq)
        mask[i, :l_seq] = 1.0
        chain_M[i, :l_seq] = 1.0 

        chain_keys = [k.replace('seq_chain_', '') for k in b.keys() if k.startswith('seq_chain_')]
        global_idx = 0
        all_ca_coords = []
        
        for c_idx, chain_id in enumerate(chain_keys):
            c_seq = b[f'seq_chain_{chain_id}']
            c_len = len(c_seq)
            
            for j, aa in enumerate(c_seq):
                S[i, global_idx + j] = alphabet.index(aa) if aa in alphabet else 20
            
            chain_encoding_all[i, global_idx:global_idx + c_len] = c_idx + 1
            residue_idx[i, global_idx:global_idx + c_len] = 100 * c_idx + np.arange(c_len)

            c_coords = b[f'coords_chain_{chain_id}']
            X[i, global_idx:global_idx + c_len, 0, :] = c_coords[f'N_chain_{chain_id}']
            X[i, global_idx:global_idx + c_len, 1, :] = c_coords[f'CA_chain_{chain_id}']
            X[i, global_idx:global_idx + c_len, 2, :] = c_coords[f'C_chain_{chain_id}']
            X[i, global_idx:global_idx + c_len, 3, :] = c_coords[f'O_chain_{chain_id}']
            
            all_ca_coords.extend(c_coords[f'CA_chain_{chain_id}'])
            global_idx += c_len

        # Ligand context extraction
        if "ligand_coords" in b and len(b["ligand_coords"]) > 0:
            l_coords = np.array(b["ligand_coords"]) 
            l_types = np.array(b["ligand_types"])   
            p_ca_coords = np.array(all_ca_coords)   

            dists = np.linalg.norm(p_ca_coords[:, None, :] - l_coords[None, :, :], axis=-1)

            for r_idx in range(l_seq):
                res_dists = dists[r_idx]
                k = min(len(l_coords), atom_context_num)
                top_k_indices = np.argsort(res_dists)[:k]
                
                Y[i, r_idx, :k, :] = l_coords[top_k_indices]
                Y_t[i, r_idx, :k] = l_types[top_k_indices]
                Y_m[i, r_idx, :k] = 1.0

    return {
        "X": torch.from_numpy(X).to(dtype=torch.float32, device=device),
        "S": torch.from_numpy(S).to(dtype=torch.long, device=device),
        "mask": torch.from_numpy(mask).to(dtype=torch.float32, device=device),
        "chain_M": torch.from_numpy(chain_M).to(dtype=torch.float32, device=device),
        "residue_idx": torch.from_numpy(residue_idx).to(dtype=torch.long, device=device),
        "chain_encoding_all": torch.from_numpy(chain_encoding_all).to(dtype=torch.long, device=device),
        "Y": torch.from_numpy(Y).to(dtype=torch.float32, device=device),
        "Y_t": torch.from_numpy(Y_t).to(dtype=torch.long, device=device),
        "Y_m": torch.from_numpy(Y_m).to(dtype=torch.float32, device=device)
    }

# ==========================================
# Training Pipeline
# ==========================================
def train_model(model_name, train_loader, val_loader):
    """Handles the training loop for a specific model."""
    ckpt_path = f"../model_params/{model_name}.pt"
    if not os.path.exists(ckpt_path):
        print(f"Skipping {model_name}: Weights not found at {ckpt_path}")
        return

    # Extract noise level directly from model name (e.g., 020 -> 0.2)
    # Assumes naming convention: ligandmpnn_v_32_020_25
    parsed_noise = float(model_name.split('_')[3]) / 100.0
    print(f"\n[{model_name}] Starting training with synced noise level: {parsed_noise}A")

    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_dict = checkpoint if 'model_state_dict' not in checkpoint else checkpoint['model_state_dict']
    
    num_edges = checkpoint.get('num_edges', 25) 
    hidden_dim = checkpoint.get('hidden_dim', 128)
    num_layers = checkpoint.get('num_encoder_layers', 3)

    model = ProteinMPNN(
        num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim,
        num_encoder_layers=num_layers, num_decoder_layers=num_layers,
        k_neighbors=num_edges, augment_eps=0.0, dropout=0.1,
        model_type="ligand_mpnn", atom_context_num=ATOM_CONTEXT_NUM
    )

    model.load_state_dict(checkpoint_dict)
    model.to(device)

    # Freeze encoder, unfreeze final decoder layer and output weights
    for name, param in model.named_parameters():
        param.requires_grad = False
        if "W_out" in name or f"decoder_layers.{num_layers - 1}" in name:
           param.requires_grad = True  

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch_data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")):
            feat = featurize_ligand_mpnn(batch_data, device, ATOM_CONTEXT_NUM)

            # Synchronized noise injection
            noise_X = torch.randn_like(feat["X"]) * parsed_noise * feat["mask"][:, :, None, None]
            X_noised = feat["X"] + noise_X
            randn_order = torch.randn_like(feat["mask"])
            
            valid_mask = feat["mask"] * feat["chain_M"]
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16): 
                log_probs = model(
                    X_noised, feat["S"], feat["mask"], feat["chain_M"], 
                    feat["residue_idx"], feat["chain_encoding_all"], randn_order,
                    Y=feat["Y"], Y_t=feat["Y_t"], Y_m=feat["Y_m"]
                )
                
                _, loss_av = loss_smoothed(feat["S"], log_probs, valid_mask, weight=0.1)
                loss_scaled = loss_av / ACCUMULATION_STEPS
            
            scaler.scale(loss_scaled).backward()
            
            if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
              
            train_loss += loss_av.item()
            
        avg_train_loss = train_loss / len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for batch_data in tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]"):
                feat = featurize_ligand_mpnn(batch_data, device, ATOM_CONTEXT_NUM)
                randn_order = torch.randn_like(feat["mask"])
                valid_mask = feat["mask"] * feat["chain_M"]
                
                log_probs = model(
                    feat["X"], feat["S"], feat["mask"], feat["chain_M"], 
                    feat["residue_idx"], feat["chain_encoding_all"], randn_order,
                    Y=feat["Y"], Y_t=feat["Y_t"], Y_m=feat["Y_m"]
                )

                _, loss_av = loss_nll(feat["S"], log_probs, valid_mask)
                val_loss += loss_av.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(OUTPUT_DIR, f"{model_name}_finetuned.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, save_path)
            print(f"--> Saved best model checkpoint to: {save_path}")


def main():
    # Initialize datasets and loaders once to save I/O overhead
    print("Initializing datasets...")
    train_dataset = MixedStructureDataset(
        specific_jsonl=os.path.join(DATA_DIR, "train.jsonl"), 
        general_jsonl="ligand_mpnn_train_data.jsonl", 
        mix_ratio=0.1, 
        max_length=1200
    )
    
    val_dataset = MixedStructureDataset(
        specific_jsonl=os.path.join(DATA_DIR, "val.jsonl"),
        general_jsonl=None,
        mix_ratio=0.0
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Iterate through all configured models
    for model_name in MODELS_TO_FINETUNE:
        train_model(model_name, train_loader, val_loader)

if __name__ == "__main__":
    main()
