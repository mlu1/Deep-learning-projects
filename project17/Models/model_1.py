#!/usr/bin/env python3
"""
Lightweight Denoising Autoencoder (AE‑64) — float16 embeddings
=============================================================
Single‑stage pipeline that trains an MLP denoising autoencoder on the
hand‑crafted feature matrix and saves **64‑dim** embeddings in float16.

Changes vs. the user‑supplied draft
----------------------------------
* **Explicit float16 cast** right before saving (`emb_all.astype(np.float16)`).
* Uses `astype(np.float32)` when feeding data to the model to keep PyTorch
  math stable, then converts the aggregated vectors to float16 for storage.
* Added `torch.cuda.amp.autocast` for minor speed boost on GPU (optional).

Outputs
-------
- `my_submission/ae_client_ids.npy`
- `my_submission/ae_embeddings.npy`  (float16, N × 64)
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast

# ─── CONFIG ───────────────────────────────────────────────────────────────
FEAT_NPY   = Path("user_features.npz")
#FEAT_IDX   = Path("my_save/user_feats_idx.npy")
OUTPUT_DIR = Path("my_submission"); OUTPUT_DIR.mkdir(exist_ok=True)
BATCH_SIZE = 256
EPOCHS     = 20
LR         = 1e-3
E_DIM    = 128
H1    = 256
H2    = 128
NOISE_STD  = 0.05
DROP_PROB  = 0.2
DEVICE     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DenoisingDataset(Dataset):
    def __init__(self, feat_npz, mmap_mode=None):
        # load the .npz once
        data = np.load(feat_npz, mmap_mode=mmap_mode)
        print(data)
        # extract the X array
        self.X = data['X']
        # optionally keep client_ids around
        self.client_ids = data['client_ids']
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        # clean sample
        clean  = torch.from_numpy(self.X[idx].astype(np.float32))
        # noised version
        noised = clean + torch.randn_like(clean) * NOISE_STD
        noised = nn.functional.dropout(noised, p=DROP_PROB)
        return noised, clean

# usage
ds     = DenoisingDataset(FEAT_NPY, mmap_mode='r')  
loader = DataLoader(
    ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# ─── MODEL ────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetMLPAutoencoder(nn.Module):
    def __init__(self, in_dim, hidden1, hidden2, emb_dim):
        super().__init__()
        # Encoder
        self.enc1 = nn.Linear(in_dim,   hidden1)
        self.enc2 = nn.Linear(hidden1,  hidden2)
        self.enc3 = nn.Linear(hidden2,  emb_dim)
        
        # Decoder
        # First up-projection from emb_dim back to hidden2 space
        self.up2   = nn.Linear(emb_dim, hidden2)
        # After concatenating with h2 (hidden2), project to hidden1
        self.dec2  = nn.Linear(hidden2*2, hidden1)
        # After concatenating with h1 (hidden1), project back to input dim
        self.dec1  = nn.Linear(hidden1*2, in_dim)

    def forward(self, x):
        # --- Encoder ---
        h1 = F.relu(self.enc1(x))      # (B, hidden1)
        h2 = F.relu(self.enc2(h1))     # (B, hidden2)
        z  = self.enc3(h2)             # (B, emb_dim)

        # --- Decoder ---
        d2 = F.relu(self.up2(z))       # (B, hidden2)
        # concat skip from h2
        d2_cat = torch.cat([d2, h2], dim=1)  # (B, hidden2*2)
        d1 = F.relu(self.dec2(d2_cat))       # (B, hidden1)

        # concat skip from h1
        d1_cat = torch.cat([d1, h1], dim=1)  # (B, hidden1*2)
        recon  = self.dec1(d1_cat)            # (B, in_dim)

        return recon, z


# ─── TRAIN ────────────────────────────────────────────────────────────────
INPUT_DIM = np.load(FEAT_NPY, mmap_mode=None)['X'].shape[1]
model = UNetMLPAutoencoder(in_dim=INPUT_DIM, hidden1=H1, hidden2=H2, emb_dim=E_DIM).to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for ep in range(1, EPOCHS+1):
    total_loss = 0.0
    model.train()
    for noised, clean in loader:
        noised, clean = noised.to(DEVICE), clean.to(DEVICE)
        opt.zero_grad()
        recon, _ = model(noised)
        loss = loss_fn(recon, clean)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    print(f"Epoch {ep}/{EPOCHS}  MSE={total_loss/len(loader):.4f}")

# ─── EMBEDDING EXTRACTION ─────────────────────────────────────────────────
print("Extracting embeddings …")
model.eval(); embs=[]
feat_mem = np.load(FEAT_NPY, mmap_mode=None)['X']
with torch.no_grad():
    for i in range(0, len(ds), BATCH_SIZE):
        batch = torch.from_numpy(feat_mem[i:i+BATCH_SIZE].astype(np.float32)).to(DEVICE)
        _, z = model(batch)
        embs.append(z.cpu())
emb_all = torch.cat(embs, 0).numpy().astype(np.float16)

# ─── SAVE -----------------------------------------------------------------
np.save(OUTPUT_DIR/"client_ids.npy", np.load(FEAT_NPY)['client_ids'])
np.save(OUTPUT_DIR/"embeddings.npy", emb_all)
print(f"Saved AE embeddings {emb_all.shape} (float16) to {OUTPUT_DIR}")

