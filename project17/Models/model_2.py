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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast

# ─── CONFIG ───────────────────────────────────────────────────────────────
FEAT_NPY   = Path("my_save/user_feats.npy")
FEAT_IDX   = Path("my_save/user_feats_idx.npy")
OUTPUT_DIR = Path("my_submission"); OUTPUT_DIR.mkdir(exist_ok=True)
BATCH_SIZE = 256
EPOCHS     = 10
LR         = 1e-3
EMB_DIM    = 64
HIDDEN1    = 256
HIDDEN2    = 128
NOISE_STD  = 0.05
DROP_PROB  = 0.2
DEVICE     = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# ─── DATASET ───────────────────────────────────────────────────────────────
class DenoisingDataset(Dataset):
    def __init__(self, feat_npy):
        self.mem = np.load(feat_npy, mmap_mode='r')
    def __len__(self):
        return self.mem.shape[0]
    def __getitem__(self, idx):
        x = self.mem[idx].astype(np.float32)
        clean  = torch.from_numpy(x)
        noised = clean + torch.randn_like(clean) * NOISE_STD
        noised = nn.functional.dropout(noised, p=DROP_PROB)
        return noised, clean

ds = DenoisingDataset(FEAT_NPY)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

class ResidBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(0.2)
    def forward(self, x):
        h = self.drop(F.gelu(self.bn1(self.fc1(x))))
        h = self.drop(F.gelu(self.bn2(self.fc2(h))))
        return x + h

class Autoencoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.enc_head = nn.Sequential(nn.Linear(in_dim, HIDDEN1), nn.BatchNorm1d(HIDDEN1), nn.GELU())
        self.enc_mid  = nn.Sequential(ResidBlock(HIDDEN1), ResidBlock(HIDDEN1))
        self.mu       = nn.Linear(HIDDEN1, EMB_DIM)
        self.dec = nn.Sequential(nn.Linear(EMB_DIM, HIDDEN1), nn.GELU(), ResidBlock(HIDDEN1), nn.Linear(HIDDEN1, in_dim))
    def forward(self, x):
        h = self.enc_mid(self.enc_head(x))
        z = self.mu(h)
        recon = self.dec(z)
        return recon, z



in_dim = np.load(FEAT_NPY, mmap_mode='r').shape[1]
model = Autoencoder(in_dim).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()
print(f"Training AE on {DEVICE} …")
for ep in range(1, EPOCHS+1):
    tot = 0
    model.train()
    for noisy, clean in loader:
        noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
        opt.zero_grad()
        with autocast(enabled=DEVICE.type=='cuda:1'):
            recon, _ = model(noisy)
            loss = loss_fn(recon, clean)
        loss.backward(); opt.step(); tot += loss.item()
    print(f"Epoch {ep}/{EPOCHS}  MSE={tot/len(loader):.4f}")

# ─── EMBEDDING EXTRACTION ─────────────────────────────────────────────────
print("Extracting embeddings …")
model.eval(); embs=[]
feat_mem = np.load(FEAT_NPY, mmap_mode='r')
with torch.no_grad():
    for i in range(0, len(ds), BATCH_SIZE):
        batch = torch.from_numpy(feat_mem[i:i+BATCH_SIZE].astype(np.float32)).to(DEVICE)
        _, z = model(batch)
        embs.append(z.cpu())
emb_all = torch.cat(embs, 0).numpy().astype(np.float16)

# ─── SAVE -----------------------------------------------------------------
np.save(OUTPUT_DIR/"client_ids.npy", np.load(FEAT_IDX))
np.save(OUTPUT_DIR/"embeddings.npy", emb_all)
print(f"Saved AE embeddings {emb_all.shape} (float16) to {OUTPUT_DIR}")

