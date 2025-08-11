#!/usr/bin/env python3
"""
Single-file DL for Cadastral Plan Extraction (Upgraded)

- Polygon vertex regression with a pretrained backbone + coord head
- Image-size aware normalization, angle-sorted vertices
- AMP, cosine LR with warmup, gradient clipping, optional EMA
- Robust OCR heuristics (dates, area, units incl. acres/ha)
- Outputs proper WKT POLYGON
- No visualization. One file.
"""

import os, re, math, json, warnings, random, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.transforms as T
import torchvision.models as models

import easyocr
from shapely.geometry import Polygon
from shapely import wkt as shapely_wkt

# -------------------------
# Utils
# -------------------------

def seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sort_vertices_ccw(coords):
    """Sort (x,y) vertices counter-clockwise around centroid to keep ordering stable."""
    if len(coords) == 0:
        return coords
    c = np.mean(coords, axis=0)
    ang = np.arctan2(coords[:,1] - c[1], coords[:,0] - c[0])
    order = np.argsort(ang)
    return coords[order]

def clamp01(a):
    return np.clip(a, 0.0, 1.0)

def to_wkt(coords_xy):
    """coords_xy: (N,2) with image pixel coordinates, ensures closed polygon."""
    if coords_xy is None or len(coords_xy) < 3:
        return None
    pts = coords_xy.tolist()
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    poly = Polygon(pts)
    if not poly.is_valid or poly.area <= 0:
        return None
    return poly.wkt  # POLYGON ((x y, ...))

def angle_cycle_align(pred, target):
    """Best cyclic shift alignment to reduce permutation loss for ordered polygons of same N."""
    n = pred.shape[0]
    best = None
    best_dist = 1e9
    for s in range(n):
        rolled = np.roll(pred, -s, axis=0)
        d = np.mean((rolled - target)**2)
        if d < best_dist:
            best_dist = d
            best = rolled
    return best

# -------------------------
# Dataset
# -------------------------

class CadastralDataset(Dataset):
    """
    Expects CSV with columns: ID, geometry (WKT or coord list).
    Images in data_dir as anonymised_{ID}.jpg
    Targets are vertex coords normalized by image (w,h) to [0,1].
    """
    def __init__(self, df, data_dir, num_points=8, transform=None, train=True):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.num_points = num_points
        self.transform = transform
        self.train = train

        # Light, robust train-time augs (rotation-safe for line drawings)
        self.aug = T.Compose([
            T.RandomApply([T.ColorJitter(brightness=0.15, contrast=0.15)], p=0.3),
            T.RandomApply([T.GaussianBlur(3)], p=0.2),
            T.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.95, 1.05))
        ]) if train else None

        self.base = T.Compose([
            T.ToTensor(),
            T.Resize((512, 512), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, f"anonymised_{row['ID']}.jpg")

        # load image RGB
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((512,512,3), np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.zeros((512,512,3), np.uint8)

        h, w = img.shape[:2]

        # parse polygon
        coords = self._parse_wkt(row.get('geometry', None))
        if len(coords) < 3:
            # fallback rectangle in image space
            coords = [(w*0.1, h*0.1), (w*0.9, h*0.1), (w*0.9, h*0.9), (w*0.1, h*0.9)]

        coords = np.array(coords, dtype=np.float32)
        # normalize to [0,1] using image size
        coords[:,0] = coords[:,0] / max(w, 1)
        coords[:,1] = coords[:,1] / max(h, 1)
        coords = clamp01(coords)

        # enforce fixed N by angle order + resample
        coords = sort_vertices_ccw(coords)
        coords = self._fix_num_points(coords, self.num_points)

        pil = Image.fromarray(img)
        if self.train and self.aug:
            pil = self.aug(pil)
        x = self.base(pil)  # 3x512x512 tensor

        y = torch.tensor(coords.reshape(-1), dtype=torch.float32)  # (2N,)
        meta = {'orig_w': w, 'orig_h': h, 'id': row['ID']}
        return x, y, meta

    def _parse_wkt(self, wkt_string):
        try:
            s = str(wkt_string)
            if not s or s.lower() == 'nan':
                return []
            s_clean = re.sub(r'POLYGON\s+Z\s*', 'POLYGON ', s, flags=re.I)
            s_clean = re.sub(r'([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)', r'\1 \2', s_clean)
            if 'POLYGON' in s_clean.upper():
                poly = shapely_wkt.loads(s_clean)
                coords = list(poly.exterior.coords)[:-1]
            else:
                # fallback coord extraction "(x, y)" style
                matches = re.findall(r'[-+]?\d*\.?\d+', s_clean)
                coords = []
                for i in range(0, len(matches), 2):
                    if i+1 < len(matches):
                        coords.append((float(matches[i]), float(matches[i+1])))
            return coords
        except Exception:
            return []

    def _fix_num_points(self, coords, n):
        """Evenly sample or pad to n vertices (keeps order)."""
        k = len(coords)
        if k == 0:
            return np.zeros((n,2), np.float32)
        if k == n:
            return coords
        if k > n:
            idx = np.linspace(0, k-1, n).astype(int)
            return coords[idx]
        # k < n: repeat last vertex
        out = [tuple(c) for c in coords]
        while len(out) < n:
            out.append(out[-1])
        return np.array(out, dtype=np.float32)

# -------------------------
# Model
# -------------------------

class CoordHead(nn.Module):
    """Small head that predicts 2N normalized coords from a backbone feature vector."""
    def __init__(self, in_dim, num_points):
        super().__init__()
        hidden = 1024
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden//2), nn.ReLU(inplace=True),
            nn.Linear(hidden//2, num_points*2)
        )

    def forward(self, x):
        return self.net(x)

class SOTACadastralNet(nn.Module):
    def __init__(self, num_points=8):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        layers = list(backbone.children())
        self.backbone = nn.Sequential(*layers[:-1])  # pool output (B,2048,1,1)
        self.head = CoordHead(2048, num_points)
        self.num_points = num_points

    def forward(self, x):
        f = self.backbone(x).flatten(1)  # (B,2048)
        out = self.head(f)               # (B, 2N)
        return out

# -------------------------
# Losses & Schedules
# -------------------------

class PolygonLoss(nn.Module):
    """
    SmoothL1 between predicted and target coords + small regularizers:
    - Perimeter smoothness (discourage ragged shapes)
    - CCW ordering (angle-sorted alignment via detach on target path)
    """
    def __init__(self, num_points, lambda_perm=0.01, lambda_center=0.005):
        super().__init__()
        self.num_points = num_points
        self.l1 = nn.SmoothL1Loss()
        self.lambda_perm = lambda_perm
        self.lambda_center = lambda_center

    def forward(self, pred, target):
        # base loss
        base = self.l1(pred, target)

        # reshape
        B, _ = pred.shape
        P = self.num_points
        pred_xy = pred.view(B, P, 2)
        # perimeter regularizer (in normalized space)
        roll = torch.roll(pred_xy, shifts=-1, dims=1)
        perim = torch.mean(torch.sqrt(torch.sum((pred_xy - roll)**2, dim=2) + 1e-8))

        # encourage vertices around center to avoid collapse
        c = torch.mean(pred_xy, dim=1, keepdim=True)
        spread = torch.mean(torch.sqrt(torch.sum((pred_xy - c)**2, dim=2) + 1e-8))

        return base + self.lambda_perm*perim - self.lambda_center*spread

def cosine_with_warmup(optimizer, warmup_steps, total_steps, min_lr=1e-6):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr, 0.5*(1.0 + math.cos(math.pi*progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone()

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply_to(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])

# -------------------------
# OCR
# -------------------------

class MetadataOCR:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])

        months = r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t\.?|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)'
        self.date_patterns = [
            re.compile(rf'(\d{{1,2}}[-/]\d{{1,2}}[-/]\d{{2,4}})'),
            re.compile(rf'({months}\s+\d{{1,2}},?\s+\d{{2,4}})', re.I)
        ]
        self.area_patterns = [
            re.compile(r'area[:\s]*([0-9,]+\.?\d*)\s*(m2|m²|sq\.?\s*m|square\s*met(?:er|re)s?|ha|hectare[s]?|acres?)', re.I),
            re.compile(r'([0-9,]+\.?\d*)\s*(m2|m²|sq\.?\s*m|square\s*met(?:er|re)s?|ha|hectare[s]?|acres?)', re.I),
        ]
        self.survey_patterns = [
            re.compile(r'survey\s+for\s+([^,\n]+)', re.I),
            re.compile(r'surveyed\s+for\s+([^,\n]+)', re.I),
            re.compile(r'client[:\s]+([^,\n]+)', re.I),
        ]
        self.lot_patterns = [
            re.compile(r'(?:lt|lot|land\s+title)\s*#?\s*([a-z0-9\-]+)', re.I)
        ]
        self.parishes = [p.lower() for p in [
            'Kingston','St. Andrew','St. Thomas','Portland','St. Mary','St. Ann',
            'Trelawny','St. James','Hanover','Westmoreland','St. Elizabeth',
            'Manchester','Clarendon','St. Catherine'
        ]]

    def extract(self, image_path):
        meta = {
            'TargetSurvey': 'unknown unknown unknown',
            'Certified date': 'Unknown',
            'Total Area': 0.0,
            'Unit of Measurement': 'sq m',
            'Parish': 'Unknown',
            'LT Num': 'Unknown'
        }
        if not os.path.exists(image_path):
            return meta

        img = cv2.imread(image_path)
        if img is None:
            return meta
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(2.0, (8,8))
        enhanced = clahe.apply(gray)

        results = self.reader.readtext(enhanced)
        text = ' '.join([t for (_, t, conf) in results if conf >= 0.45])
        lower = text.lower()

        # survey
        for pat in self.survey_patterns:
            m = pat.search(text)
            if m:
                meta['TargetSurvey'] = m.group(1).strip()
                break

        # date
        for pat in self.date_patterns:
            m = pat.search(text)
            if m:
                meta['Certified date'] = m.group(1)
                break

        # area + unit
        for pat in self.area_patterns:
            m = pat.search(text)
            if m:
                val = m.group(1).replace(',', '')
                unit = m.group(2).lower()
                try:
                    area = float(val)
                    meta['Total Area'] = area
                    if 'ha' in unit or 'hectare' in unit:
                        meta['Unit of Measurement'] = 'hectare'
                    elif 'acre' in unit:
                        meta['Unit of Measurement'] = 'acre'
                    else:
                        meta['Unit of Measurement'] = 'sq m'
                except:
                    pass
                break

        # parish
        for p in self.parishes:
            if p in lower:
                meta['Parish'] = p.title()
                break

        # LT
        for pat in self.lot_patterns:
            m = pat.search(text)
            if m:
                meta['LT Num'] = m.group(1).upper()
                break

        return meta

# -------------------------
# Trainer / Inference
# -------------------------

class SOTADeepLearningExtractor:
    def __init__(self, num_points=8, ema=True, seed=1234):
        seed_everything(seed)
        self.num_points = num_points
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SOTACadastralNet(num_points)
        self.model.to(self.device)
        self.ocr = MetadataOCR()
        self.ema = EMA(self.model, decay=0.999) if ema else None
        print(f"✅ Init on {self.device} | points={num_points} | EMA={'on' if ema else 'off'}")

        self.infer_base = T.Compose([
            T.ToTensor(),
            T.Resize((512,512), antialias=True),
            T.Normalize(mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225])
        ])

    def train(self, train_csv, data_dir, epochs=30, batch_size=8, lr=1e-4, val_split=0.1):
        df = pd.read_csv(train_csv)
        full = CadastralDataset(df, data_dir, num_points=self.num_points, train=True)
        n = len(full)
        n_val = max(1, int(n * val_split))
        n_train = n - n_val
        tr, va = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        criterion = PolygonLoss(self.num_points)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        total_steps = epochs * max(1, len(train_loader))
        scheduler = cosine_with_warmup(optimizer, warmup_steps=min(500, total_steps//20), total_steps=total_steps)
        scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type=='cuda'))

        best_val = 1e9
        self.model.train()
        step = 0
        for ep in range(1, epochs+1):
            self.model.train()
            ep_loss = 0.0
            for xb, yb, _ in train_loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(self.device.type=='cuda')):
                    pred = self.model(xb)
                    loss = criterion(pred, yb)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                if self.ema: self.ema.update(self.model)

                ep_loss += loss.item()
                step += 1

            val_loss = self._validate(val_loader, criterion)
            print(f"Epoch {ep:02d}/{epochs} | train_loss {ep_loss/len(train_loader):.5f} | val_loss {val_loss:.5f}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(self.model.state_dict(), 'sota_cadastral_model.pth')
                print(f"💾 Saved best model (val_loss={best_val:.5f})")

        print("🎯 Training complete.")

    @torch.no_grad()
    def _validate(self, loader, criterion):
        self.model.eval()
        total = 0.0
        cnt = 0
        for xb, yb, _ in loader:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(self.device.type=='cuda')):
                pred = self.model(xb)
                loss = criterion(pred, yb)
            total += loss.item()
            cnt += 1
        return total / max(1, cnt)

    def load_model(self, path='sota_cadastral_model.pth'):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            if self.ema:
                # Optionally, keep EMA for inference if you trained with it
                pass
            self.model.eval()
            print(f"✅ Loaded model from {path}")
            return True
        print(f"❌ Model not found: {path}")
        return False

    @torch.no_grad()
    def predict_polygon(self, image_path):
        if not os.path.exists(image_path): return None
        img = cv2.imread(image_path)
        if img is None: return None
        h, w = img.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # small TTA (identity + horizontal flip)
        imgs = [pil, pil.transpose(Image.FLIP_LEFT_RIGHT)]
        preds = []
        for i, im in enumerate(imgs):
            x = self.infer_base(im).unsqueeze(0).to(self.device)
            if self.ema:
                # temp apply EMA weights
                ema_model = SOTACadastralNet(self.num_points).to(self.device)
                ema_model.load_state_dict(self.model.state_dict())
                self.ema.apply_to(ema_model)
                ema_model.eval()
                out = ema_model(x)
            else:
                out = self.model(x)

            coords = out.squeeze(0).cpu().numpy().reshape(self.num_points, 2)

            if i == 1:
                # unflip x back: x' = 1 - x
                coords[:,0] = 1.0 - coords[:,0]

            preds.append(coords)

        coords = np.mean(np.stack(preds, axis=0), axis=0)
        coords = clamp01(coords)
        coords = sort_vertices_ccw(coords)
        # to absolute pixels
        coords[:,0] = coords[:,0] * w
        coords[:,1] = coords[:,1] * h

        # sanity clamp
        coords[:,0] = np.clip(coords[:,0], 0, w)
        coords[:,1] = np.clip(coords[:,1], 0, h)

        return [(float(x), float(y)) for x,y in coords]

    def extract_metadata(self, image_path):
        return self.ocr.extract(image_path)

    def process_test_data(self, test_csv, data_dir='data', out_csv='final_test_predictions.csv'):
        df = pd.read_csv(test_csv)
        out_rows = []
        ok = 0
        for i, row in df.iterrows():
            img_id = row['ID']
            path = os.path.join(data_dir, f"anonymised_{img_id}.jpg")
            meta = {
                'TargetSurvey': 'unknown unknown unknown',
                'Certified date': 'Unknown',
                'Total Area': 0.0,
                'Unit of Measurement': 'sq m',
                'Parish': 'Unknown',
                'LT Num': 'Unknown',
            }
            geom_wkt = None
            if os.path.exists(path):
                try:
                    poly = self.predict_polygon(path)
                    if poly and len(poly) >= 3:
                        wkt_str = to_wkt(np.array(poly, dtype=np.float32))
                        if wkt_str:
                            geom_wkt = wkt_str
                            ok += 1
                    meta.update(self.extract_metadata(path))
                except Exception as e:
                    print(f"[{img_id}] inference error: {e}")

            out_rows.append({
                'ID': img_id,
                **meta,
                'geometry': geom_wkt
            })

            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/{len(df)}")

        out_df = pd.DataFrame(out_rows, columns=[
            'ID','TargetSurvey','Certified date','Total Area',
            'Unit of Measurement','Parish','LT Num','geometry'
        ])
        out_df.to_csv(out_csv, index=False)
        rate = ok / max(1,len(out_df)) * 100
        print(f"✅ Saved to {out_csv} | valid polygons: {ok}/{len(out_df)} ({rate:.1f}%)")
        return out_df

# -------------------------
# Main
# -------------------------

def main():
    print("🚀 Upgraded Cadastral Extraction (single file)")
    extractor = SOTADeepLearningExtractor(num_points=8, ema=True)

    if not extractor.load_model('sota_cadastral_model.pth'):
        if os.path.exists('Train.csv'):
            print("📚 Training from Train.csv ...")
            extractor.train('Train.csv', data_dir='data', epochs=30, batch_size=8, lr=2e-4)
        else:
            print("❌ Train.csv not found. Cannot train.")
            return

    if os.path.exists('Test.csv'):
        extractor.process_test_data('Test.csv', data_dir='data', out_csv='final_test_predictions.csv')
        print("🎯 Done.")
    else:
        print("❌ Test.csv not found.")

if __name__ == "__main__":
    main()
