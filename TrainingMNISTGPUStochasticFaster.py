import os
import pandas as pd
import time
import numpy as np
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import urllib.request
import zipfile
import os
import ssl
import kornia as K

# Import the model from your SinogramsStochasticMNIST.py
import TTNsinogramsStochastic as sinograms

ssl._create_default_https_context = ssl._create_unverified_context
url = "http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip"
zip_path = "mnist_rotation.zip"
print("Downloading dataset...")
urllib.request.urlretrieve(url, zip_path)
print("Download complete!")

# Extract files
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("rotated_mnist")

# -----------------------------
# .amat Dataset Loader
# -----------------------------
class AmatDataset(Dataset):
    def __init__(self, file_path: str, img_size: int = 28):
        """
        Loads .amat files where each row is [pixels... , label]
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find {file_path}")
        
        print(f"Loading {file_path}...")
        # Loading with numpy (skipping potential headers if they exist, 
        # though usually .amat files are pure space-separated text)
        data = np.loadtxt(file_path)
        
        # Split features and labels
        # Standard Rotated MNIST: 784 pixels, 1 label = 785 columns
        self.x = data[:, :-1].astype(np.float32)
        self.y = data[:, -1].astype(np.int64)
        
        self.img_size = img_size
        print(f"Loaded {len(self.y)} samples.")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Reshape to [1, H, W]
        img = torch.from_numpy(self.x[idx]).view(1, self.img_size, self.img_size)
        label = self.y[idx]
        return img, label

# -----------------------------
# GPU rotation bank (optimized for T=360)
# -----------------------------
def make_rotation_bank_gpu(x0: torch.Tensor, angles_deg: torch.Tensor) -> torch.Tensor:
    """
    x0: [B,1,28,28] on GPU
    angles_deg: [360] on GPU
    returns: [B,1,28,28,360] on GPU
    """
    B, C, H, W = x0.shape
    T = angles_deg.numel()

    # Expand batch across T
    x_rep = x0.unsqueeze(1).expand(B, T, C, H, W).reshape(B * T, C, H, W)
    angles_rep = angles_deg.unsqueeze(0).expand(B, T).reshape(B * T)

    x_rot = K.geometry.transform.rotate(
        x_rep,
        angles_rep,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

    return x_rot.view(B, T, C, H, W).permute(0, 2, 3, 4, 1).contiguous()

# -----------------------------
# Config
# -----------------------------
@dataclass
class TrainConfig:
    # Update paths to your .amat files
    train_path: str = 'rotated_mnist/mnist_all_rotation_normalized_float_train_valid.amat'
    test_path: str = 'rotated_mnist/mnist_all_rotation_normalized_float_test.amat'
    out_dir: str = "./results_Stochastic"
    filename_base = 'result_Mnist'

    img_size: int = 28
    n_transformations: int = 180  # As requested
    in_channels: int = 1

    # Features strictly restricted to requested list
    features: Tuple[str, ...] = ("sum_r", "max_r", "mean_r", "std_r")

    # Model params (Adjusted for 28x28)
    use_patch_embed: bool = True
    patch_t: int = 4  # Larger T patches since T=360
    patch_r: int = 7
    embed_channels: int = 64
    cnn_channels: int = 96
    num_blocks: int = 2
    dropout: float = 0.1
    scalar_softmax: bool = True

    # Training
    epochs: int = 250
    batch_size: int = 64  # Reduced batch size because T=360 uses significant GPU VRAM
    lr: float = 5e-4
    weight_decay: float = 1e-4
    use_amp: bool = True

    num_workers: int = 0
    device: str = "cuda"
    seed: int = 42
    log_every: int = 50

def set_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

# -----------------------------
# Train / Eval Loops
# -----------------------------
@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    model.eval()
    crit = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    amp_enabled = (use_amp and device.type == "cuda")

    for x0, y in loader:
        x0, y = x0.to(device), y.to(device)
       
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(x0) 
            loss = crit(logits, y)

        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
        loss_sum += loss.item() * y.size(0)

    return loss_sum / total, correct / total

def train_one_epoch(model, loader, optimizer, scaler, device, use_amp, log_every):
    model.train()
    crit = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    
    t0 = time.time()
    for step, (x0, y) in enumerate(loader, start=1):
        x0, y = x0.to(device), y.to(device)
        #x_bank = make_rotation_bank_gpu(x0, angles)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x0)
            loss = crit(logits, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
        loss_sum += loss.item() * y.size(0)

        if step % log_every == 0:
            dt = time.time() - t0
            print(f"  step {step:4d}/{len(loader)} | loss {loss_sum/total:.4f} | acc {correct/total:.4f} | {y.size(0)*log_every/dt:.1f} img/s")
            t0 = time.time()

    return loss_sum / total, correct / total

# -----------------------------
# Main
# -----------------------------
def main(cfg: TrainConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Custom Amat Dataloader
    train_ds = AmatDataset(cfg.train_path, img_size=cfg.img_size)
    test_ds = AmatDataset(cfg.test_path, img_size=cfg.img_size)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # Initialize SinogramStochasticCIFAR
    base_model = sinograms.TTNSinogramStochastic(
        input_size=cfg.img_size,
        n_transformations=cfg.n_transformations,
        in_channels=cfg.in_channels,
        n_classes=10,
        features=cfg.features,
        patch_t=cfg.patch_t,
        patch_r=cfg.patch_r,
        embed_channels=cfg.embed_channels,
        cnn_channels=cfg.cnn_channels,
        num_blocks=cfg.num_blocks,
        dropout=cfg.dropout,
        scalar_softmax=cfg.scalar_softmax,
        use_patch_embed=cfg.use_patch_embed,
    )
    
    model = sinograms.StochasticTraceWrapper(
        base_model, 
        n_transformations=cfg.n_transformations).to(device)

    # --- Parameter Count Calculation ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("-" * 30)
    print(f"Model Summary:")
    print(f"  Total Parameters:     {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print("-" * 30)
    # -----------------------------------

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # ... rest of the code remains the same

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    # Only enable AMP if CUDA is actually available
    amp_enabled = cfg.use_amp and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # Dataframes to hold results (matches your previous style)
    train_stats = pd.DataFrame(columns=['epoch', 'loss', 'acc']).astype({
        'epoch': int, 
        'loss': float, 
        'acc': float
    })
    test_stats = pd.DataFrame(columns=['epoch', 'loss', 'acc']).astype({
        'epoch': int, 
        'loss': float, 
        'acc': float
    })

    # 360 degree rotation bank
    #angles = torch.linspace(0, 360, steps=cfg.n_transformations + 1, device=device)[:-1]
    total_start_time = time.time()
    best_acc = 0.0
    best_epoch = 0
    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg.use_amp, cfg.log_every)
        val_loss, val_acc = evaluate(model, val_loader, device, cfg.use_amp)

        # Store Training Stats
        new_train_row = pd.DataFrame([[epoch, tr_loss, tr_acc]], columns=['epoch', 'loss', 'acc'])
        if not new_train_row.dropna(how='all').empty:
            train_stats = pd.concat([train_stats, new_train_row], ignore_index=True)

        # Store Testing Stats
        new_test_row = pd.DataFrame([[epoch, val_loss, val_acc]], columns=['epoch', 'loss', 'acc'])
        test_stats = pd.concat([test_stats, new_test_row], ignore_index=True)

        scheduler.step()

        print(f"Result -> Train Acc: {tr_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(cfg.out_dir, "best_mnist.pt"))
            print(f"*** New Best Model at Epoch {epoch} ***")
    

    # --- End Total Training Timer ---
    total_end_time = time.time()
    total_duration_mins = (total_end_time - total_start_time) / 60

    print("\n" + "="*30)
    print("TRAINING COMPLETE")
    print(f"Total Training Time: {total_duration_mins:.2f} minutes")
    print(f"Best Val Acc: {best_acc:.4f} (at Epoch {best_epoch})")
    print("="*30)

    # --- Save CSV Results ---
    train_stats.to_csv(os.path.join(cfg.out_dir, cfg.filename_base + '_train.csv'), index=False)
    test_stats.to_csv(os.path.join(cfg.out_dir, cfg.filename_base + '_test.csv'), index=False)

    print(f"\nTraining Complete. Best Val Acc: {best_acc:.4f}")

if __name__ == "__main__":
    main(TrainConfig())