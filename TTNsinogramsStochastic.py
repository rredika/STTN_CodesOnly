import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.geometry.transform as K
import math

def _compute_patch_hyperparams(T, H, num_patches_t, num_patches_r,
                               patch_t, patch_r, overlap_t, overlap_r,
                               allow_padding):
    def one_axis(N, num_patches, patch, overlap):
        """Helper to compute stride, padding, and output size for one dimension."""
        if num_patches is not None:
            if N % num_patches == 0:
                patch = N // num_patches
                pad = 0
            else:
                if not allow_padding:
                    raise ValueError(f"N={N} not divisible by {num_patches}")
                patch = math.ceil(N / num_patches)
                total = num_patches * patch
                pad = (total - N) // 2
        else:
            if patch <= 0: 
                raise ValueError("patch must be > 0")
            pad = 0
            
        stride = max(1, patch - max(0, overlap))
        out = math.floor((N + 2*pad - patch) / stride) + 1
        return patch, stride, pad, out

    # Correct unpacking: 4 values from each axis call
    pt, st, pad_t, Tp = one_axis(T, num_patches_t, patch_t, overlap_t)
    pr, sr, pad_r, Hp = one_axis(H, num_patches_r, patch_r, overlap_r)
    
    return pt, pr, st, sr, pad_t, pad_r, Tp, Hp

class StochasticTraceWrapper(nn.Module):
    """
    Wraps the Sinogram CNN to perform GPU-accelerated Stochastic Projections.
    Ensures every training batch samples from the continuous SO(2) rotation group.
    """
    def __init__(self, base_model, n_transformations=36):
        super().__init__()
        self.base_model = base_model
        self.T = n_transformations

    def forward(self, x):
        # x: [B, 1, H, W]
        B, C, H, W = x.shape
        device = x.device
        
        # Monte Carlo sampling of rotation angles
        # angles = torch.rand(self.T, device=device) * 360.0
        # angles, _ = torch.sort(angles)
        # Check if model is training or evaluating
        if self.training:
            angles = torch.rand(self.T, device=device) * 360.0
        else:
            angles = torch.linspace(0, 360.0, steps=self.T + 1, device=device)[:-1]
            
        angles, _ = torch.sort(angles)

        stack_list = []
        for i in range(self.T):
            angle_tensor = torch.full((B,), angles[i], device=device)
            rotated = K.rotate(x, angle_tensor, padding_mode='zeros', align_corners=True)
            stack_list.append(rotated)

        x_5d = torch.stack(stack_list, dim=-1) # [B, 1, H, W, T]
        return self.base_model(x_5d)

class TTNSinogramStochastic(nn.Module):
    def __init__(self, input_size=28, n_transformations=36, in_channels=1, n_classes=10,
                 features=("sum_r","max_r","mean_r","std_r"),
                 patch_t=4, patch_r=7, num_patches_t=None, num_patches_r=None,
                 overlap_t=0, overlap_r=0, allow_padding=False,
                 embed_channels=64, cnn_channels=96, num_blocks=1, dropout=0.1,
                 scalar_softmax=True, use_patch_embed=True):
        super().__init__()
        self.features = list(features)
        self.cin = len(self.features)
        self.scalar_softmax = scalar_softmax
        self.use_patch_embed = use_patch_embed
        self.alpha = nn.Parameter(torch.ones(self.cin))

        if self.use_patch_embed:
            # Correctly receiving 8 values from the fixed hyperparams function
            pt, pr, st, sr, pad_t, pad_r, self.Tp, self.Hp = _compute_patch_hyperparams(
                T=n_transformations, H=input_size, num_patches_t=num_patches_t,
                num_patches_r=num_patches_r, patch_t=patch_t, patch_r=patch_r,
                overlap_t=overlap_t, overlap_r=overlap_r, allow_padding=allow_padding
            )
            self.patch_stem = nn.Sequential(
                nn.Conv2d(self.cin, embed_channels, kernel_size=(pt, pr), stride=(st, sr), padding=(pad_t, pad_r), bias=False),
                nn.BatchNorm2d(embed_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.direct_stem = nn.Sequential(
                nn.Conv2d(self.cin, embed_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(embed_channels),
                nn.ReLU(inplace=True)
            )

        blocks = []
        in_ch = embed_channels
        for _ in range(num_blocks):
            # Using GroupNorm(8, ...) as per your original methodology
            blocks += [
                nn.Conv2d(in_ch, cnn_channels, 3, padding=1, bias=False),
                nn.GroupNorm(8, cnn_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv2d(cnn_channels, cnn_channels, 3, padding=1, bias=False),
                nn.GroupNorm(8, cnn_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ]
            in_ch = cnn_channels
        self.cnn = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(cnn_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        # x: [B, 1, H, W, T]
        B, C, H, W, T = x.shape
        xp = x.permute(0, 4, 1, 2, 3) # [B, T, C, H, W]
        
        # Trace transform functional extraction
        f_list = []
        if "sum_r" in self.features: f_list.append(xp.sum(dim=-1).mean(dim=2))
        if "max_r" in self.features: f_list.append(xp.amax(dim=-1).mean(dim=2))
        if "mean_r" in self.features: f_list.append(xp.mean(dim=-1).mean(dim=2))
        if "std_r" in self.features: f_list.append(xp.std(dim=-1, unbiased=False).mean(dim=2))
        
        # Build sinogram stack [B, Cin, T, H]
        sinogram = torch.stack(f_list, dim=1)
        
        # Normalization
        m = sinogram.mean(dim=(2,3), keepdim=True)
        s = sinogram.std(dim=(2,3), keepdim=True).clamp_min(1e-5)
        sinogram = (sinogram - m) / s
        
        # Learnable attention over features
        w = torch.softmax(self.alpha, dim=0) if self.scalar_softmax else self.alpha
        sinogram = sinogram * w.view(1, -1, 1, 1)

        z = self.patch_stem(sinogram) if self.use_patch_embed else self.direct_stem(sinogram)
        z = self.cnn(z)
        return self.head(z)