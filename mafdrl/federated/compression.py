# mafdrl/federated/compression.py
import torch
from torch import Tensor
from typing import Dict, Tuple

def _tensor_bits(t: Tensor) -> int:
    # approximate comm cost in bits for dense float32 tensor
    return t.numel() * 32

def dense_serialize(params: Dict[str, Tensor]) -> int:
    """Return baseline (uncompressed) bit count."""
    return sum(_tensor_bits(p) for p in params.values())

# ---------- Top-K sparsification ----------
def topk_compress(t: Tensor, k_frac: float = 0.01) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Return (values, indices, shape) for sparse Top-K of a flattened tensor.
    """
    flat = t.view(-1)
    k = max(1, int(k_frac * flat.numel()))
    vals, idx = torch.topk(flat.abs(), k, sorted=False)
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[idx] = True
    signed_vals = flat[mask]
    return signed_vals, idx, torch.tensor(t.shape, device=t.device)

def topk_decompress(vals: Tensor, idx: Tensor, shape: Tensor) -> Tensor:
    out = torch.zeros(int(torch.prod(shape)), device=vals.device, dtype=vals.dtype)
    out[idx] = vals
    return out.view(tuple(shape.tolist()))

def topk_bits(vals: Tensor, idx: Tensor) -> int:
    # values ~32 bits each, indices need ceil(log2(N)) per entry
    N = idx.numel() if idx.numel() > 0 else 1
    if N == 0: 
        return 0
    logN = max(1, int(torch.ceil(torch.log2(torch.tensor(int(vals.numel()) + 1.0)))))
    return int(vals.numel() * (32 + logN))

# ---------- QSGD 8-bit uniform quantization ----------
def qsgd_quantize(t: Tensor, levels: int = 256) -> Tuple[Tensor, float, float]:
    flat = t.view(-1)
    tmin, tmax = flat.min().item(), flat.max().item()
    if tmax == tmin:
        q = torch.zeros_like(flat, dtype=torch.uint8)
        return q, tmin, tmax
    scale = (tmax - tmin) / (levels - 1)
    q = torch.clamp(((flat - tmin) / scale).round(), 0, levels-1).to(torch.uint8)
    return q, tmin, tmax

def qsgd_dequantize(q: Tensor, tmin: float, tmax: float, shape) -> Tensor:
    levels = 256
    scale = (tmax - tmin) / (levels - 1) if tmax != tmin else 1.0
    flat = q.to(torch.float32) * scale + tmin
    return flat.view(shape)

def qsgd_bits(q: Tensor) -> int:
    return q.numel() * 8 + 64  # 8 bits per value + header (tmin,tmax)

# ---------- Sign compression (1-bit + scale) ----------
def sign_compress(t: Tensor) -> Tuple[Tensor, float, Tuple[int, ...]]:
    flat = t.view(-1)
    scale = flat.abs().mean().item() if flat.numel() > 0 else 0.0
    signs = (flat >= 0).to(torch.uint8)  # 1 bit per entry packed later
    return signs, scale, tuple(t.shape)

def sign_decompress(signs: Tensor, scale: float, shape: Tuple[int, ...]) -> Tensor:
    s = signs.to(torch.float32) * 2 - 1  # {0,1} -> {-1,+1}
    return (s * scale).view(shape)

def sign_bits(signs: Tensor) -> int:
    return signs.numel() * 1 + 32  # 1 bit/value + scale header
