import copy
from typing import List, Dict, Tuple
import math

try:
    import torch
except ImportError:
    torch = None  # only needed if you use compression

# --------------------------
# Helper utilities (no deps)
# --------------------------
def _as_tensor(x):
    # works with torch.Tensor; if numpy/others appear, convert here
    if torch is None:
        raise RuntimeError("Compression requires torch; install PyTorch or set compress='none'.")
    if isinstance(x, torch.Tensor):
        return x
    # fall back: try to make a tensor
    return torch.tensor(x)

def _dense_bits(sd: Dict) -> int:
    bits = 0
    for v in sd.values():
        t = _as_tensor(v)
        bits += t.numel() * 32
    return bits

# --- Top-K ---
def _topk_compress(t: "torch.Tensor", k_frac: float = 0.01):
    flat = t.view(-1)
    k = max(1, int(k_frac * flat.numel()))
    if k >= flat.numel():
        # degenerate -> send dense
        idx = torch.arange(flat.numel(), device=flat.device)
        vals = flat
    else:
        _, idx = torch.topk(flat.abs(), k, sorted=False)
        vals = flat[idx]
    shape = torch.tensor(t.shape, device=t.device)
    return vals, idx, shape

def _topk_decompress(vals, idx, shape, device=None, dtype=None):
    N = int(torch.prod(shape))
    out = torch.zeros(N, device=device, dtype=dtype)
    out[idx] = vals
    return out.view(tuple(shape.tolist()))

def _topk_bits(vals, idx, total_len: int) -> int:
    # 32 bits per value + ⌈log2(total_len)⌉ bits per index
    if vals.numel() == 0:
        return 0
    logN = max(1, math.ceil(math.log2(max(1, total_len))))
    return int(vals.numel() * (32 + logN))

# --- QSGD-8b ---
def _qsgd_quantize(t: "torch.Tensor", levels: int = 256):
    flat = t.view(-1)
    tmin = flat.min().item()
    tmax = flat.max().item()
    if tmax == tmin:
        q = torch.zeros_like(flat, dtype=torch.uint8)
        return q, tmin, tmax, t.shape
    scale = (tmax - tmin) / (levels - 1)
    q = torch.clamp(((flat - tmin) / scale).round(), 0, levels - 1).to(torch.uint8)
    return q, tmin, tmax, t.shape

def _qsgd_dequantize(q: "torch.Tensor", tmin: float, tmax: float, shape):
    levels = 256
    scale = (tmax - tmin) / (levels - 1) if tmax != tmin else 1.0
    flat = q.to(torch.float32) * scale + tmin
    return flat.view(shape)

def _qsgd_bits(q: "torch.Tensor") -> int:
    # 8 bits/value + small header (~64 bits for min/max)
    return int(q.numel() * 8 + 64)

# --- Sign (1-bit + scale) ---
def _sign_compress(t: "torch.Tensor"):
    flat = t.view(-1)
    scale = float(flat.abs().mean()) if flat.numel() > 0 else 0.0
    # store as uint8 holding {0,1}; assume later bit-packing in practice
    signs = (flat >= 0).to(torch.uint8)
    return signs, scale, t.shape

def _sign_decompress(signs: "torch.Tensor", scale: float, shape):
    s = signs.to(torch.float32) * 2.0 - 1.0  # {0,1} -> {-1,+1}
    return (s * scale).view(shape)

def _sign_bits(signs: "torch.Tensor") -> int:
    # 1 bit/value + ~32-bit scale header
    return int(signs.numel() * 1 + 32)

# ----------------------------------------------------------
# Backward-compatible FedAvg with optional compression toggles
# ----------------------------------------------------------
def fedavg(
    list_of_state_dict_lists: List[List[Dict]],
    compress: str = "none",          # "none" | "topk" | "qsgd8" | "sign"
    topk_frac: float = 0.01,
    return_comm_stats: bool = False
):
    """
    Args
    ----
    list_of_state_dict_lists:
        [
          [sd_actor_0, sd_actor_1, ..., sd_actor_{U-1}]  # client 0
          [sd_actor_0, sd_actor_1, ..., sd_actor_{U-1}]  # client 1
          ...
        ]
    compress: str
        "none" (default, identical to your original), or "topk" | "qsgd8" | "sign".
    topk_frac: float
        Fraction for Top-K sparsification if compress == "topk".
    return_comm_stats: bool
        If True, also return a dict with {'dense_bits_total', 'comp_bits_total', 'ratio'}.

    Returns
    -------
    out : List[Dict]
        Averaged state_dict list for all U actors (same shape as input[0]).
    (optional) stats : Dict[str, float]
        Only if return_comm_stats=True.
    """
    U = len(list_of_state_dict_lists[0])
    num_clients = len(list_of_state_dict_lists)

    # fast path: no compression -> exactly your original logic
    if compress == "none":
        out = []
        for u in range(U):
            avg = copy.deepcopy(list_of_state_dict_lists[0][u])
            for k in avg.keys():
                for c in range(1, num_clients):
                    avg[k] += list_of_state_dict_lists[c][u][k]
                avg[k] /= num_clients
            out.append(avg)
        if return_comm_stats:
            dense_total = sum(_dense_bits(sd) for client in list_of_state_dict_lists for sd in client)
            stats = dict(dense_bits_total=dense_total, comp_bits_total=dense_total, ratio=1.0)
            return out, stats
        return out

    # compression paths
    if torch is None:
        raise RuntimeError("Compression requires torch; install PyTorch or use compress='none'.")

    out = []
    dense_bits_total = 0
    comp_bits_total = 0

    for u in range(U):
        # gather per-client parameter tensors for actor u
        # we will decompress each client's params, then average
        # (your optimizer/local updates stay unchanged)
        keys = list(list_of_state_dict_lists[0][u].keys())
        avg_sd = {}

        # precompute dense bits (for baseline)
        for c in range(num_clients):
            dense_bits_total += _dense_bits(list_of_state_dict_lists[c][u])

        for k in keys:
            # stack "reconstructed" tensors from each client for this param
            rec_list = []

            for c in range(num_clients):
                t = _as_tensor(list_of_state_dict_lists[c][u][k]).float()

                if compress == "topk":
                    vals, idx, shape = _topk_compress(t, topk_frac)
                    comp_bits_total += _topk_bits(vals, idx, t.numel())
                    rec = _topk_decompress(vals, idx, shape, device=t.device, dtype=t.dtype)

                elif compress == "qsgd8":
                    q, tmin, tmax, shape = _qsgd_quantize(t)
                    comp_bits_total += _qsgd_bits(q)
                    rec = _qsgd_dequantize(q, tmin, tmax, shape).to(t.dtype).to(t.device)

                elif compress == "sign":
                    signs, scale, shape = _sign_compress(t)
                    comp_bits_total += _sign_bits(signs)
                    rec = _sign_decompress(signs, scale, shape).to(t.dtype).to(t.device)

                else:
                    # fallback -> dense
                    rec = t.clone()

                rec_list.append(rec)

            avg_param = torch.stack(rec_list, dim=0).mean(dim=0)
            # keep original dtype/device of first client’s tensor for this key
            ref = list_of_state_dict_lists[0][u][k]
            if isinstance(ref, torch.Tensor):
                avg_param = avg_param.to(ref.dtype).to(ref.device)
            avg_sd[k] = avg_param

        out.append(avg_sd)

    if return_comm_stats:
        ratio = (comp_bits_total / dense_bits_total) if dense_bits_total > 0 else 1.0
        stats = dict(
            dense_bits_total=int(dense_bits_total),
            comp_bits_total=int(comp_bits_total),
            ratio=float(ratio)
        )
        return out, stats

    return out
