# show_ckpt_keys.py
# Inspect the structure of a PyTorch checkpoint so we know how to load it.

import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python show_ckpt_keys.py <checkpoint_path>")
        return

    ckpt_path = sys.argv[1]
    if not os.path.isfile(ckpt_path):
        print(f"[ERR] File not found: {ckpt_path}")
        return

    try:
        import torch
    except ImportError:
        print("[ERR] torch is not installed in this Python environment.")
        return

    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    print(f"\nType of checkpoint object: {type(ckpt)}")

    if isinstance(ckpt, dict):
        keys = list(ckpt.keys())
        print(f"\nTop-level keys ({len(keys)}):")
        for k in keys:
            v = ckpt[k]
            if isinstance(v, dict):
                print(f"  - {k}: dict with {len(v)} keys")
            else:
                print(f"  - {k}: {type(v)}")

        # If there's a state_dict key, drill one level deeper
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
            print(f"\nstate_dict keys ({len(sd)}):")
            for k in list(sd.keys())[:30]:  # show first 30 to avoid spam
                print(f"  - {k}")
    else:
        print("\n[WARN] Checkpoint is not a dict; printing repr:")
        print(repr(ckpt))

if __name__ == "__main__":
    main()
