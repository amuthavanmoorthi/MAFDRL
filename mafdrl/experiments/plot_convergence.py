# mafdrl/experiments/plot_convergence.py
import argparse, os, glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# === Global IEEE-style plotting setup (Times New Roman + tight margins) ===
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Nimbus Roman No9 L', 'DejaVu Serif']  # fallback chain
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 11
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'   # remove outer whitespace
mpl.rcParams['savefig.pad_inches'] = 0
mpl.rcParams['figure.autolayout'] = True
plt.rcParams["pdf.fonttype"] = 42        # keep text editable in PDF
plt.rcParams["ps.fonttype"] = 42

def moving_avg(x, w):
    if w <= 1:
        return x
    s = pd.Series(x, dtype=float).rolling(window=w, min_periods=1, center=False).mean().values
    return s

def find_event_dirs(logroot: str):
    """Return subdirs that contain TensorBoard event files."""
    dirs = set()
    for path in glob.glob(os.path.join(logroot, "**", "events.out.tfevents.*"), recursive=True):
        dirs.add(os.path.dirname(path))
    return sorted(dirs)

def read_tag(series_dir: str, tag: str):
    """Read scalar series `tag` from a TensorBoard event dir."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(series_dir, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None
    sc = ea.Scalars(tag)
    steps = np.array([e.step for e in sc], dtype=int)
    vals  = np.array([e.value for e in sc], dtype=float)
    return steps, vals

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logroot", default="runs", help="Root folder containing TensorBoard runs")
    ap.add_argument("--tag", default="train/round_mean_reward", help="Scalar tag to extract")
    ap.add_argument("--smooth", type=int, default=9, help="Moving average window (rounds)")
    ap.add_argument("--out", default="figures/convergence_plot.png", help="Output figure path")
    ap.add_argument("--csv", default="figures/convergence_plot.csv", help="Also save CSV")
    ap.add_argument("--ylim", type=float, nargs=2, default=None, help="y-limits, e.g., --ylim 1.7 2.1")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)

    # 1) collect all event dirs
    dirs = find_event_dirs(args.logroot)
    if not dirs:
        raise SystemExit(f"No TensorBoard event files found under: {args.logroot}")

    # 2) load the requested tag for each dir (seed/run)
    runs = []
    for d in dirs:
        r = read_tag(d, args.tag)
        if r is None:
            continue
        steps, vals = r
        runs.append((os.path.basename(d), steps, vals))

    if not runs:
        raise SystemExit(f"Tag '{args.tag}' not found in any runs under: {args.logroot}")

    # Normalize to common x-axis (round indices start at 1)
    max_round = max(int(steps.max()) for _, steps, _ in runs)
    xs = np.arange(1, max_round + 1, dtype=int)

    # Build matrix [n_runs, n_rounds] with NaNs, then fill by step index
    mat = np.full((len(runs), len(xs)), np.nan, dtype=float)
    names = []
    for i, (name, steps, vals) in enumerate(runs):
        names.append(name)
        # If steps start at 0, shift to start at 1
        shift = 1 - int(steps.min())
        for s, v in zip(steps, vals):
            idx = int(s + shift) - 1
            if 0 <= idx < len(xs):
                mat[i, idx] = v

    # Smoothing per run
    for i in range(mat.shape[0]):
        row = mat[i]
        ok = ~np.isnan(row)
        row[ok] = moving_avg(row[ok], args.smooth)

    # Aggregate across runs (ignore NaNs)
    mean = np.nanmean(mat, axis=0)
    std  = np.nanstd(mat, axis=0)
    n    = np.sum(~np.isnan(mat), axis=0)
    se   = std / np.sqrt(np.maximum(n, 1))
    ci95_low = mean - 1.96 * se
    ci95_hi  = mean + 1.96 * se

    # 3) Plot (no titles per your professor’s ask)
    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=200)

    # light lines for individual runs
    for i in range(mat.shape[0]):
        ax.plot(xs, mat[i], lw=1.0, alpha=0.35, color="gray")

    # mean + CI
    ax.plot(xs, mean, lw=2.0, color="black", label="MA-FDRL (mean)")
    ax.fill_between(xs, ci95_low, ci95_hi, color="gray", alpha=0.2, label="95% CI")

    ax.set_xlabel("Federated round")
    ax.set_ylabel("Mean reward")
    if args.ylim is not None:
        ax.set_ylim(args.ylim)

    # Remove inner whitespace and clamp axes tightly to data
    ax.margins(x=0, y=0)
    ax.set_xlim(xs.min(), xs.max())
    if args.ylim is None:
        ymin = np.nanmin(ci95_low)
        ymax = np.nanmax(ci95_hi)
        if np.isfinite(ymin) and np.isfinite(ymax) and ymin < ymax:
            rng = ymax - ymin
            pad = 0.01 * rng  # tiny pad to avoid clipping markers
            ax.set_ylim(ymin - pad, ymax + pad)

    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(frameon=False, fontsize=10)

    fig.savefig(args.out, dpi=300)  # bbox + pad already set in rcParams
    print(f"Saved plot -> {args.out}")

    # 4) CSV
    df = pd.DataFrame({
        "round": xs,
        "mean_reward": mean,
        "std_reward": std,
        "n_runs": n,
        "ci95_low": ci95_low,
        "ci95_hi": ci95_hi,
    })
    df.to_csv(args.csv, index=False)
    print(f"Saved CSV  -> {args.csv}")

if __name__ == "__main__":
    main()
