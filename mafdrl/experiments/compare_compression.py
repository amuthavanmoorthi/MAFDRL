# mafdrl/experiments/compare_compression.py
import argparse, subprocess, sys, os, glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# === IEEE-style formatting (Times New Roman w/ safe fallbacks + tight margins) ===
mpl.rcParams['font.family']  = 'serif'
mpl.rcParams['font.serif']   = ['Times New Roman', 'Times', 'Nimbus Roman No9 L', 'DejaVu Serif']
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 11
mpl.rcParams['ytick.labelsize'] = 11
mpl.rcParams['savefig.bbox']  = 'tight'   # trim outside whitespace
mpl.rcParams['savefig.pad_inches'] = 0
mpl.rcParams['figure.autolayout'] = True
plt.rcParams["pdf.fonttype"] = 42         # keep text editable in PDF
plt.rcParams["ps.fonttype"]  = 42


def run(mode: str, extra: str = "") -> int:
    """Launch a short training experiment to populate runs/ with TB logs."""
    cmd = [
        sys.executable, "-m", "mafdrl.experiments.long_train",
        "--rounds", "40", "--local-iters", "400",
        "--seeds", "2",
        "--compress", mode
    ]
    if extra:
        cmd += extra.split()
    print(">>", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def _load_scalar(logdir: str, tag: str) -> pd.DataFrame | None:
    """Read a scalar from a TensorBoard run directory into a DataFrame(step, value)."""
    try:
        ea = EventAccumulator(logdir)
        ea.Reload()
        if tag not in ea.Tags().get("scalars", []):
            return None
        sc = ea.Scalars(tag)
        return pd.DataFrame([(s.step, s.value) for s in sc], columns=["step", "value"])
    except Exception:
        return None


def collect_runs(root: str = "runs"):
    """
    Gather reward and comm stats for each compression mode from TB logs.
    Returns a list of dicts: {mode, steps, reward, mb}
    """
    rows = []
    modes = ["none", "topk", "qsgd8", "sign"]
    for mode in modes:
        # find all subdirs that contain this mode in their name
        for d in glob.glob(os.path.join(root, f"*{mode}*")):
            rwd = _load_scalar(d, "train_long/round_mean_reward")
            comp_bits = _load_scalar(d, "comm/comp_bits_total")
            if rwd is None:
                continue
            steps = rwd["step"].to_numpy()
            rewards = rwd["value"].to_numpy()
            # Convert bits to MB (if comm scalars exist); otherwise NaN
            if comp_bits is not None:
                mb = (comp_bits["value"].to_numpy() / 8.0) / (1024.0 ** 2)
                n = min(len(steps), len(mb))  # align lengths if needed
                steps, rewards, mb = steps[:n], rewards[:n], mb[:n]
            else:
                mb = np.full_like(steps, np.nan, dtype=float)
            rows.append(dict(mode=mode, steps=steps, reward=rewards, mb=mb, dir=d))
    return rows


def smooth(arr: np.ndarray, w: int = 5) -> np.ndarray:
    if arr.size < w or w <= 1:
        return arr
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="same")


def plot_lines(rows, ykey: str, xlabel: str, ylabel: str, outfile: str, smooth_w: int = 5):
    """Generic line plot helper with IEEE styling and zero inner/outer whitespace."""
    if not rows:
        print(f"[warn] nothing to plot for {outfile} (no runs found)")
        return

    plt.figure(figsize=(6, 4))
    xmax = 1
    ymin, ymax = float("inf"), -float("inf")

    for r in rows:
        x = r["steps"]
        y = r[ykey]
        if np.all(np.isnan(y)):
            continue
        y_plot = smooth(y, smooth_w)
        plt.plot(x, y_plot, marker="o", markersize=2.5, linewidth=1.0, label=r["mode"])
        if len(x):
            xmax = max(xmax, int(np.max(x)))
        if np.isfinite(np.nanmin(y_plot)) and np.isfinite(np.nanmax(y_plot)):
            ymin = min(ymin, float(np.nanmin(y_plot)))
            ymax = max(ymax, float(np.nanmax(y_plot)))

    ax = plt.gca()
    # remove inner padding and clamp to data range
    ax.margins(x=0, y=0)
    plt.xlim(1, xmax)
    if ymin < ymax and np.isfinite(ymin) and np.isfinite(ymax):
        rng = ymax - ymin
        pad = 0.01 * rng  # tiny pad to avoid marker clipping
        plt.ylim(ymin - pad, ymax + pad)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(frameon=False, ncol=1)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"[ok] saved {outfile}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["none", "topk", "qsgd8", "sign"])
    parser.add_argument("--topk-frac", type=float, default=0.01)
    parser.add_argument("--skip-run", action="store_true",
                        help="If set, do not launch training; just parse existing runs/")
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--local-iters", type=int, default=400)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--runs-root", type=str, default="runs")
    parser.add_argument("--outdir", type=str, default="figures")
    parser.add_argument("--smooth", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.environ["MAFDRL_SAVE_TAG"] = "cmp"  # so runs/ tags are grouped

    if not args.skip_run:
        # Run each selected mode
        for m in args.modes:
            extra = ""
            if m == "topk":
                extra = f"--topk-frac {args.topk_frac}"
            # propagate rounds/iters/seeds so CLI fully controls experiment size
            extra = f"{extra} --rounds {args.rounds} --local-iters {args.local_iters} --seeds {args.seeds}"
            rc = run(m, extra)
            if rc != 0:
                print(f"[warn] run for mode={m} exited with code {rc}")

    # Collect and plot
    rows = collect_runs(args.runs_root)

    # Reward vs. round
    plot_lines(
        rows, ykey="reward",
        xlabel="Federated round",
        ylabel="Mean reward",
        outfile=os.path.join(args.outdir, "compress_reward.png"),
        smooth_w=args.smooth
    )

    # MB sent per round (global sum)
    plot_lines(
        rows, ykey="mb",
        xlabel="Federated round",
        ylabel="MB sent per round (global)",
        outfile=os.path.join(args.outdir, "compress_comm.png"),
        smooth_w=args.smooth
    )


if __name__ == "__main__":
    main()
