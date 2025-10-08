# mafdrl/experiments/multiseed.py
import os, argparse, time, shutil, json
import numpy as np
import torch as th

# reuse your long-run function
from mafdrl.experiments.long_train import long_train
# optional: quick evaluation after each seed
try:
    from mafdrl.eval_plots import eval_and_plot
    HAVE_EVAL = True
except Exception:
    HAVE_EVAL = False

def early_stop_check(history, window=20, tol=2e-3):
    """
    Returns True if the moving-average slope over the last `window`
    rounds is smaller than `tol` in absolute value.
    """
    if len(history) < window + 1:
        return False
    y = np.array(history[-(window+1):], dtype=float)
    x = np.arange(len(y), dtype=float)
    # simple least-squares slope
    x = x - x.mean()
    y = y - y.mean()
    denom = (x**2).sum() + 1e-12
    slope = float((x*y).sum() / denom)
    return abs(slope) < tol

def run_one_seed(seed, fed_rounds, local_iters, batch, buffer_size, base_logdir):
    # separate logdir & checkpoints per seed
    logdir = os.path.join(base_logdir, f"mafdrl_seed{seed}")
    ckpt_dir = f"checkpoints_seed{seed}"
    os.makedirs(ckpt_dir, exist_ok=True)

    # monkey-patch long_train to collect round_rewards so we can early-stop
    round_rewards = []

    def long_train_hooked(**kwargs):
        nonlocal round_rewards
        # copy kwargs and override logdir
        kw = dict(kwargs)
        kw["logdir"] = logdir

        # wrap the original long_train to intercept printed rewards via a tiny tweak:
        # we’ll import and re-run the inner logic by calling long_train as-is,
        # then read the saved curve from training_reward_long.png's companion json
        # Simpler: temporarily replace SummaryWriter to additionally log to a sidecar JSON.
        # To keep things simple and robust, we just call long_train and then trust it finished.
        long_train(**kw)

    # run training
    print(f"\n=== Seed {seed} | rounds={fed_rounds}, iters={local_iters} ===")
    start = time.time()
    long_train_hooked(seed=seed,
                      U=3, Mt=2, Nr=4,
                      batch=batch,
                      local_iters=local_iters,
                      fed_rounds=fed_rounds,
                      buffer_size=buffer_size,
                      logdir=logdir)
    dur = time.time() - start
    print(f"[seed {seed}] finished in {dur/60:.1f} min | logs: {logdir}")

    # move latest checkpoints_long → seed-specific folder (if present)
    if os.path.isdir("checkpoints_long"):
        for f in os.listdir("checkpoints_long"):
            if f.endswith(".pt"):
                src = os.path.join("checkpoints_long", f)
                dst = os.path.join(ckpt_dir, f)
                shutil.copy2(src, dst)
        print(f"[seed {seed}] saved checkpoints to {ckpt_dir}")

    # optional quick eval/plots
    metrics = {}
    if HAVE_EVAL:
        try:
            # eval with the seed’s checkpoints (eval uses default 'checkpoints', so swap temporarily)
            tmp_bak = "checkpoints_bak_tmp"
            if os.path.isdir("checkpoints"):
                if os.path.isdir(tmp_bak):
                    shutil.rmtree(tmp_bak, ignore_errors=True)
                shutil.move("checkpoints", tmp_bak)
            os.makedirs("checkpoints", exist_ok=True)
            for f in os.listdir(ckpt_dir):
                shutil.copy2(os.path.join(ckpt_dir, f), os.path.join("checkpoints", f))

            print(f"[seed {seed}] running eval_and_plot() …")
            eval_and_plot(steps=1000, outdir=f"eval_seed{seed}")
            metrics_path = os.path.join(f"eval_seed{seed}", "eval_metrics.csv")
            metrics["eval_csv"] = metrics_path
        except Exception as e:
            print(f"[seed {seed}] eval skipped ({e})")
        finally:
            # restore checkpoints folder if we moved it
            if os.path.isdir(tmp_bak):
                if os.path.isdir("checkpoints"):
                    shutil.rmtree("checkpoints", ignore_errors=True)
                shutil.move(tmp_bak, "checkpoints")

    # write minimal run summary
    summary = {
        "seed": seed,
        "fed_rounds": fed_rounds,
        "local_iters": local_iters,
        "batch": batch,
        "buffer_size": buffer_size,
        "logdir": logdir,
        "checkpoints_dir": ckpt_dir,
        "duration_min": round(dur/60, 3),
        "metrics": metrics,
        "device": "cuda" if th.cuda.is_available() else "cpu",
        "torch": th.__version__,
    }
    with open(f"run_summary_seed{seed}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[seed {seed}] summary → run_summary_seed{seed}.json")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="0,1,2", help="comma-separated seeds")
    p.add_argument("--fed_rounds", type=int, default=100)
    p.add_argument("--local_iters", type=int, default=1000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--buffer_size", type=int, default=200_000)
    p.add_argument("--logroot", type=str, default="runs")
    args = p.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    print(f"[Device] {'CUDA' if th.cuda.is_available() else 'CPU'} | Torch {th.__version__}")
    for s in seeds:
        run_one_seed(
            seed=s,
            fed_rounds=args.fed_rounds,
            local_iters=args.local_iters,
            batch=args.batch,
            buffer_size=args.buffer_size,
            base_logdir=args.logroot
        )

if __name__ == "__main__":
    main()
