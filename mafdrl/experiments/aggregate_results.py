# mafdrl/experiments/aggregate_results.py
import os, glob, math
import pandas as pd
import numpy as np

def load_seed_csvs():
    paths = sorted(glob.glob("eval_seed*/eval_metrics.csv"))
    if not paths:
        raise FileNotFoundError("No eval_seed*/eval_metrics.csv files found. Run multiseed/eval first.")
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["seed_dir"] = os.path.dirname(p)
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {p}: {e}")
    if not dfs:
        raise RuntimeError("No readable CSVs.")
    return pd.concat(dfs, ignore_index=True), paths

def num_cols(df):
    return [c for c in df.columns if df[c].dtype.kind in "fi" and c not in ("step","round")]

def agg_mean_std(df):
    cols = num_cols(df)
    g = df[cols].agg(['mean','std']).T.reset_index()
    g.columns = ["metric","mean","std"]
    # pretty columns
    g["pretty"] = g.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)
    return g

def to_latex_table(g):
    # pick some likely metrics if present
    candidates = [
        ("mean_reward", "Mean Reward"),
        ("mean_sinr_db", "Mean SINR (dB)"),
        ("mean_latency_ms", "Mean Latency (ms)"),
        ("mean_T_loc_ms", "Local Exec (ms)"),
        ("mean_T_tx_ms", "Uplink (ms)"),
        ("mean_T_Q_ms", "Queue (ms)"),
        ("mean_T_cpu_ms", "Edge CPU (ms)"),
        ("energy_per_step", "Energy / step"),
    ]
    rows = []
    for key, label in candidates:
        m = g[g.metric==key]
        if not m.empty:
            rows.append((label, m.iloc[0].pretty))
    if not rows:
        # fallback: use all
        for _, r in g.iterrows():
            rows.append((r.metric, r.pretty))
    # build simple LaTeX tabular
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Aggregate results across seeds (mean $\pm$ std).}",
        r"\label{tab:aggregate}",
        r"\begin{tabular}{l c}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
    ]
    for lab, val in rows:
        lines.append(f"{lab} & {val} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        ""
    ]
    return "\n".join(lines)

def main():
    df, paths = load_seed_csvs()
    print(f"Loaded {len(paths)} CSVs:", *paths, sep="\n  - ")
    g = agg_mean_std(df)
    os.makedirs("aggregates", exist_ok=True)
    g.to_csv("aggregates/aggregate_mean_std.csv", index=False)
    with open("aggregates/aggregate_table.tex","w") as f:
        f.write(to_latex_table(g))
    print("\nSaved:")
    print("  - aggregates/aggregate_mean_std.csv")
    print("  - aggregates/aggregate_table.tex")
    print("\nPreview (first few metrics):")
    print(g.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
