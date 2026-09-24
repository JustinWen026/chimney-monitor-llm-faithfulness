"""
Fig. 5 — (a) narrative mention rate, (b) injected-noise groups cited per report.

All numbers are computed from the per-run scoring output; nothing is typed in by hand.
Error bars are percentile bootstrap 95% CIs of the mean (5,000 resamples, Section 5.5),
so they can never exceed 100% or fall below 0.

Usage:
    python fig5_narrative_noise.py --csv per_run_scores.csv
    python fig5_narrative_noise.py --csv per_run_scores.csv --no-check   # skip Table 5/6 check

Expected CSV columns (rename COL_* below if yours differ):
    scale         "single" or "concurrent"
    condition     "Baseline", "Tool-only" or "Proposed"
    mention_rate  0-100; runs that produced no report are scored 0 (as in Table 6)
    noise_groups  number of Table 3 noise groups cited in the narrative
    has_report    True/False (only matters for the 3 Proposed concurrent runs)
"""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# >>> 改成你的逐次評分 CSV 路徑 <<<
DEFAULT_CSV = "per_run_scores.csv"

COL_SCALE, COL_COND = "scale", "condition"
COL_RATE, COL_NOISE, COL_REPORT = "mention_rate", "noise_groups", "has_report"

SCALES = ["single", "concurrent"]
CONDS = ["Baseline", "Tool-only", "Proposed"]
COLORS = {"Baseline": "#B2B0A6", "Tool-only": "#5EC6A4", "Proposed": "#1A65A8"}
N_BOOT = 5000
SEED = 20260924

# Values printed in Tables 5 and 6: (mention rate %, noise groups, N).
# The script refuses to draw if the data do not reproduce them.
EXPECTED = {
    ("single", "Baseline"): (95.0, 1.26, 80),
    ("single", "Tool-only"): (98.8, 1.43, 80),
    ("single", "Proposed"): (99.6, 1.75, 80),
    ("concurrent", "Baseline"): (96.9, 1.01, 80),
    ("concurrent", "Tool-only"): (99.0, 0.03, 80),
    ("concurrent", "Proposed"): (94.9, 0.01, 79),
}
EXPECTED_PROPOSED_WITH_REPORT = (98.7, 76)  # Table 6, "excluding runs without a report"

# Holm-adjusted Welch p-values from Tables 5 and 6. Only significant pairs are drawn.
BRACKETS_A = [("single", "Baseline", "Tool-only", "p = .030"),
              ("single", "Baseline", "Proposed", "p = .005")]
BRACKETS_B = [("single", "Baseline", "Proposed", "p = .006"),
              ("concurrent", "Baseline", "Tool-only", "p < .001"),
              ("concurrent", "Baseline", "Proposed", "p < .001")]


def boot_ci(values, rng):
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def summarize(df, col, rng):
    out = {}
    for s in SCALES:
        for c in CONDS:
            v = df.loc[(df[COL_SCALE] == s) & (df[COL_COND] == c), col].to_numpy(float)
            lo, hi = boot_ci(v, rng)
            out[(s, c)] = dict(mean=v.mean(), lo=lo, hi=hi, n=len(v))
    return out


def check_against_tables(rate, noise, df):
    problems = []
    for key, (exp_rate, exp_noise, exp_n) in EXPECTED.items():
        r, z = rate[key], noise[key]
        if r["n"] != exp_n:
            problems.append(f"{key}: N = {r['n']}, table says {exp_n}")
        if round(r["mean"], 1) != exp_rate:
            problems.append(f"{key}: mention rate {r['mean']:.2f}, table says {exp_rate}")
        if round(z["mean"], 2) != exp_noise:
            problems.append(f"{key}: noise groups {z['mean']:.3f}, table says {exp_noise}")
    sub = df[(df[COL_SCALE] == "concurrent") & (df[COL_COND] == "Proposed") & df[COL_REPORT]]
    if (round(sub[COL_RATE].mean(), 1), len(sub)) != EXPECTED_PROPOSED_WITH_REPORT:
        problems.append(f"Proposed concurrent with report: {sub[COL_RATE].mean():.2f} "
                        f"(N = {len(sub)}), table says {EXPECTED_PROPOSED_WITH_REPORT}")
    return problems, sub[COL_RATE].mean()


def draw_panel(ax, stats, width, ylim):
    x = np.arange(len(SCALES))
    offsets = {c: (i - 1) * width for i, c in enumerate(CONDS)}
    handles = []
    for c in CONDS:
        m = np.array([stats[(s, c)]["mean"] for s in SCALES])
        lo = np.array([stats[(s, c)]["lo"] for s in SCALES])
        hi = np.array([stats[(s, c)]["hi"] for s in SCALES])
        # bars start at the axis floor so a truncated axis (panel a) is drawn honestly
        h = ax.bar(x + offsets[c], m - ylim[0], width, bottom=ylim[0], color=COLORS[c],
                   edgecolor="black", linewidth=0.8, label=c,
                   yerr=[m - lo, hi - m], capsize=4,
                   error_kw=dict(lw=1.1, capthick=1.1))
        handles.append(h)
    ax.set_xticks(x)
    ax.set_xticklabels(["Single incident\n(N = 80)", "Three concurrent\n(N = 80 / 80 / 79)"],
                       fontsize=11)
    ax.set_ylim(*ylim)
    ax.spines[["top", "right"]].set_visible(False)
    return x, offsets, handles


def add_brackets(ax, stats, x, offsets, brackets, step, gap):
    """Stack brackets above the highest CI in each group so they never overlap bars."""
    level = {}
    for s, c1, c2, label in brackets:
        i = SCALES.index(s)
        top = max(stats[(s, c)]["hi"] for c in CONDS)
        y = level.get(s, top + gap)
        x1, x2 = x[i] + offsets[c1], x[i] + offsets[c2]
        ax.plot([x1, x1, x2, x2], [y, y + step * 0.4, y + step * 0.4, y], lw=1, c="black")
        ax.text((x1 + x2) / 2, y + step * 0.45, label, ha="center", va="bottom", fontsize=9)
        level[s] = y + step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--out", default="fig5_narrative_noise")
    args = ap.parse_args()

    import os
    if not os.path.exists(args.csv):
        print(f"找不到逐次評分資料：{args.csv}\n"
              "這張圖必須從每一次 run 的評分結果計算（bootstrap CI 無法只用平均數和 SD 算出）。\n"
              "請把評分程式輸出的 CSV 路徑填到 DEFAULT_CSV，或執行：\n"
              "  python fig5_narrative_noise.py --csv 你的檔案.csv")
        sys.exit(1)
    df = pd.read_csv(args.csv)
    df[COL_REPORT] = df[COL_REPORT].astype(str).str.lower().isin(["true", "1", "yes"])
    rng = np.random.default_rng(SEED)
    rate = summarize(df, COL_RATE, rng)
    noise = summarize(df, COL_NOISE, rng)

    problems, proposed_with_report = check_against_tables(rate, noise, df)
    if problems and not args.no_check:
        print("Data do not reproduce Tables 5/6 — fix the data or the tables first:")
        print("\n".join("  - " + p for p in problems))
        sys.exit(1)

    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "axes.linewidth": 1.0})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.3))
    width = 0.26

    # (a) narrative mention rate; axis starts at 80% (state this in the caption)
    x, off, handles = draw_panel(ax1, rate, width, (80, 106))
    ax1.set_yticks(range(80, 101, 5))
    ax1.set_ylabel("Narrative mention rate (%)", fontsize=11)
    ax1.set_title("(a) Core facts stated in narrative", fontsize=12)
    add_brackets(ax1, rate, x, off, BRACKETS_A, step=2.0, gap=0.8)
    # Proposed, concurrent: mean over produced reports only (3 no-report runs scored 0)
    xp = x[1] + off["Proposed"]
    ax1.plot(xp, proposed_with_report, marker="D", ms=6, mfc="white", mec="black",
             mew=1.1, zorder=5)
    ax1.annotate(f"{proposed_with_report:.1f}%\nreports only\n(N = 76)",
                 xy=(xp, proposed_with_report), xytext=(xp + 0.2, 86),
                 fontsize=8, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", lw=0.7, color="black"))

    # (b) noise groups per report
    x, off, _ = draw_panel(ax2, noise, width, (0, 2.6))
    ax2.set_yticks(np.arange(0, 2.01, 0.5))
    ax2.set_ylabel("Noise groups cited per report", fontsize=11)
    ax2.set_title("(b) Injected noise cited in narrative", fontsize=12)
    add_brackets(ax2, noise, x, off, BRACKETS_B, step=0.2, gap=0.08)

    fig.legend(handles, CONDS, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=3, frameon=False, fontsize=10)
    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    fig.savefig(f"{args.out}.pdf", bbox_inches="tight")
    fig.savefig(f"{args.out}.png", dpi=300, bbox_inches="tight")

    print("condition / scale: mean [95% bootstrap CI]")
    for key in rate:
        r, z = rate[key], noise[key]
        print(f"  {key[1]:9s} {key[0]:10s}  rate {r['mean']:5.1f} [{r['lo']:5.1f}, {r['hi']:5.1f}]"
              f"   noise {z['mean']:.2f} [{z['lo']:.2f}, {z['hi']:.2f}]")


if __name__ == "__main__":
    main()