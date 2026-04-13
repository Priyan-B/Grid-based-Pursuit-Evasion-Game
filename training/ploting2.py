"""
plot_report.py — Paper-ready figures from training logs.

Generates in training/logs/:
    report_overview.png       – Bar chart: final results all stages
    report_curriculum.png     – 2×2: training curves Stages 1–4
    report_adversarial.png    – Stage 5 adversarial dynamics (headline)
    report_stage5.png         – Stage 5 detail: breakdown + rewards + entropy
"""

import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Paths ──
LOG_DIR = "logs"
OUT_DIR = "logs"

# ── Color palette (print-friendly, accessible) ──
C = {
    "thief":   "#1b7837",   # rich green
    "police":  "#762a83",   # deep purple
    "trap":    "#d95f02",   # burnt orange
    "reward":  "#2166ac",   # steel blue
    "cctv":    "#b2abd2",   # lavender
    "gray":    "#636363",
    "accent":  "#e7298a",   # pink accent
}

# ── Style ──
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 250,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linewidth": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def smooth(arr, window):
    """Simple rolling mean."""
    if len(arr) < window:
        return np.array(arr, dtype=float)
    k = np.ones(window) / window
    return np.convolve(arr, k, mode="valid")


def safe_load(path, fields):
    """Load specific columns from CSV. Returns dict of lists or None."""
    fp = os.path.join(LOG_DIR, path)
    if not os.path.exists(fp):
        print(f"  SKIP: {path} not found")
        return None
    data = {f: [] for f in fields}
    with open(fp) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for field in fields:
                val = row.get(field, "").strip()
                if val == "":
                    continue
                try:
                    data[field].append(float(val))
                except ValueError:
                    data[field].append(val)
    print(f"  Loaded {path} ({len(data[fields[0]])} rows)")
    return data


def kfmt(x, _):
    return f"{x/1000:.0f}K" if x >= 1000 else f"{x:.0f}"


# ═══════════════════════════════════════════════════════
#  FIG 1: Overview bar chart
# ═══════════════════════════════════════════════════════

def plot_overview():
    print("\n[1] Overview bar chart")

    labels = ["Stage 1\nNavigation", "Stage 2\nTraps+Traffic",
              "Stage 3\nRandom Police", "Stage 4\nPolice Train",
              "Stage 5\nAdversarial"]
    thief_vals  = [98, 93, 63, 70, 66]
    police_vals = [0,  0,  33, 22, 32]

    x = np.arange(len(labels))
    w = 0.32

    fig, ax = plt.subplots(figsize=(9, 4))

    b1 = ax.bar(x - w/2, thief_vals, w, color=C["thief"], label="Thief Escape %",
                edgecolor="white", linewidth=0.6)
    b2 = ax.bar(x + w/2, police_vals, w, color=C["police"], label="Police Catch %",
                edgecolor="white", linewidth=0.6)

    for bar in b1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 1.5,
                    f"{h:.0f}%", ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color=C["thief"])
    for bar in b2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 1.5,
                    f"{h:.0f}%", ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color=C["police"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 112)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_title("Final Performance Across All Stages")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "report_overview.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ═══════════════════════════════════════════════════════
#  FIG 2: Curriculum stages 1–4 (2×2)
# ═══════════════════════════════════════════════════════

def plot_curriculum():
    print("\n[2] Curriculum (Stages 1–4)")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("Curriculum Training: Stages 1–4", fontsize=13, fontweight="bold", y=0.98)

    # ── Stage 1 ──
    d = safe_load("training_log.csv", ["episode", "reached_goal"])
    if d:
        eps = np.array(d["episode"])
        sr = smooth([float(x) for x in d["reached_goal"]], 2000) * 100
        ax = axes[0, 0]
        ax.plot(eps[:len(sr)], sr, color=C["thief"], linewidth=1.5)
        ax.set_title("Stage 1 — Navigation")
        ax.set_ylabel("Success Rate (%)")
        ax.set_xlabel("Episode")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(kfmt))
        ax.set_ylim(0, 105)

    # ── Stage 2 ──
    ax = axes[0, 1]
    for fname, label, ls in [("stage2_phaseA_log.csv", "Phase A", "-"),
                              ("stage2_phaseB_log.csv", "Phase B", "--")]:
        d = safe_load(fname, ["episode", "reached_goal", "trap_hits"])
        if d:
            eps = np.array(d["episode"])
            w = 2000
            sr = smooth([float(x) for x in d["reached_goal"]], w) * 100
            tr = smooth([float(x) for x in d["trap_hits"]], w) * 100
            ax.plot(eps[:len(sr)], sr, color=C["thief"], linewidth=1.5,
                    linestyle=ls, label=f"Success ({label})")
            ax.plot(eps[:len(tr)], tr, color=C["trap"], linewidth=1.2,
                    linestyle=ls, alpha=0.8, label=f"Trap death ({label})")
    ax.set_title("Stage 2 — Traps + Traffic")
    ax.set_ylabel("Rate (%)")
    ax.set_xlabel("Episode")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(kfmt))
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, loc="center right", framealpha=0.9)

    # ── Stage 3 ──
    d = safe_load("stage3_log.csv", ["episode", "reached_goal", "caught_by_police"])
    if d:
        eps = np.array(d["episode"])
        w = 5000
        sr = smooth([float(x) for x in d["reached_goal"]], w) * 100
        cr = smooth([float(x) for x in d["caught_by_police"]], w) * 100
        ax = axes[1, 0]
        ax.plot(eps[:len(sr)], sr, color=C["thief"], linewidth=1.5, label="Thief escape")
        ax.plot(eps[:len(cr)], cr, color=C["police"], linewidth=1.5, label="Police catch")
        ax.set_title("Stage 3 — Random Police + CCTV")
        ax.set_ylabel("Rate (%)")
        ax.set_xlabel("Episode")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(kfmt))
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8, framealpha=0.9)

    # ── Stage 4 ──
    d = safe_load("stage4_log.csv", ["episode", "outcome"])
    if d:
        eps = np.array(d["episode"])
        w = 10000
        catches = np.array([1.0 if o == "caught" else 0.0 for o in d["outcome"]])
        escapes = np.array([1.0 if o == "escaped" else 0.0 for o in d["outcome"]])
        c_sm = smooth(catches, w) * 100
        e_sm = smooth(escapes, w) * 100
        ax = axes[1, 1]
        ax.plot(eps[:len(c_sm)], c_sm, color=C["police"], linewidth=1.5, label="Police catch")
        ax.plot(eps[:len(e_sm)], e_sm, color=C["thief"], linewidth=1.5, label="Thief escape")
        ax.set_title("Stage 4 — Police Training (Frozen Thief)")
        ax.set_ylabel("Rate (%)")
        ax.set_xlabel("Episode")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(kfmt))
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8, framealpha=0.9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT_DIR, "report_curriculum.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ═══════════════════════════════════════════════════════
#  FIG 3: Stage 5 adversarial dynamics (headline)
# ═══════════════════════════════════════════════════════

def plot_adversarial():
    print("\n[3] Adversarial dynamics (headline)")

    d = safe_load("stage5_log.csv", ["episode", "outcome"])
    if d is None:
        return

    eps = np.array(d["episode"])
    W = 8000  # heavy smoothing for 400K episodes
    escapes = np.array([1.0 if o == "escaped" else 0.0 for o in d["outcome"]])
    catches = np.array([1.0 if o == "caught"  else 0.0 for o in d["outcome"]])
    traps   = np.array([1.0 if o == "trap"    else 0.0 for o in d["outcome"]])

    e_sm = smooth(escapes, W) * 100
    c_sm = smooth(catches, W) * 100
    t_sm = smooth(traps, W)   * 100
    x = eps[:len(e_sm)]

    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.plot(x, e_sm, color=C["thief"],  linewidth=2.2, label="Thief Escape Rate")
    ax.plot(x, c_sm, color=C["police"], linewidth=2.2, label="Police Catch Rate")
    ax.plot(x, t_sm, color=C["trap"],   linewidth=1.2, alpha=0.6, label="Trap Death Rate")

    ax.fill_between(x, e_sm, alpha=0.08, color=C["thief"])
    ax.fill_between(x, c_sm, alpha=0.08, color=C["police"])

    # Start / end annotations
    ax.annotate(f"{e_sm[0]:.0f}%", xy=(x[0], e_sm[0]), xytext=(x[0]+15000, e_sm[0]+5),
                fontsize=9, color=C["thief"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C["thief"], lw=0.8))
    ax.annotate(f"{e_sm[-1]:.0f}%", xy=(x[-1], e_sm[-1]), xytext=(x[-1]-60000, e_sm[-1]+6),
                fontsize=9, color=C["thief"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C["thief"], lw=0.8))
    ax.annotate(f"{c_sm[0]:.0f}%", xy=(x[0], c_sm[0]), xytext=(x[0]+15000, c_sm[0]-7),
                fontsize=9, color=C["police"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C["police"], lw=0.8))
    ax.annotate(f"{c_sm[-1]:.0f}%", xy=(x[-1], c_sm[-1]), xytext=(x[-1]-60000, c_sm[-1]-7),
                fontsize=9, color=C["police"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C["police"], lw=0.8))

    ax.set_title("Stage 5 — Adversarial Co-Training Dynamics")
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Rate (%)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(kfmt))
    ax.set_ylim(0, 95)
    ax.legend(loc="center right", fontsize=10, framealpha=0.95)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "report_adversarial.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ═══════════════════════════════════════════════════════
#  FIG 4: Stage 5 detail (outcome breakdown + rewards)
# ═══════════════════════════════════════════════════════

def plot_stage5_detail():
    print("\n[4] Stage 5 detail")

    d = safe_load("stage5_log.csv", [
        "episode", "outcome", "thief_reward",
        "police0_reward", "police1_reward",
        "thief_entropy", "police0_entropy", "police1_entropy",
    ])
    if d is None:
        return

    eps = np.array(d["episode"])
    W = 8000
    escapes = np.array([1.0 if o == "escaped" else 0.0 for o in d["outcome"]])
    catches = np.array([1.0 if o == "caught"  else 0.0 for o in d["outcome"]])
    traps   = np.array([1.0 if o == "trap"    else 0.0 for o in d["outcome"]])

    e_sm = smooth(escapes, W)
    c_sm = smooth(catches, W)
    t_sm = smooth(traps, W)
    x = eps[:len(e_sm)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Stage 5 — Detailed Training Metrics", fontsize=12, fontweight="bold")

    # Outcome breakdown (stacked)
    ax = axes[0]
    ax.fill_between(x, 0, e_sm*100, alpha=0.7, color=C["thief"], label="Escaped")
    ax.fill_between(x, e_sm*100, (e_sm+c_sm)*100, alpha=0.7, color=C["police"], label="Caught")
    ax.fill_between(x, (e_sm+c_sm)*100, (e_sm+c_sm+t_sm)*100, alpha=0.7, color=C["trap"], label="Trap")
    ax.set_title("Outcome Breakdown")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative %")
    ax.set_ylim(0, 105)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(kfmt))
    ax.legend(fontsize=8, loc="center right")

    # Rewards
    ax = axes[1]
    thief_r = smooth(np.array(d["thief_reward"], dtype=float), W)
    pol_r = smooth((np.array(d["police0_reward"], dtype=float) +
                    np.array(d["police1_reward"], dtype=float)) / 2, W)
    ax.plot(x[:len(thief_r)], thief_r, color=C["thief"], linewidth=1.5, label="Thief")
    ax.plot(x[:len(pol_r)], pol_r, color=C["police"], linewidth=1.5, label="Police (avg)")
    ax.axhline(0, color=C["gray"], linewidth=0.5, linestyle="--")
    ax.set_title("Average Reward")
    ax.set_xlabel("Episode")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(kfmt))
    ax.legend(fontsize=8)

    # Entropy
    ax = axes[2]
    te = smooth(np.array(d["thief_entropy"], dtype=float), W)
    pe0 = smooth(np.array(d["police0_entropy"], dtype=float), W)
    pe1 = smooth(np.array(d["police1_entropy"], dtype=float), W)
    ax.plot(x[:len(te)], te, color=C["thief"], linewidth=1.5, label="Thief")
    ax.plot(x[:len(pe0)], pe0, color=C["police"], linewidth=1.5, label="Police 0")
    ax.plot(x[:len(pe1)], pe1, color=C["accent"], linewidth=1.5, label="Police 1")
    ax.set_title("Policy Entropy")
    ax.set_xlabel("Episode")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(kfmt))
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUT_DIR, "report_stage5.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Generating paper figures …")
    plot_overview()
    plot_curriculum()
    plot_adversarial()
    plot_stage5_detail()
    print("\n✓ Done. Figures saved to logs/report_*.png")