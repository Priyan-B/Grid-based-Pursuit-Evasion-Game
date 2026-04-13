"""
Master plotting script — generates report-ready figures from all stage CSVs.

Usage:
    python plot_report.py

Expects logs in training/logs/:
    training_log.csv          (Stage 1)
    stage2_phaseA_log.csv     (Stage 2A)
    stage2_phaseB_log.csv     (Stage 2B)
    stage3_log.csv            (Stage 3)
    stage4_log.csv            (Stage 4)
    stage5_log.csv            (Stage 5)

Generates PNGs in training/logs/report_*.png
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Style setup ──
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

LOG_DIR = "logs"
OUT_DIR = "logs"


def smooth(arr, window=100):
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='valid')


def safe_load(path, fields):
    """Load specific fields from a CSV. Returns dict of lists."""
    data = {f: [] for f in fields}
    full_path = os.path.join(LOG_DIR, path)
    if not os.path.exists(full_path):
        print(f"  Skipping {path} (not found)")
        return None
    with open(full_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for field in fields:
                val = row.get(field, "")
                if val == "":
                    continue
                try:
                    data[field].append(float(val))
                except ValueError:
                    data[field].append(val)
    print(f"  Loaded {path} ({len(data[fields[0]])} rows)")
    return data


# ══════════════════════════════════════════════════════════
#  FIGURE 1: Full Project Journey (single overview figure)
# ══════════════════════════════════════════════════════════

def plot_project_overview():
    """One figure showing success/catch rate across all stages."""
    print("\nPlotting: Project Overview")

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle("Project Progression — Thief vs Police Across All Stages",
                 fontsize=15, fontweight='bold')

    bar_labels = [
        "Stage 1\nNavigation",
        "Stage 2A\nStatic Traps",
        "Stage 2B\nDynamic Traps",
        "Stage 3\nvs Random Police",
        "Stage 4\nPolice Training",
        "Stage 5\nAdversarial",
    ]

    # These are your final results — update if different
    thief_sr =    [95,   90,   90,   64,   64,  65.6]
    police_catch = [0,    0,    0,    0,   29,  32.1]

    x = np.arange(len(bar_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, thief_sr, width, label='Thief Escape %',
                   color='#2ecc71', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, police_catch, width, label='Police Catch %',
                   color='#2c3e50', edgecolor='black', linewidth=0.5)

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                    f'{h:.0f}%', ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color='#27ae60')
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                    f'{h:.0f}%', ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color='#2c3e50')

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=10)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right')
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "report_overview.png")
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {out}")


# ══════════════════════════════════════════════════════════
#  FIGURE 2: Stage 1 — Navigation Learning
# ══════════════════════════════════════════════════════════

def plot_stage1():
    print("\nPlotting: Stage 1")
    data = safe_load("training_log.csv", [
        "episode", "ep_reward", "ep_length", "reached_goal",
        "eval_avg_reward", "eval_success_rate",
    ])
    if data is None:
        return

    eps = [int(e) for e in data["episode"]]
    rewards = data["ep_reward"]
    lengths = [int(l) for l in data["ep_length"]]
    successes = [int(s) for s in data["reached_goal"]]
    w = 100

    eval_eps = []
    eval_sr = []
    for i, e in enumerate(eps):
        if i < len(data.get("eval_success_rate", [])):
            val = data["eval_success_rate"][i]
            if isinstance(val, float):
                eval_eps.append(int(e))
                eval_sr.append(val)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("Stage 1 — Thief Learns Basic Navigation", fontsize=14, fontweight='bold')

    # Reward
    ax = axes[0]
    ax.plot(eps, rewards, alpha=0.08, color='steelblue')
    ax.plot(eps[w-1:], smooth(rewards, w), color='steelblue', linewidth=2)
    ax.set_title("Episode Reward")
    ax.set_xlabel("Episode")

    # Success rate
    ax = axes[1]
    sr_sm = smooth(successes, w)
    ax.plot(eps[w-1:], np.array(sr_sm) * 100, color='#2ecc71', linewidth=2)
    ax.set_title("Success Rate (%)")
    ax.set_xlabel("Episode")
    ax.set_ylim(-5, 105)

    # Episode length
    ax = axes[2]
    ax.plot(eps, lengths, alpha=0.08, color='darkorange')
    ax.plot(eps[w-1:], smooth(lengths, w), color='darkorange', linewidth=2)
    ax.set_title("Episode Length")
    ax.set_xlabel("Episode")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "report_stage1.png")
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {out}")


# ══════════════════════════════════════════════════════════
#  FIGURE 3: Stage 2 — Traps and Traffic
# ══════════════════════════════════════════════════════════

def plot_stage2():
    print("\nPlotting: Stage 2")
    w = 100

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("Stage 2 — Thief Learns Trap Avoidance + Traffic Dodging",
                 fontsize=14, fontweight='bold')

    for phase_idx, (filename, label) in enumerate([
        ("stage2_phaseA_log.csv", "Phase A (Static Traps)"),
        ("stage2_phaseB_log.csv", "Phase B (Dynamic Traps)"),
    ]):
        data = safe_load(filename, [
            "episode", "ep_reward", "reached_goal", "trap_hits", "traffic_hits",
        ])
        if data is None:
            continue

        eps = [int(e) for e in data["episode"]]
        rewards = data["ep_reward"]
        successes = [int(s) for s in data["reached_goal"]]
        traps = [int(t) for t in data["trap_hits"]]
        traffic = [int(t) for t in data["traffic_hits"]]

        row = phase_idx

        # Reward
        ax = axes[row, 0]
        ax.plot(eps, rewards, alpha=0.08, color='steelblue')
        ax.plot(eps[w-1:], smooth(rewards, w), color='steelblue', linewidth=2)
        ax.set_title(f"{label} — Reward")
        ax.set_xlabel("Episode")

        # Success rate
        ax = axes[row, 1]
        ax.plot(eps[w-1:], np.array(smooth(successes, w)) * 100,
                color='#2ecc71', linewidth=2, label='Success')
        ax.plot(eps[w-1:], np.array(smooth(traps, w)) * 100,
                color='#e67e22', linewidth=2, label='Trap Death')
        ax.set_title(f"{label} — Success & Trap Rate (%)")
        ax.set_xlabel("Episode")
        ax.set_ylim(-5, 105)
        ax.legend()

        # Traffic hits
        ax = axes[row, 2]
        ax.plot(eps, traffic, alpha=0.1, color='#f1c40f')
        ax.plot(eps[w-1:], smooth(traffic, w), color='#f1c40f', linewidth=2)
        ax.set_title(f"{label} — Traffic Hits / Episode")
        ax.set_xlabel("Episode")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "report_stage2.png")
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {out}")


# ══════════════════════════════════════════════════════════
#  FIGURE 4: Stage 3 — Police and CCTV
# ══════════════════════════════════════════════════════════

def plot_stage3():
    print("\nPlotting: Stage 3")
    data = safe_load("stage3_log.csv", [
        "episode", "ep_reward", "ep_length", "reached_goal",
        "caught_by_police", "trap_hits", "cctv_sightings",
    ])
    if data is None:
        return

    eps = [int(e) for e in data["episode"]]
    successes = [int(s) for s in data["reached_goal"]]
    catches = [int(c) for c in data["caught_by_police"]]
    traps = [int(t) for t in data["trap_hits"]]
    cctv = [int(c) for c in data["cctv_sightings"]]
    w = 100

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("Stage 3 — Thief Learns to Evade Random Police",
                 fontsize=14, fontweight='bold')

    # Success vs Catch rate
    ax = axes[0]
    ax.plot(eps[w-1:], np.array(smooth(successes, w)) * 100,
            color='#2ecc71', linewidth=2, label='Escape Rate')
    ax.plot(eps[w-1:], np.array(smooth(catches, w)) * 100,
            color='#2c3e50', linewidth=2, label='Catch Rate')
    ax.plot(eps[w-1:], np.array(smooth(traps, w)) * 100,
            color='#e67e22', linewidth=2, label='Trap Death Rate')
    ax.set_title("Outcome Rates (%)")
    ax.set_xlabel("Episode")
    ax.set_ylim(-5, 105)
    ax.legend()

    # Outcome stacked
    ax = axes[1]
    s_sm = smooth(successes, w)
    c_sm = smooth(catches, w)
    t_sm = smooth(traps, w)
    x = eps[w-1:]
    ax.fill_between(x, 0, np.array(s_sm)*100,
                    alpha=0.6, color='#2ecc71', label='Escaped')
    ax.fill_between(x, np.array(s_sm)*100,
                    (np.array(s_sm)+np.array(c_sm))*100,
                    alpha=0.6, color='#2c3e50', label='Caught')
    ax.fill_between(x, (np.array(s_sm)+np.array(c_sm))*100,
                    (np.array(s_sm)+np.array(c_sm)+np.array(t_sm))*100,
                    alpha=0.6, color='#e67e22', label='Trap')
    ax.set_title("Outcome Breakdown (%)")
    ax.set_xlabel("Episode")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)

    # CCTV sightings
    ax = axes[2]
    ax.plot(eps, cctv, alpha=0.1, color='#9b59b6')
    ax.plot(eps[w-1:], smooth(cctv, w), color='#9b59b6', linewidth=2)
    ax.set_title("CCTV Sightings / Episode")
    ax.set_xlabel("Episode")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "report_stage3.png")
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {out}")


# ══════════════════════════════════════════════════════════
#  FIGURE 5: Stage 4 — Police Learning
# ══════════════════════════════════════════════════════════

def plot_stage4():
    print("\nPlotting: Stage 4")
    data = safe_load("stage4_log.csv", [
        "episode", "ep_length", "outcome",
        "police0_reward", "police1_reward",
        "had_cctv_sighting",
    ])
    if data is None:
        return

    eps = [int(e) for e in data["episode"]]
    catches = [1.0 if o == "caught" else 0.0 for o in data["outcome"]]
    escapes = [1.0 if o == "escaped" else 0.0 for o in data["outcome"]]
    lengths = [int(l) for l in data["ep_length"]]
    cctv = [int(c) for c in data["had_cctv_sighting"]]
    w = 100

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("Stage 4 — Police Learn to Catch Frozen Thief",
                 fontsize=14, fontweight='bold')

    # Catch vs Escape
    ax = axes[0]
    ax.plot(eps[w-1:], np.array(smooth(catches, w)) * 100,
            color='#2c3e50', linewidth=2, label='Police Catch Rate')
    ax.plot(eps[w-1:], np.array(smooth(escapes, w)) * 100,
            color='#2ecc71', linewidth=2, label='Thief Escape Rate')
    ax.set_title("Catch vs Escape Rate (%)")
    ax.set_xlabel("Episode")
    ax.set_ylim(-5, 105)
    ax.legend()

    # Episode length
    ax = axes[1]
    ax.plot(eps, lengths, alpha=0.08, color='darkorange')
    ax.plot(eps[w-1:], smooth(lengths, w), color='darkorange', linewidth=2)
    ax.set_title("Episode Length")
    ax.set_xlabel("Episode")

    # CCTV usage
    ax = axes[2]
    ax.plot(eps[w-1:], np.array(smooth(cctv, w)) * 100,
            color='#9b59b6', linewidth=2)
    ax.set_title("Episodes with CCTV Sighting (%)")
    ax.set_xlabel("Episode")
    ax.set_ylim(-5, 105)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "report_stage4.png")
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {out}")


# ══════════════════════════════════════════════════════════
#  FIGURE 6: Stage 5 — Adversarial Dynamics (THE KEY PLOT)
# ══════════════════════════════════════════════════════════

def plot_stage5():
    print("\nPlotting: Stage 5 (Adversarial)")
    data = safe_load("stage5_log.csv", [
        "episode", "ep_length", "outcome",
        "thief_reward", "police0_reward", "police1_reward",
        "thief_entropy", "police0_entropy", "police1_entropy",
    ])
    if data is None:
        return

    eps = [int(e) for e in data["episode"]]
    catches = [1.0 if o == "caught" else 0.0 for o in data["outcome"]]
    escapes = [1.0 if o == "escaped" else 0.0 for o in data["outcome"]]
    traps = [1.0 if o == "trap" else 0.0 for o in data["outcome"]]
    lengths = [int(l) for l in data["ep_length"]]
    w = 200  # wider smoothing for adversarial

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Stage 5 — Adversarial Co-Training",
                 fontsize=15, fontweight='bold')

    # THE KEY PLOT — adversarial dynamics
    ax = axes[0, 0]
    x = eps[w-1:]
    escape_sm = smooth(escapes, w)
    catch_sm = smooth(catches, w)
    ax.plot(x, np.array(escape_sm) * 100,
            color='#2ecc71', linewidth=2.5, label='Thief Escape %')
    ax.plot(x, np.array(catch_sm) * 100,
            color='#2c3e50', linewidth=2.5, label='Police Catch %')
    ax.fill_between(x, np.array(escape_sm)*100, alpha=0.15, color='#2ecc71')
    ax.fill_between(x, np.array(catch_sm)*100, alpha=0.15, color='#2c3e50')
    ax.set_title("★ Adversarial Dynamics ★", fontsize=13)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=11, loc='center right')

    # Outcome breakdown
    ax = axes[0, 1]
    trap_sm = smooth(traps, w)
    ax.fill_between(x, 0, np.array(escape_sm)*100,
                    alpha=0.6, color='#2ecc71', label='Escaped')
    ax.fill_between(x, np.array(escape_sm)*100,
                    (np.array(escape_sm)+np.array(catch_sm))*100,
                    alpha=0.6, color='#2c3e50', label='Caught')
    ax.fill_between(x,
                    (np.array(escape_sm)+np.array(catch_sm))*100,
                    (np.array(escape_sm)+np.array(catch_sm)+np.array(trap_sm))*100,
                    alpha=0.6, color='#e67e22', label='Trap')
    ax.set_title("Outcome Breakdown (%)")
    ax.set_xlabel("Episode")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)

    # Episode length
    ax = axes[1, 0]
    ax.plot(eps, lengths, alpha=0.05, color='darkorange')
    ax.plot(x, smooth(lengths, w), color='darkorange', linewidth=2)
    ax.set_title("Episode Length (longer = more evasion)")
    ax.set_xlabel("Episode")

    # Entropy all agents
    ax = axes[1, 1]
    t_ent = data["thief_entropy"]
    p0_ent = data["police0_entropy"]
    p1_ent = data["police1_entropy"]
    ax.plot(x, smooth(t_ent, w), color='#e74c3c', linewidth=2, label='Thief')
    ax.plot(x, smooth(p0_ent, w), color='#3498db', linewidth=2, label='Police 0')
    ax.plot(x, smooth(p1_ent, w), color='#9b59b6', linewidth=2, label='Police 1')
    ax.set_title("Policy Entropy (exploration level)")
    ax.set_xlabel("Episode")
    ax.legend()

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "report_stage5.png")
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {out}")


# ══════════════════════════════════════════════════════════
#  FIGURE 7: Adversarial standalone (for report headline)
# ══════════════════════════════════════════════════════════

def plot_adversarial_headline():
    """Single clean figure — just the adversarial dynamics curve."""
    print("\nPlotting: Adversarial Headline")
    data = safe_load("stage5_log.csv", ["episode", "outcome"])
    if data is None:
        return

    eps = [int(e) for e in data["episode"]]
    catches = [1.0 if o == "caught" else 0.0 for o in data["outcome"]]
    escapes = [1.0 if o == "escaped" else 0.0 for o in data["outcome"]]
    w = 200

    fig, ax = plt.subplots(figsize=(12, 5))
    x = eps[w-1:]
    escape_sm = smooth(escapes, w)
    catch_sm = smooth(catches, w)

    ax.plot(x, np.array(escape_sm)*100,
            color='#2ecc71', linewidth=3, label='Thief Escape Rate')
    ax.plot(x, np.array(catch_sm)*100,
            color='#2c3e50', linewidth=3, label='Police Catch Rate')
    ax.fill_between(x, np.array(escape_sm)*100, alpha=0.12, color='#2ecc71')
    ax.fill_between(x, np.array(catch_sm)*100, alpha=0.12, color='#2c3e50')

    ax.set_title("Adversarial Co-Training Dynamics",
                 fontsize=15, fontweight='bold')
    ax.set_xlabel("Training Episode", fontsize=12)
    ax.set_ylabel("Rate (%)", fontsize=12)
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=12, loc='center right')

    # Annotations
    ax.annotate('Police improving →', xy=(x[-1]*0.7, catch_sm[-1]*100+5),
                fontsize=10, color='#2c3e50', fontstyle='italic')
    ax.annotate('← Thief adapting', xy=(x[-1]*0.7, escape_sm[-1]*100-5),
                fontsize=10, color='#27ae60', fontstyle='italic',
                ha='left')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "report_adversarial.png")
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {out}")


# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Generating report figures...")

    plot_project_overview()
    plot_stage1()
    plot_stage2()
    plot_stage3()
    plot_stage4()
    plot_stage5()
    plot_adversarial_headline()

    print("\nDone! All figures saved to logs/report_*.png")