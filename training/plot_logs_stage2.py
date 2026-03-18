"""
Plot Stage 2 training curves from CSV logs.

Usage:
    python plot_logs_stage2.py                                  # both phases
    python plot_logs_stage2.py logs/stage2_phaseA_log.csv       # single file
    python plot_logs_stage2.py phaseA.csv phaseB.csv            # explicit pair
"""

import sys
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def smooth(arr, window=50):
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='valid')


def load_log(path):
    data = dict(
        episodes=[], rewards=[], lengths=[], successes=[],
        trap_hits=[], traffic_hits=[],
        pg_losses=[], vf_losses=[], entropies=[], clip_fracs=[],
        eval_eps=[], eval_rewards=[], eval_srs=[],
        eval_trap=[], eval_traffic=[],
    )
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["episodes"].append(int(row["episode"]))
            data["rewards"].append(float(row["ep_reward"]))
            data["lengths"].append(int(row["ep_length"]))
            data["successes"].append(int(row["reached_goal"]))
            data["trap_hits"].append(int(row["trap_hits"]))
            data["traffic_hits"].append(int(row["traffic_hits"]))
            data["pg_losses"].append(float(row["pg_loss"]))
            data["vf_losses"].append(float(row["vf_loss"]))
            data["entropies"].append(float(row["entropy"]))
            data["clip_fracs"].append(float(row["clip_frac"]))
            if row["eval_avg_reward"]:
                data["eval_eps"].append(int(row["episode"]))
                data["eval_rewards"].append(float(row["eval_avg_reward"]))
                data["eval_srs"].append(float(row["eval_success_rate"]))
                data["eval_trap"].append(float(row["eval_avg_trap_hits"]))
                data["eval_traffic"].append(float(row["eval_avg_traffic_hits"]))
    return data


def plot_phase(data, phase_label, out_path):
    """Plot one phase's curves — 3×3 grid."""
    episodes = data["episodes"]
    w = 50

    fig, axes = plt.subplots(3, 3, figsize=(20, 14))
    fig.suptitle(f"Stage 2 — Phase {phase_label} Training Curves",
                 fontsize=16, fontweight="bold")

    # ── Row 1 ──

    # 1. Episode reward
    ax = axes[0, 0]
    ax.plot(episodes, data["rewards"], alpha=0.15, color="steelblue")
    sm = smooth(data["rewards"], w)
    ax.plot(episodes[w-1:], sm, color="steelblue", linewidth=2)
    if data["eval_eps"]:
        ax.plot(data["eval_eps"], data["eval_rewards"], 'o-',
                color="crimson", markersize=4, label="Greedy eval")
        ax.legend()
    ax.set_title("Episode Reward")
    ax.set_xlabel("Episode")
    ax.grid(alpha=0.3)

    # 2. Success rate
    ax = axes[0, 1]
    sr_sm = smooth(data["successes"], w)
    ax.plot(episodes[w-1:], np.array(sr_sm) * 100, color="seagreen", linewidth=2)
    if data["eval_eps"]:
        ax.plot(data["eval_eps"], [s*100 for s in data["eval_srs"]], 'o-',
                color="crimson", markersize=4, label="Greedy eval")
        ax.legend()
    ax.set_title("Success Rate (%)")
    ax.set_xlabel("Episode")
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3)

    # 3. Episode length
    ax = axes[0, 2]
    ax.plot(episodes, data["lengths"], alpha=0.15, color="darkorange")
    ax.plot(episodes[w-1:], smooth(data["lengths"], w),
            color="darkorange", linewidth=2)
    ax.set_title("Episode Length")
    ax.set_xlabel("Episode")
    ax.grid(alpha=0.3)

    # ── Row 2: losses + entropy ──

    ax = axes[1, 0]
    ax.plot(episodes, data["pg_losses"], alpha=0.3, color="purple")
    ax.plot(episodes[w-1:], smooth(data["pg_losses"], w),
            color="purple", linewidth=2)
    ax.set_title("Policy Loss")
    ax.set_xlabel("Episode")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(episodes, data["vf_losses"], alpha=0.3, color="teal")
    ax.plot(episodes[w-1:], smooth(data["vf_losses"], w),
            color="teal", linewidth=2)
    ax.set_title("Value Loss")
    ax.set_xlabel("Episode")
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.plot(episodes, data["entropies"], alpha=0.3, color="coral")
    ax.plot(episodes[w-1:], smooth(data["entropies"], w),
            color="coral", linewidth=2, label="Entropy")
    ax2 = ax.twinx()
    ax2.plot(episodes, data["clip_fracs"], alpha=0.3, color="gray")
    ax2.plot(episodes[w-1:], smooth(data["clip_fracs"], w),
             color="gray", linewidth=2, label="Clip frac")
    ax.set_title("Entropy & Clip Fraction")
    ax.set_xlabel("Episode")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # ── Row 3: trap hits, traffic hits, combined hazards ──

    ax = axes[2, 0]
    ax.plot(episodes, data["trap_hits"], alpha=0.2, color="#e67e22")
    ax.plot(episodes[w-1:], smooth(data["trap_hits"], w),
            color="#e67e22", linewidth=2)
    if data["eval_eps"] and data["eval_trap"]:
        ax.plot(data["eval_eps"], data["eval_trap"], 'o-',
                color="crimson", markersize=4, label="Eval avg")
        ax.legend()
    ax.set_title("Trap Hits / Episode")
    ax.set_xlabel("Episode")
    ax.grid(alpha=0.3)

    ax = axes[2, 1]
    ax.plot(episodes, data["traffic_hits"], alpha=0.2, color="#f1c40f")
    ax.plot(episodes[w-1:], smooth(data["traffic_hits"], w),
            color="#f1c40f", linewidth=2)
    if data["eval_eps"] and data["eval_traffic"]:
        ax.plot(data["eval_eps"], data["eval_traffic"], 'o-',
                color="crimson", markersize=4, label="Eval avg")
        ax.legend()
    ax.set_title("Traffic Hits / Episode")
    ax.set_xlabel("Episode")
    ax.grid(alpha=0.3)

    # Combined hazard rate (trap + traffic per step)
    ax = axes[2, 2]
    combined = np.array(data["trap_hits"]) + np.array(data["traffic_hits"])
    per_step = combined / np.maximum(np.array(data["lengths"]), 1)
    ax.plot(episodes, per_step, alpha=0.2, color="#8e44ad")
    ax.plot(episodes[w-1:], smooth(per_step, w),
            color="#8e44ad", linewidth=2)
    ax.set_title("Hazard Rate (hits / step)")
    ax.set_xlabel("Episode")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot → {out_path}")
    plt.close(fig)


def main():
    if len(sys.argv) == 1:
        # Default: plot both phases
        pairs = [
            ("logs/stage2_phaseA_log.csv", "A", "logs/stage2_phaseA_curves.png"),
            ("logs/stage2_phaseB_log.csv", "B", "logs/stage2_phaseB_curves.png"),
        ]
    elif len(sys.argv) == 2:
        p = sys.argv[1]
        label = "A" if "phaseA" in p else ("B" if "phaseB" in p else "?")
        pairs = [(p, label, p.replace(".csv", "_curves.png"))]
    else:
        pairs = []
        for p in sys.argv[1:]:
            label = "A" if "phaseA" in p else ("B" if "phaseB" in p else "?")
            pairs.append((p, label, p.replace(".csv", "_curves.png")))

    for csv_path, label, out_path in pairs:
        if os.path.exists(csv_path):
            data = load_log(csv_path)
            plot_phase(data, label, out_path)
        else:
            print(f"Log not found: {csv_path}")


if __name__ == "__main__":
    main()