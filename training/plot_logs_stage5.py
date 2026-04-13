"""
Plot Stage 5 adversarial training curves.

The KEY plot: thief escape rate vs police catch rate on the same graph.
This is the main analytical output of the entire project.

Usage:
    python plot_logs_stage5.py
    python plot_logs_stage5.py logs/stage5_log.csv
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
        episodes=[], lengths=[], outcomes=[],
        thief_reward=[], p0_reward=[], p1_reward=[],
        thief_pg=[], thief_vf=[], thief_ent=[],
        p0_pg=[], p0_vf=[], p0_ent=[],
        p1_pg=[], p1_vf=[], p1_ent=[],
        eval_eps=[], eval_catch=[], eval_escape=[],
        eval_trap=[], eval_timeout=[], eval_steps=[],
    )
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["episodes"].append(int(row["episode"]))
            data["lengths"].append(int(row["ep_length"]))
            data["outcomes"].append(row["outcome"])
            data["thief_reward"].append(float(row["thief_reward"]))
            data["p0_reward"].append(float(row["police0_reward"]))
            data["p1_reward"].append(float(row["police1_reward"]))
            data["thief_pg"].append(float(row["thief_pg_loss"]))
            data["thief_vf"].append(float(row["thief_vf_loss"]))
            data["thief_ent"].append(float(row["thief_entropy"]))
            data["p0_pg"].append(float(row["police0_pg_loss"]))
            data["p0_vf"].append(float(row["police0_vf_loss"]))
            data["p0_ent"].append(float(row["police0_entropy"]))
            data["p1_pg"].append(float(row["police1_pg_loss"]))
            data["p1_vf"].append(float(row["police1_vf_loss"]))
            data["p1_ent"].append(float(row["police1_entropy"]))
            if row["eval_catch_rate"]:
                data["eval_eps"].append(int(row["episode"]))
                data["eval_catch"].append(float(row["eval_catch_rate"]))
                data["eval_escape"].append(float(row["eval_escape_rate"]))
                data["eval_trap"].append(float(row["eval_trap_rate"]))
                data["eval_timeout"].append(float(row["eval_timeout_rate"]))
                data["eval_steps"].append(float(row["eval_avg_steps"]))
    return data


def main(log_path="logs/stage5_log.csv"):
    data = load_log(log_path)
    episodes = data["episodes"]
    w = 100  # wider smoothing for adversarial curves

    catches = [1.0 if o == "caught" else 0.0 for o in data["outcomes"]]
    escapes = [1.0 if o == "escaped" else 0.0 for o in data["outcomes"]]
    traps = [1.0 if o == "trap" else 0.0 for o in data["outcomes"]]

    fig, axes = plt.subplots(3, 3, figsize=(20, 14))
    fig.suptitle("Stage 5 — Adversarial Co-Training",
                 fontsize=16, fontweight="bold")

    # ═══════════════════════════════════════════════════════
    #  ROW 1: THE KEY ADVERSARIAL PLOTS
    # ═══════════════════════════════════════════════════════

    # 1. THE MAIN PLOT — Escape vs Catch on same axes
    ax = axes[0, 0]
    x = episodes[w-1:]
    escape_sm = smooth(escapes, w)
    catch_sm = smooth(catches, w)
    ax.plot(x, np.array(escape_sm) * 100,
            color="#2ecc71", linewidth=2.5, label="Thief Escape %")
    ax.plot(x, np.array(catch_sm) * 100,
            color="#2c3e50", linewidth=2.5, label="Police Catch %")
    if data["eval_eps"]:
        ax.plot(data["eval_eps"], [e*100 for e in data["eval_escape"]],
                'o', color="#2ecc71", markersize=5, alpha=0.7)
        ax.plot(data["eval_eps"], [c*100 for c in data["eval_catch"]],
                's', color="#2c3e50", markersize=5, alpha=0.7)
    ax.set_title("★ Adversarial Dynamics ★", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # 2. Outcome breakdown (stacked)
    ax = axes[0, 1]
    trap_sm = smooth(traps, w)
    ax.fill_between(x, 0, np.array(escape_sm)*100,
                    alpha=0.5, color='#2ecc71', label='Escaped')
    ax.fill_between(x, np.array(escape_sm)*100,
                    (np.array(escape_sm) + np.array(catch_sm))*100,
                    alpha=0.5, color='#2c3e50', label='Caught')
    ax.fill_between(x,
                    (np.array(escape_sm) + np.array(catch_sm))*100,
                    (np.array(escape_sm) + np.array(catch_sm) + np.array(trap_sm))*100,
                    alpha=0.5, color='#e67e22', label='Trap')
    ax.set_title("Outcome Breakdown (%)")
    ax.set_xlabel("Episode")
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

    # 3. Episode length
    ax = axes[0, 2]
    ax.plot(episodes, data["lengths"], alpha=0.1, color="darkorange")
    ax.plot(x, smooth(data["lengths"], w),
            color="darkorange", linewidth=2)
    if data["eval_eps"]:
        ax.plot(data["eval_eps"], data["eval_steps"], 'o-',
                color="crimson", markersize=4, label="Eval avg")
        ax.legend()
    ax.set_title("Episode Length")
    ax.set_xlabel("Episode")
    ax.grid(alpha=0.3)

    # ═══════════════════════════════════════════════════════
    #  ROW 2: Rewards and entropy
    # ═══════════════════════════════════════════════════════

    # 4. Thief vs Police rewards
    ax = axes[1, 0]
    ax.plot(episodes, data["thief_reward"], alpha=0.05, color="#e74c3c")
    ax.plot(x, smooth(data["thief_reward"], w),
            color="#e74c3c", linewidth=2, label="Thief")
    p_avg = [(a+b)/2 for a, b in zip(data["p0_reward"], data["p1_reward"])]
    ax.plot(episodes, p_avg, alpha=0.05, color="#2c3e50")
    ax.plot(x, smooth(p_avg, w),
            color="#2c3e50", linewidth=2, label="Police (avg)")
    ax.set_title("Rewards")
    ax.set_xlabel("Episode")
    ax.legend()
    ax.grid(alpha=0.3)

    # 5. All entropies
    ax = axes[1, 1]
    ax.plot(x, smooth(data["thief_ent"], w),
            color="#e74c3c", linewidth=2, label="Thief")
    ax.plot(x, smooth(data["p0_ent"], w),
            color="#3498db", linewidth=2, label="Police 0")
    ax.plot(x, smooth(data["p1_ent"], w),
            color="#9b59b6", linewidth=2, label="Police 1")
    ax.set_title("Entropy (all agents)")
    ax.set_xlabel("Episode")
    ax.legend()
    ax.grid(alpha=0.3)

    # 6. Thief reward distribution over time (shows adaptation)
    ax = axes[1, 2]
    ax.plot(episodes, data["thief_reward"], alpha=0.08, color="#e74c3c")
    ax.plot(x, smooth(data["thief_reward"], w),
            color="#e74c3c", linewidth=2)
    ax.axhline(y=100, color='#2ecc71', linestyle='--', alpha=0.5, label='Goal reward')
    ax.axhline(y=-100, color='#2c3e50', linestyle='--', alpha=0.5, label='Caught penalty')
    ax.set_title("Thief Reward")
    ax.set_xlabel("Episode")
    ax.legend()
    ax.grid(alpha=0.3)

    # ═══════════════════════════════════════════════════════
    #  ROW 3: Loss curves
    # ═══════════════════════════════════════════════════════

    # 7. Policy losses
    ax = axes[2, 0]
    ax.plot(x, smooth(data["thief_pg"], w),
            color="#e74c3c", linewidth=2, label="Thief")
    ax.plot(x, smooth(data["p0_pg"], w),
            color="#3498db", linewidth=2, label="Police 0")
    ax.plot(x, smooth(data["p1_pg"], w),
            color="#9b59b6", linewidth=2, label="Police 1")
    ax.set_title("Policy Loss")
    ax.set_xlabel("Episode")
    ax.legend()
    ax.grid(alpha=0.3)

    # 8. Value losses
    ax = axes[2, 1]
    ax.plot(x, smooth(data["thief_vf"], w),
            color="#e74c3c", linewidth=2, label="Thief")
    ax.plot(x, smooth(data["p0_vf"], w),
            color="#3498db", linewidth=2, label="Police 0")
    ax.plot(x, smooth(data["p1_vf"], w),
            color="#9b59b6", linewidth=2, label="Police 1")
    ax.set_title("Value Loss")
    ax.set_xlabel("Episode")
    ax.legend()
    ax.grid(alpha=0.3)

    # 9. Police individual catch contribution
    ax = axes[2, 2]
    p0_catches = []
    p1_catches = []
    # Reconstruct from rewards: +100 = you caught, +50 = teammate caught
    for r0, r1 in zip(data["p0_reward"], data["p1_reward"]):
        p0_catches.append(1.0 if r0 >= 90 else 0.0)   # +100 = own catch
        p1_catches.append(1.0 if r1 >= 90 else 0.0)
    ax.plot(x, np.array(smooth(p0_catches, w)) * 100,
            color="#3498db", linewidth=2, label="Police 0 catches")
    ax.plot(x, np.array(smooth(p1_catches, w)) * 100,
            color="#9b59b6", linewidth=2, label="Police 1 catches")
    ax.set_title("Individual Catch Rate (%)")
    ax.set_xlabel("Episode")
    ax.set_ylim(-5, 60)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = log_path.replace(".csv", "_curves.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "logs/stage5_log.csv"
    if os.path.exists(path):
        main(path)
    else:
        print(f"Log not found: {path}")