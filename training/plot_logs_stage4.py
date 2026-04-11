"""
Plot Stage 4 training curves from CSV log.

Usage:
    python plot_logs_stage4.py                     # default path
    python plot_logs_stage4.py logs/stage4_log.csv # custom path
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
        catcher_idx=[], had_cctv=[],
        p0_reward=[], p1_reward=[],
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
            data["catcher_idx"].append(int(row["catcher_idx"]))
            data["had_cctv"].append(int(row["had_cctv_sighting"]))
            data["p0_reward"].append(float(row["police0_reward"]))
            data["p1_reward"].append(float(row["police1_reward"]))
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


def main(log_path="logs/stage4_log.csv"):
    data = load_log(log_path)
    episodes = data["episodes"]
    w = 50

    catches = [1.0 if o == "caught" else 0.0 for o in data["outcomes"]]
    escapes = [1.0 if o == "escaped" else 0.0 for o in data["outcomes"]]
    traps = [1.0 if o == "trap" else 0.0 for o in data["outcomes"]]

    fig, axes = plt.subplots(3, 3, figsize=(20, 14))
    fig.suptitle("Stage 4 — Police Training Curves",
                 fontsize=16, fontweight="bold")

    # ── Row 1: catch rate, escape rate, episode length ──

    ax = axes[0, 0]
    ax.plot(episodes[w-1:], np.array(smooth(catches, w)) * 100,
            color="#2c3e50", linewidth=2, label="Training")
    if data["eval_eps"]:
        ax.plot(data["eval_eps"], [c*100 for c in data["eval_catch"]],
                'o-', color="crimson", markersize=4, label="Greedy eval")
        ax.legend()
    ax.set_title("Police Catch Rate (%)")
    ax.set_xlabel("Episode")
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(episodes[w-1:], np.array(smooth(escapes, w)) * 100,
            color="#2ecc71", linewidth=2, label="Training")
    if data["eval_eps"]:
        ax.plot(data["eval_eps"], [e*100 for e in data["eval_escape"]],
                'o-', color="crimson", markersize=4, label="Greedy eval")
        ax.legend()
    ax.set_title("Thief Escape Rate (%)")
    ax.set_xlabel("Episode")
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.plot(episodes, data["lengths"], alpha=0.15, color="darkorange")
    ax.plot(episodes[w-1:], smooth(data["lengths"], w),
            color="darkorange", linewidth=2)
    ax.set_title("Episode Length")
    ax.set_xlabel("Episode")
    ax.grid(alpha=0.3)

    # ── Row 2: outcome breakdown, police rewards, CCTV ──

    ax = axes[1, 0]
    catch_sm = smooth(catches, w)
    escape_sm = smooth(escapes, w)
    trap_sm = smooth(traps, w)
    x = episodes[w-1:]
    ax.fill_between(x, 0, np.array(catch_sm)*100,
                    alpha=0.5, color='#2c3e50', label='Caught')
    ax.fill_between(x, np.array(catch_sm)*100,
                    (np.array(catch_sm) + np.array(escape_sm))*100,
                    alpha=0.5, color='#2ecc71', label='Escaped')
    ax.fill_between(x,
                    (np.array(catch_sm) + np.array(escape_sm))*100,
                    (np.array(catch_sm) + np.array(escape_sm) + np.array(trap_sm))*100,
                    alpha=0.5, color='#e67e22', label='Trap')
    ax.set_title("Outcome Breakdown (%)")
    ax.set_xlabel("Episode")
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(episodes, data["p0_reward"], alpha=0.1, color="#3498db")
    ax.plot(episodes[w-1:], smooth(data["p0_reward"], w),
            color="#3498db", linewidth=2, label="Police 0")
    ax.plot(episodes, data["p1_reward"], alpha=0.1, color="#e74c3c")
    ax.plot(episodes[w-1:], smooth(data["p1_reward"], w),
            color="#e74c3c", linewidth=2, label="Police 1")
    ax.set_title("Police Rewards")
    ax.set_xlabel("Episode")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.plot(episodes[w-1:], np.array(smooth(data["had_cctv"], w)) * 100,
            color="#9b59b6", linewidth=2)
    ax.set_title("Episodes with CCTV Sighting (%)")
    ax.set_xlabel("Episode")
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3)

    # ── Row 3: policy losses and entropy ──

    ax = axes[2, 0]
    ax.plot(episodes, data["p0_pg"], alpha=0.3, color="#3498db")
    ax.plot(episodes[w-1:], smooth(data["p0_pg"], w),
            color="#3498db", linewidth=2, label="Police 0")
    ax.plot(episodes, data["p1_pg"], alpha=0.3, color="#e74c3c")
    ax.plot(episodes[w-1:], smooth(data["p1_pg"], w),
            color="#e74c3c", linewidth=2, label="Police 1")
    ax.set_title("Policy Loss")
    ax.set_xlabel("Episode")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2, 1]
    ax.plot(episodes, data["p0_vf"], alpha=0.3, color="#3498db")
    ax.plot(episodes[w-1:], smooth(data["p0_vf"], w),
            color="#3498db", linewidth=2, label="Police 0")
    ax.plot(episodes, data["p1_vf"], alpha=0.3, color="#e74c3c")
    ax.plot(episodes[w-1:], smooth(data["p1_vf"], w),
            color="#e74c3c", linewidth=2, label="Police 1")
    ax.set_title("Value Loss")
    ax.set_xlabel("Episode")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2, 2]
    ax.plot(episodes, data["p0_ent"], alpha=0.3, color="#3498db")
    ax.plot(episodes[w-1:], smooth(data["p0_ent"], w),
            color="#3498db", linewidth=2, label="Police 0")
    ax.plot(episodes, data["p1_ent"], alpha=0.3, color="#e74c3c")
    ax.plot(episodes[w-1:], smooth(data["p1_ent"], w),
            color="#e74c3c", linewidth=2, label="Police 1")
    ax.set_title("Entropy")
    ax.set_xlabel("Episode")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = log_path.replace(".csv", "_curves.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "logs/stage4_log.csv"
    if os.path.exists(path):
        main(path)
    else:
        print(f"Log not found: {path}")