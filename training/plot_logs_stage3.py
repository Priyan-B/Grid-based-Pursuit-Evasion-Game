"""
Plot Stage 3 training curves from CSV log.

Usage:
    python plot_logs_stage3.py                       # default path
    python plot_logs_stage3.py logs/stage3_log.csv   # custom path
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
        catches=[], trap_hits=[], traffic_hits=[], cctv_sightings=[],
        pg_losses=[], vf_losses=[], entropies=[], clip_fracs=[],
        eval_eps=[], eval_rewards=[], eval_srs=[], eval_crs=[],
        eval_trap=[], eval_traffic=[],
    )
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["episodes"].append(int(row["episode"]))
            data["rewards"].append(float(row["ep_reward"]))
            data["lengths"].append(int(row["ep_length"]))
            data["successes"].append(int(row["reached_goal"]))
            data["catches"].append(int(row["caught_by_police"]))
            data["trap_hits"].append(int(row["trap_hits"]))
            data["traffic_hits"].append(int(row["traffic_hits"]))
            data["cctv_sightings"].append(int(row["cctv_sightings"]))
            data["pg_losses"].append(float(row["pg_loss"]))
            data["vf_losses"].append(float(row["vf_loss"]))
            data["entropies"].append(float(row["entropy"]))
            data["clip_fracs"].append(float(row["clip_frac"]))
            if row["eval_avg_reward"]:
                data["eval_eps"].append(int(row["episode"]))
                data["eval_rewards"].append(float(row["eval_avg_reward"]))
                data["eval_srs"].append(float(row["eval_success_rate"]))
                data["eval_crs"].append(float(row["eval_catch_rate"]))
                data["eval_trap"].append(float(row["eval_avg_trap_deaths"]))
                data["eval_traffic"].append(float(row["eval_avg_traffic_hits"]))
    return data


def main(log_path="logs/stage3_log.csv"):
    data = load_log(log_path)
    episodes = data["episodes"]
    w = 50

    fig, axes = plt.subplots(4, 3, figsize=(20, 18))
    fig.suptitle("Stage 3 — Police + CCTV Training Curves",
                 fontsize=16, fontweight="bold")

    # ── Row 1: reward, success, episode length ──

    ax = axes[0, 0]
    ax.plot(episodes, data["rewards"], alpha=0.15, color="steelblue")
    ax.plot(episodes[w-1:], smooth(data["rewards"], w),
            color="steelblue", linewidth=2)
    if data["eval_eps"]:
        ax.plot(data["eval_eps"], data["eval_rewards"], 'o-',
                color="crimson", markersize=4, label="Greedy eval")
        ax.legend()
    ax.set_title("Episode Reward")
    ax.set_xlabel("Episode")
    ax.grid(alpha=0.3)

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

    ax = axes[0, 2]
    ax.plot(episodes, data["lengths"], alpha=0.15, color="darkorange")
    ax.plot(episodes[w-1:], smooth(data["lengths"], w),
            color="darkorange", linewidth=2)
    ax.set_title("Episode Length")
    ax.set_xlabel("Episode")
    ax.grid(alpha=0.3)

    # ── Row 2: catch rate, trap hits, traffic hits ──

    ax = axes[1, 0]
    cr_sm = smooth(data["catches"], w)
    ax.plot(episodes[w-1:], np.array(cr_sm) * 100, color="#2c3e50", linewidth=2,
            label="Training")
    if data["eval_eps"]:
        ax.plot(data["eval_eps"], [c*100 for c in data["eval_crs"]], 'o-',
                color="crimson", markersize=4, label="Greedy eval")
        ax.legend()
    ax.set_title("Catch Rate by Police (%)")
    ax.set_xlabel("Episode")
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(episodes, data["trap_hits"], alpha=0.2, color="#e67e22")
    ax.plot(episodes[w-1:], smooth(data["trap_hits"], w),
            color="#e67e22", linewidth=2)
    ax.set_title("Trap Deaths / Episode")
    ax.set_xlabel("Episode")
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.plot(episodes, data["traffic_hits"], alpha=0.2, color="#f1c40f")
    ax.plot(episodes[w-1:], smooth(data["traffic_hits"], w),
            color="#f1c40f", linewidth=2)
    ax.set_title("Traffic Hits / Episode")
    ax.set_xlabel("Episode")
    ax.grid(alpha=0.3)

    # ── Row 3: losses + entropy ──

    ax = axes[2, 0]
    ax.plot(episodes, data["pg_losses"], alpha=0.3, color="purple")
    ax.plot(episodes[w-1:], smooth(data["pg_losses"], w),
            color="purple", linewidth=2)
    ax.set_title("Policy Loss")
    ax.set_xlabel("Episode")
    ax.grid(alpha=0.3)

    ax = axes[2, 1]
    ax.plot(episodes, data["vf_losses"], alpha=0.3, color="teal")
    ax.plot(episodes[w-1:], smooth(data["vf_losses"], w),
            color="teal", linewidth=2)
    ax.set_title("Value Loss")
    ax.set_xlabel("Episode")
    ax.grid(alpha=0.3)

    ax = axes[2, 2]
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

    # ── Row 4: CCTV sightings, hazard rate, outcome breakdown ──

    ax = axes[3, 0]
    ax.plot(episodes, data["cctv_sightings"], alpha=0.2, color="#9b59b6")
    ax.plot(episodes[w-1:], smooth(data["cctv_sightings"], w),
            color="#9b59b6", linewidth=2)
    ax.set_title("CCTV Sightings / Episode")
    ax.set_xlabel("Episode")
    ax.grid(alpha=0.3)

    # Outcome breakdown: stacked area
    ax = axes[3, 1]
    success_sm = smooth(data["successes"], w)
    catch_sm = smooth(data["catches"], w)
    trap_sm = smooth(data["trap_hits"], w)
    timeout_sm = 1.0 - np.array(success_sm) - np.array(catch_sm) - np.array(trap_sm)
    timeout_sm = np.clip(timeout_sm, 0, 1)
    x = episodes[w-1:]
    ax.fill_between(x, 0, np.array(success_sm)*100,
                    alpha=0.5, color='#2ecc71', label='Goal')
    ax.fill_between(x, np.array(success_sm)*100,
                    (np.array(success_sm) + np.array(catch_sm))*100,
                    alpha=0.5, color='#2c3e50', label='Caught')
    ax.fill_between(x,
                    (np.array(success_sm) + np.array(catch_sm))*100,
                    (np.array(success_sm) + np.array(catch_sm) + np.array(trap_sm))*100,
                    alpha=0.5, color='#e67e22', label='Trap')
    ax.set_title("Outcome Breakdown (%)")
    ax.set_xlabel("Episode")
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(alpha=0.3)

    # Combined hazard rate
    ax = axes[3, 2]
    combined = (np.array(data["catches"]) + np.array(data["trap_hits"])
                + np.array(data["traffic_hits"]))
    per_step = combined / np.maximum(np.array(data["lengths"]), 1)
    ax.plot(episodes, per_step, alpha=0.2, color="#8e44ad")
    ax.plot(episodes[w-1:], smooth(per_step, w),
            color="#8e44ad", linewidth=2)
    ax.set_title("Hazard Rate (dangers / step)")
    ax.set_xlabel("Episode")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = log_path.replace(".csv", "_curves.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "logs/stage3_log.csv"
    if os.path.exists(path):
        main(path)
    else:
        print(f"Log not found: {path}")