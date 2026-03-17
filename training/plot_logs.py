"""
Plot training curves from the CSV log produced by train_ppo.py.

Usage:  python plot_logs.py              (reads logs/training_log.csv)
        python plot_logs.py path/to.csv  (reads custom path)
"""

import sys
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless — saves to file
import matplotlib.pyplot as plt


def smooth(arr, window=50):
    """Simple moving average."""
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='valid')


def main(log_path="logs/training_log.csv"):
    episodes, rewards, lengths = [], [], []
    successes, pg_losses, vf_losses = [], [], []
    entropies, clip_fracs = [], []
    eval_eps, eval_rewards, eval_srs = [], [], []

    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["ep_reward"]))
            lengths.append(int(row["ep_length"]))
            successes.append(int(row["reached_goal"]))
            pg_losses.append(float(row["pg_loss"]))
            vf_losses.append(float(row["vf_loss"]))
            entropies.append(float(row["entropy"]))
            clip_fracs.append(float(row["clip_frac"]))
            if row["eval_avg_reward"]:
                eval_eps.append(int(row["episode"]))
                eval_rewards.append(float(row["eval_avg_reward"]))
                eval_srs.append(float(row["eval_success_rate"]))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("PPO GridWorld Training Curves", fontsize=16, fontweight="bold")

    w = 50  # smoothing window

    # 1. Episode reward
    ax = axes[0, 0]
    ax.plot(episodes, rewards, alpha=0.15, color="steelblue")
    sm = smooth(rewards, w)
    ax.plot(episodes[w - 1:], sm, color="steelblue", linewidth=2)
    if eval_eps:
        ax.plot(eval_eps, eval_rewards, 'o-', color="crimson",
                markersize=4, label="Greedy eval")
        ax.legend()
    ax.set_title("Episode Reward")
    ax.set_xlabel("Episode")
    ax.grid(alpha=0.3)

    # 2. Success rate (rolling)
    ax = axes[0, 1]
    sr_smooth = smooth(successes, w)
    ax.plot(episodes[w - 1:], sr_smooth * 100, color="seagreen", linewidth=2)
    if eval_eps:
        ax.plot(eval_eps, [s * 100 for s in eval_srs], 'o-',
                color="crimson", markersize=4, label="Greedy eval")
        ax.legend()
    ax.set_title("Success Rate (%)")
    ax.set_xlabel("Episode")
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3)

    # 3. Episode length
    ax = axes[0, 2]
    ax.plot(episodes, lengths, alpha=0.15, color="darkorange")
    ax.plot(episodes[w - 1:], smooth(lengths, w),
            color="darkorange", linewidth=2)
    ax.set_title("Episode Length")
    ax.set_xlabel("Episode")
    ax.grid(alpha=0.3)

    # 4. Policy loss
    ax = axes[1, 0]
    ax.plot(episodes, pg_losses, alpha=0.3, color="purple")
    ax.plot(episodes[w - 1:], smooth(pg_losses, w),
            color="purple", linewidth=2)
    ax.set_title("Policy Loss")
    ax.set_xlabel("Episode")
    ax.grid(alpha=0.3)

    # 5. Value loss
    ax = axes[1, 1]
    ax.plot(episodes, vf_losses, alpha=0.3, color="teal")
    ax.plot(episodes[w - 1:], smooth(vf_losses, w),
            color="teal", linewidth=2)
    ax.set_title("Value Loss")
    ax.set_xlabel("Episode")
    ax.grid(alpha=0.3)

    # 6. Entropy + clip fraction
    ax = axes[1, 2]
    ax.plot(episodes, entropies, alpha=0.3, color="coral")
    ax.plot(episodes[w - 1:], smooth(entropies, w),
            color="coral", linewidth=2, label="Entropy")
    ax2 = ax.twinx()
    ax2.plot(episodes, clip_fracs, alpha=0.3, color="gray")
    ax2.plot(episodes[w - 1:], smooth(clip_fracs, w),
             color="gray", linewidth=2, label="Clip frac")
    ax.set_title("Entropy & Clip Fraction")
    ax.set_xlabel("Episode")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = "logs/training_curves.png"
    plt.savefig(out, dpi=150)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "logs/training_log.csv"
    main(path)