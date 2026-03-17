"""
Training script for PPO on the GridWorld thief navigation task.

Key design decisions:
─────────────────────────────────────────────────────────────────────
• We collect a full episode (variable length) as one rollout, then
  call agent.update().  This is fine for short episodes (≤200 steps).
  If you later want longer horizons, switch to fixed-length rollouts
  of e.g. 2048 steps regardless of episode boundaries.

• Every EVAL_EVERY episodes we run a greedy (argmax) evaluation to
  track policy quality without exploration noise.

• Metrics are printed and optionally saved to a CSV for plotting.
─────────────────────────────────────────────────────────────────────
"""

import os
import csv
import time
import numpy as np
import torch

from env.grid_world import GridWorld
from agents.ppo_agent import PPOAgent

# ──────────────────────────────────────────────────────────
#  Hyperparameters  (tweak these)
# ──────────────────────────────────────────────────────────

SEED = 42
GRID_SIZE = 15
MAX_STEPS = 200         # max steps per episode

TOTAL_EPISODES = 3000   # total training episodes
EVAL_EVERY = 50         # run greedy eval every N episodes
EVAL_EPISODES = 10      # how many greedy episodes per eval
LOG_EVERY = 50          # print metrics every N episodes
SAVE_EVERY = 500        # checkpoint every N episodes

# ---- visualisation control ----
# Master toggle for GUI rendering. Set False for headless/faster training.
GUI = True
# Show the agent moving live every RENDER_EVERY episodes.
# Set to 1 to watch every episode (slow), 10-50 for a good balance.
RENDER_EVERY = 50
PAUSE = 0.02  # seconds per frame when rendering

# Only import matplotlib stuff if we actually need it
if GUI:
    from utils.visualize import show_grid, reset_view

# PPO hyper-parameters (passed to agent)
PPO_CFG = dict(
    clip_eps=0.2,
    gamma=0.99,
    gae_lambda=0.95,
    vf_coef=0.5,
    vf_clip=10.0,       # value loss clipping — prevents NaN from huge returns
    ent_coef=0.02,       # start with more exploration
    ent_coef_end=0.001,  # decay to less exploration over time
    max_grad_norm=0.5,
    n_epochs=4,
    batch_size=64,
    min_batch_size=8,    # skip tiny tail batches (avoids std() on 1 element)
    lr=3e-4,
    lr_end=3e-5,
    total_updates=TOTAL_EPISODES,
)

# ──────────────────────────────────────────────────────────
#  Setup
# ──────────────────────────────────────────────────────────

np.random.seed(SEED)
torch.manual_seed(SEED)

env = GridWorld(size=GRID_SIZE, max_steps=MAX_STEPS, rng_seed=SEED)
agent = PPOAgent(env.OBS_DIM, 5, **PPO_CFG)

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

log_path = "logs/training_log.csv"
log_file = open(log_path, "w", newline="")
log_writer = csv.writer(log_file)
log_writer.writerow([
    "episode", "ep_reward", "ep_length", "reached_goal",
    "pg_loss", "vf_loss", "entropy", "clip_frac", "lr", "ent_coef",
    "eval_avg_reward", "eval_success_rate",
])


def greedy_eval(env, agent, n_episodes=10):
    """Run n episodes with argmax actions (no sampling)."""
    total_r = 0.0
    successes = 0
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, _ = agent.policy(state_t)
            action = logits.argmax(dim=-1).item()
            state, reward, done = env.step(action)
            ep_r += reward
        total_r += ep_r
        if env.agent_pos == env.goal:
            successes += 1
    return total_r / n_episodes, successes / n_episodes


# ──────────────────────────────────────────────────────────
#  Training loop
# ──────────────────────────────────────────────────────────

recent_rewards = []
recent_lengths = []
recent_successes = []

t_start = time.time()

for ep in range(1, TOTAL_EPISODES + 1):

    state = env.reset()
    done = False
    ep_reward = 0.0

    render_this_ep = GUI and (ep % RENDER_EVERY == 0)
    path = [env.agent_pos] if render_this_ep else None

    while not done:
        action, log_prob, value = agent.select_action(state)
        next_state, reward, done = env.step(action)

        agent.store_transition(state, action, log_prob, reward, value, done)

        state = next_state
        ep_reward += reward

        # ---- live render ----
        if render_this_ep:
            path.append(env.agent_pos)
            show_grid(
                env.render(), env.goal,
                agent_pos=env.agent_pos,
                path=path,
                title=f'Episode {ep}  |  Step {env.steps}  |  R={ep_reward:.0f}',
                pause=PAUSE,
            )

    # PPO update at end of episode
    metrics = agent.update(last_state=state, last_done=done)

    if render_this_ep:
        reset_view()

    reached = env.agent_pos == env.goal
    recent_rewards.append(ep_reward)
    recent_lengths.append(env.steps)
    recent_successes.append(float(reached))

    # --- periodic evaluation ---
    eval_avg_r, eval_sr = "", ""
    if ep % EVAL_EVERY == 0:
        eval_avg_r, eval_sr = greedy_eval(env, agent, EVAL_EPISODES)

    # --- log row ---
    log_writer.writerow([
        ep, f"{ep_reward:.1f}", env.steps, int(reached),
        f"{metrics['pg_loss']:.4f}", f"{metrics['vf_loss']:.4f}",
        f"{metrics['entropy']:.4f}", f"{metrics['clip_frac']:.3f}",
        f"{metrics['lr']:.6f}", f"{metrics['ent_coef']:.5f}",
        eval_avg_r if isinstance(eval_avg_r, str) else f"{eval_avg_r:.1f}",
        eval_sr if isinstance(eval_sr, str) else f"{eval_sr:.2f}",
    ])

    # --- console print ---
    if ep % LOG_EVERY == 0:
        window = min(LOG_EVERY, len(recent_rewards))
        avg_r = np.mean(recent_rewards[-window:])
        avg_len = np.mean(recent_lengths[-window:])
        avg_sr = np.mean(recent_successes[-window:])
        elapsed = time.time() - t_start

        print(
            f"Ep {ep:>5}/{TOTAL_EPISODES} │ "
            f"R={avg_r:>7.1f} │ "
            f"Len={avg_len:>5.1f} │ "
            f"SR={avg_sr:>5.1%} │ "
            f"PG={metrics['pg_loss']:>7.4f} │ "
            f"VF={metrics['vf_loss']:>8.2f} │ "
            f"H={metrics['entropy']:>.3f} │ "
            f"Clip={metrics['clip_frac']:>.2f} │ "
            f"LR={metrics['lr']:.1e} │ "
            f"{elapsed:.0f}s"
        )

    # --- checkpoint ---
    if ep % SAVE_EVERY == 0:
        ckpt_path = f"checkpoints/ppo_ep{ep}.pt"
        agent.save(ckpt_path)
        print(f"  ↳ Saved checkpoint: {ckpt_path}")

log_file.close()

# Final save
agent.save("checkpoints/ppo_final.pt")
print(f"\nTraining complete. Log saved to {log_path}")
print(f"Final checkpoint: checkpoints/ppo_final.pt")