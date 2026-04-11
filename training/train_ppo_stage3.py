"""
Training script for Stage 3: Police + CCTV.

Loads Stage 2 checkpoint (28-dim obs) into Stage 3 agent (40-dim obs).

Observation layout — first 28 dims unchanged from stage 2:
    Stage 2: [walls(9), traps(9), traffic(9), dist(1)]           = 28
    Stage 3: [walls(9), traps(9), traffic(9), dist(1),
              police_overlay(9), police_dists(3)]                 = 40

    Cols [0:28]  → direct copy from stage 2  (no remapping needed!)
    Cols [28:40] → near-zero init  (new police channels)

Thief trains against 3 random-walking police cars.
CCTV runs silently in background, collecting sighting data.

Produces:
    checkpoints/stage3_final.pt
    logs/stage3_log.csv
    logs/cctv_data.pkl         (CCTV sightings for stage 4)
"""

import os
import sys
import csv
import time
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.grid_world_stage3 import GridWorldStage3
from agents.ppo_agent import PPOAgent

# ══════════════════════════════════════════════════════════
#  Hyperparameters
# ══════════════════════════════════════════════════════════

SEED = 42
GRID_SIZE = 15
MAX_STEPS = 200

STAGE2_CHECKPOINT = "checkpoints/stage2_phaseB_final.pt"

TOTAL_EPISODES = 200000

EVAL_EVERY = 5000
EVAL_EPISODES = 5
LOG_EVERY = 5000
SAVE_EVERY = 10000

GUI = False
RENDER_EVERY = 50
PAUSE = 0.02

if GUI:
    from utils.visualize_stage3 import show_grid_s3, reset_view_s3

PPO_CFG = dict(
    clip_eps=0.2,
    gamma=0.99,
    gae_lambda=0.95,
    vf_coef=0.5,
    vf_clip=3.0,
    ent_coef=0.05,
    ent_coef_end=0.02,
    max_grad_norm=0.5,
    n_epochs=4,
    batch_size=64,
    min_batch_size=8,
    lr=3e-4,
    lr_end=3e-5,
)


# ══════════════════════════════════════════════════════════
#  Stage 2 → Stage 3  weight transplant
# ══════════════════════════════════════════════════════════

def load_stage2_into_stage3(agent, checkpoint_path, old_obs_dim=28):
    """
    Load Stage 2 checkpoint (obs=28) into Stage 3 agent (obs=40).

    First 28 dims are identical between stages — no remapping needed.
    Just copy first 28 columns of the input layer, zero-init the
    remaining 12 columns (police overlay + police distances).

    All other layers (shared.2, policy_head, value_head) are the
    same shape — copied directly.
    """
    print(f"\n  Loading Stage 2 checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, weights_only=True)
    s2_state = ckpt["policy_state"]

    s3_state = agent.policy.state_dict()

    transferred = 0
    for key in s3_state:
        if key == "shared.0.weight":
            s2_w = s2_state[key]                # (128, 28)
            s3_w = s3_state[key]                 # (128, 40)
            old_in = s2_w.shape[1]

            assert old_in == old_obs_dim, (
                f"Stage 2 input dim mismatch: expected {old_obs_dim}, "
                f"got {old_in}"
            )

            # Near-zero everywhere first
            torch.nn.init.normal_(s3_w, mean=0.0, std=0.01)

            # Copy all 28 stage 2 columns directly — no remapping
            s3_w[:, :old_in] = s2_w

            s3_state[key] = s3_w
            print(f"    {key}: transplanted ({s2_w.shape} → {s3_w.shape})")
            print(f"      stage 2 cols [0:28] → [0:28]  (direct copy)")
            print(f"      police cols  [28:40] → near-zero init")

        elif key == "shared.0.bias":
            s3_state[key] = s2_state[key]
            print(f"    {key}: copied directly {s2_state[key].shape}")

        else:
            assert s2_state[key].shape == s3_state[key].shape, (
                f"Shape mismatch on {key}: "
                f"s2={s2_state[key].shape} vs s3={s3_state[key].shape}"
            )
            s3_state[key] = s2_state[key]
            print(f"    {key}: copied directly {s2_state[key].shape}")

        transferred += 1

    agent.policy.load_state_dict(s3_state)
    print(f"  ✓ Transplanted {transferred} parameters from Stage 2")
    print(f"  ✓ Optimizer reset (fresh Adam state)\n")
    return agent


# ══════════════════════════════════════════════════════════
#  Greedy evaluation
# ══════════════════════════════════════════════════════════

def greedy_eval(env, agent, n_episodes=10):
    total_r = 0.0
    successes = 0
    catches = 0
    trap_deaths = 0
    total_traffic = 0
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
        if env.caught_by_police:
            catches += 1
        trap_deaths += env.trap_hits
        total_traffic += env.traffic_hits
    n = max(n_episodes, 1)
    return (total_r / n, successes / n, catches / n,
            trap_deaths / n, total_traffic / n)


# ══════════════════════════════════════════════════════════
#  Training loop
# ══════════════════════════════════════════════════════════

def train(env, agent, total_episodes, log_path):

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "episode", "ep_reward", "ep_length", "reached_goal",
        "caught_by_police", "trap_hits", "traffic_hits", "cctv_sightings",
        "pg_loss", "vf_loss", "entropy", "clip_frac", "lr", "ent_coef",
        "eval_avg_reward", "eval_success_rate", "eval_catch_rate",
        "eval_avg_trap_deaths", "eval_avg_traffic_hits",
    ])

    recent_rewards = []
    recent_lengths = []
    recent_successes = []
    recent_catches = []
    t_start = time.time()

    for ep in range(1, total_episodes + 1):

        state = env.reset()
        done = False
        ep_reward = 0.0

        render_this = GUI and (ep % RENDER_EVERY == 0)
        path = [env.agent_pos] if render_this else None

        while not done:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, log_prob, reward, value, done)
            state = next_state
            ep_reward += reward

            if render_this:
                path.append(env.agent_pos)
                traffic_pos = [car.pos for car in env.traffic_cars]
                police_pos = [pc.pos for pc in env.police_cars]
                show_grid_s3(
                    env.render(), env.goal,
                    agent_pos=env.agent_pos,
                    path=path,
                    traps=env.traps,
                    traffic_positions=traffic_pos,
                    police_positions=police_pos,
                    cctv_cells=env.cctv_cells,
                    title=(f'Stage 3 Ep {ep} | '
                           f'Step {env.steps} | R={ep_reward:.0f}'),
                    pause=PAUSE,
                )

        metrics = agent.update(last_state=state, last_done=done)

        if render_this:
            reset_view_s3()

        reached = env.agent_pos == env.goal
        recent_rewards.append(ep_reward)
        recent_lengths.append(env.steps)
        recent_successes.append(float(reached))
        recent_catches.append(float(env.caught_by_police))

        # ── eval ──
        eval_r, eval_sr, eval_cr = "", "", ""
        eval_trap, eval_traffic = "", ""
        if ep % EVAL_EVERY == 0:
            eval_r, eval_sr, eval_cr, eval_trap, eval_traffic = greedy_eval(
                env, agent, EVAL_EPISODES
            )

        # ── CSV ──
        n_sightings = len(env.cctv_log)
        log_writer.writerow([
            ep, f"{ep_reward:.1f}", env.steps, int(reached),
            int(env.caught_by_police), env.trap_hits, env.traffic_hits,
            n_sightings,
            f"{metrics['pg_loss']:.4f}", f"{metrics['vf_loss']:.4f}",
            f"{metrics['entropy']:.4f}", f"{metrics['clip_frac']:.3f}",
            f"{metrics['lr']:.6f}", f"{metrics['ent_coef']:.5f}",
            eval_r if isinstance(eval_r, str) else f"{eval_r:.1f}",
            eval_sr if isinstance(eval_sr, str) else f"{eval_sr:.2f}",
            eval_cr if isinstance(eval_cr, str) else f"{eval_cr:.2f}",
            eval_trap if isinstance(eval_trap, str) else f"{eval_trap:.1f}",
            eval_traffic if isinstance(eval_traffic, str) else f"{eval_traffic:.1f}",
        ])

        # ── console ──
        if ep % LOG_EVERY == 0:
            window = min(LOG_EVERY, len(recent_rewards))
            avg_r = np.mean(recent_rewards[-window:])
            avg_len = np.mean(recent_lengths[-window:])
            avg_sr = np.mean(recent_successes[-window:])
            avg_cr = np.mean(recent_catches[-window:])
            elapsed = time.time() - t_start

            print(
                f"Ep {ep:>6}/{total_episodes} │ "
                f"R={avg_r:>7.1f} │ "
                f"Len={avg_len:>5.1f} │ "
                f"SR={avg_sr:>5.1%} │ "
                f"Caught={avg_cr:>5.1%} │ "
                f"Traps={env.trap_hits} │ "
                f"PG={metrics['pg_loss']:>7.4f} │ "
                f"VF={metrics['vf_loss']:>8.2f} │ "
                f"H={metrics['entropy']:.3f} │ "
                f"LR={metrics['lr']:.1e} │ "
                f"{elapsed:.0f}s"
            )

        # ── checkpoint ──
        if ep % SAVE_EVERY == 0:
            p = f"checkpoints/stage3_ep{ep}.pt"
            agent.save(p)
            print(f"  ↳ Saved checkpoint: {p}")

    log_file.close()

    # Final saves
    agent.save("checkpoints/stage3_final.pt")
    env.save_cctv_data("logs/cctv_data.pkl")
    print(f"\nStage 3 training complete.")
    print(f"  Log:       {log_path}")
    print(f"  Checkpoint: checkpoints/stage3_final.pt")
    print(f"  CCTV data:  logs/cctv_data.pkl")
    total_sightings = sum(len(ep) for ep in env.cctv_all_episodes)
    print(f"  CCTV total sightings across all episodes: {total_sightings}")


# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if not os.path.exists(STAGE2_CHECKPOINT):
        print(f"ERROR: Stage 2 checkpoint not found at '{STAGE2_CHECKPOINT}'")
        print(f"Run train_ppo_stage2.py first, or update STAGE2_CHECKPOINT path.")
        sys.exit(1)

    print("=" * 65)
    print("  STAGE 3  —  Police + CCTV")
    print("=" * 65)

    env = GridWorldStage3(
        size=GRID_SIZE, max_steps=MAX_STEPS, rng_seed=SEED,
    )

    print(f"  Grid: {GRID_SIZE}×{GRID_SIZE}")
    print(f"  CCTV cameras at: {sorted(env.cctv_cells)}")
    print(f"  Police cars: {env.n_police}")
    print(f"  Obs dim: {env.OBS_DIM}")

    agent = PPOAgent(
        env.OBS_DIM, 5,
        **PPO_CFG,
        total_updates=TOTAL_EPISODES,
    )
    agent = load_stage2_into_stage3(agent, STAGE2_CHECKPOINT, old_obs_dim=28)

    # Pre-training sanity check
    pre_r, pre_sr, pre_cr, _, _ = greedy_eval(env, agent, EVAL_EPISODES)
    print(f"  Pre-training eval (transplanted weights):")
    print(f"    Avg reward: {pre_r:.1f}  |  Success: {pre_sr:.0%}  |  Caught: {pre_cr:.0%}")
    print()

    train(
        env, agent,
        total_episodes=TOTAL_EPISODES,
        log_path="logs/stage3_log.csv",
    )