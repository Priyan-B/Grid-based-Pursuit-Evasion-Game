"""
Training script for Stage 2: Traps and Traffic.

IMPORTANT — loads Stage 1 checkpoint to preserve navigation skills.

The obs dimension changes from 10 (stage 1) to 28 (stage 2), so we
can't just do agent.load().  Instead we surgically transplant weights:

    shared.0.weight  (128, 10) → (128, 28)
        Cols [0:9]  (wall patch) → copied to [0:9]   (same position)
        Col  [9]    (distance)   → copied to [27]     (remapped!)
        Cols [9:27] (traps+traffic) → near-zero init  (new channels)
    shared.0.bias    (128,)     → copied directly
    shared.2.*       (128, 128) → copied directly
    policy_head.*    (5, 128)   → copied directly
    value_head.*     (1, 128)   → copied directly

The distance column moves from index 9 to 27 because the trap and
traffic layers are inserted between the wall patch and distance.
The agent starts Phase A already knowing how to navigate, and the
new input channels begin near-zero so they don't disrupt the
existing policy.  The agent just has to learn what traps and traffic
mean on top of its existing skills.

Two training phases, run sequentially:
    Phase A — Static traps (fixed per episode) + traffic cars.
    Phase B — Dynamic traps (teleport every 15 steps) + traffic cars.

Produces:
    checkpoints/stage2_phaseA_final.pt
    checkpoints/stage2_phaseB_final.pt   (← stage 2 deliverable)
    logs/stage2_phaseA_log.csv
    logs/stage2_phaseB_log.csv
"""

import os
import sys
import csv
import time
import numpy as np
import torch

# ── path setup so imports work from project root ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.grid_world_stage2 import GridWorldStage2
from agents.ppo_agent import PPOAgent

# ══════════════════════════════════════════════════════════
#  Hyperparameters
# ══════════════════════════════════════════════════════════

SEED = 42
GRID_SIZE = 15
MAX_STEPS = 200

# ── Stage 1 checkpoint to load from ──
# This should be the final checkpoint from train_ppo.py (stage 1).
# The agent trained on the 10-dim obs with basic navigation.
STAGE1_CHECKPOINT = "checkpoints/ppo_final.pt"

# Phase A — static traps
PHASE_A_EPISODES = 2000

# Phase B — dynamic traps (continues from Phase A)
PHASE_B_EPISODES = 2000

EVAL_EVERY = 50
EVAL_EPISODES = 10
LOG_EVERY = 50
SAVE_EVERY = 500

# ── visualisation (headless-safe) ──
GUI = False            # set True if you have a display
RENDER_EVERY = 50
PAUSE = 0.02

if GUI:
    from utils.visualize_stage2 import show_grid_s2, reset_view_s2

# PPO config — same hypers as stage 1, just different obs dim
PPO_CFG = dict(
    clip_eps=0.2,
    gamma=0.99,
    gae_lambda=0.95,
    vf_coef=0.5,
    vf_clip=10.0,
    ent_coef=0.02,
    ent_coef_end=0.001,
    max_grad_norm=0.5,
    n_epochs=4,
    batch_size=64,
    min_batch_size=8,
    lr=3e-4,
    lr_end=3e-5,
)


# ══════════════════════════════════════════════════════════
#  Stage 1 → Stage 2  weight transplant
# ══════════════════════════════════════════════════════════

def load_stage1_into_stage2(agent, checkpoint_path, old_obs_dim=10):
    """
    Load a Stage 1 checkpoint (obs_dim=10) into a Stage 2 agent
    (obs_dim=28).  Surgically transplants weights:

    Layer layout (PPOPolicy):
        shared.0  = Linear(state_size → 128)   ← INPUT LAYER, size changes
        shared.2  = Linear(128 → 128)          ← same shape
        policy_head = Linear(128 → 5)          ← same shape
        value_head  = Linear(128 → 1)          ← same shape

    Observation layout change:
        Stage 1: [wall_patch(0:9), distance(9)]                          = 10
        Stage 2: [wall_patch(0:9), traps(9:18), traffic(18:27), dist(27)] = 28

    The distance feature moved from column 9 to column 27.  A naive copy
    of the first 10 columns would put the distance weight at column 9,
    which in stage 2 is the first trap channel — WRONG.

    Correct column mapping for shared.0.weight:
        Stage 1 cols [0:9]  → Stage 2 cols [0:9]   (wall patch — direct)
        Stage 1 col  [9]    → Stage 2 col  [27]    (distance  — remapped)
        Stage 2 cols [9:27] → near-zero init        (traps + traffic — new)

    The optimizer is freshly initialised (can't reuse stage 1's Adam
    state because the parameter shape changed for the first layer).
    """
    print(f"\n  Loading Stage 1 checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, weights_only=True)
    s1_state = ckpt["policy_state"]

    s2_state = agent.policy.state_dict()

    # ── transplant each parameter ──
    transferred = 0
    for key in s2_state:
        if key == "shared.0.weight":
            # Stage 1: (128, 10)  →  Stage 2: (128, 28)
            s1_w = s1_state[key]                       # (128, 10)
            s2_w = s2_state[key]                        # (128, 28)
            old_in = s1_w.shape[1]                      # 10
            new_in = s2_w.shape[1]                      # 28
            assert old_in == old_obs_dim, (
                f"Stage 1 input dim mismatch: expected {old_obs_dim}, "
                f"got {old_in}"
            )

            # Start with near-zero everywhere (new channels)
            torch.nn.init.normal_(s2_w, mean=0.0, std=0.01)

            # Copy wall patch weights: s1 cols [0:9] → s2 cols [0:9]
            s2_w[:, :9] = s1_w[:, :9]

            # Copy distance weight: s1 col [9] → s2 col [27]
            s2_w[:, 27] = s1_w[:, 9]

            # Columns 9-26 (traps + traffic) stay near-zero

            s2_state[key] = s2_w
            print(f"    {key}: transplanted ({s1_w.shape} → {s2_w.shape})")
            print(f"      wall patch cols [0:9]  → [0:9]  (direct copy)")
            print(f"      distance  col   [9]    → [27]   (remapped)")
            print(f"      trap+traffic    [9:27] → near-zero init")

        elif key == "shared.0.bias":
            # Same size (128,), copy directly
            s2_state[key] = s1_state[key]
            print(f"    {key}: copied directly {s1_state[key].shape}")

        else:
            # shared.2.*, policy_head.*, value_head.* — identical shapes
            assert s1_state[key].shape == s2_state[key].shape, (
                f"Shape mismatch on {key}: "
                f"stage1={s1_state[key].shape} vs stage2={s2_state[key].shape}"
            )
            s2_state[key] = s1_state[key]
            print(f"    {key}: copied directly {s1_state[key].shape}")

        transferred += 1

    agent.policy.load_state_dict(s2_state)
    print(f"  ✓ Transplanted {transferred} parameters from Stage 1")
    print(f"  ✓ Optimizer reset (fresh Adam state for new architecture)\n")

    # NOTE: we intentionally do NOT load optimizer state — the Adam
    # moments for shared.0.weight have the wrong shape, and even for
    # matching layers, starting fresh is cleaner since the loss
    # landscape changes with the new observation channels.

    return agent


# ══════════════════════════════════════════════════════════
#  Greedy evaluation (same logic as stage 1)
# ══════════════════════════════════════════════════════════

def greedy_eval(env, agent, n_episodes=10):
    total_r = 0.0
    successes = 0
    total_trap_hits = 0
    total_traffic_hits = 0
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
        total_trap_hits += env.trap_hits
        total_traffic_hits += env.traffic_hits
    n = max(n_episodes, 1)
    return (total_r / n, successes / n,
            total_trap_hits / n, total_traffic_hits / n)


# ══════════════════════════════════════════════════════════
#  Single-phase training loop (reused for A and B)
# ══════════════════════════════════════════════════════════

def train_phase(env, agent, total_episodes, log_path, ckpt_prefix,
                phase_label="A"):
    """
    Trains for `total_episodes`, writing CSV logs and periodic checkpoints.
    Returns the agent (same object, mutated in place).
    """

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "episode", "ep_reward", "ep_length", "reached_goal",
        "trap_hits", "traffic_hits",
        "pg_loss", "vf_loss", "entropy", "clip_frac", "lr", "ent_coef",
        "eval_avg_reward", "eval_success_rate",
        "eval_avg_trap_hits", "eval_avg_traffic_hits",
    ])

    recent_rewards = []
    recent_lengths = []
    recent_successes = []
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
                show_grid_s2(
                    env.render(), env.goal,
                    agent_pos=env.agent_pos,
                    path=path,
                    traps=env.traps,
                    traffic_positions=traffic_pos,
                    title=(f'Phase {phase_label} Ep {ep} | '
                           f'Step {env.steps} | R={ep_reward:.0f}'),
                    pause=PAUSE,
                )

        # PPO update
        metrics = agent.update(last_state=state, last_done=done)

        if render_this:
            reset_view_s2()

        reached = env.agent_pos == env.goal
        recent_rewards.append(ep_reward)
        recent_lengths.append(env.steps)
        recent_successes.append(float(reached))

        # ── periodic evaluation ──
        eval_avg_r, eval_sr = "", ""
        eval_trap, eval_traffic = "", ""
        if ep % EVAL_EVERY == 0:
            eval_avg_r, eval_sr, eval_trap, eval_traffic = greedy_eval(
                env, agent, EVAL_EPISODES
            )

        # ── CSV row ──
        log_writer.writerow([
            ep, f"{ep_reward:.1f}", env.steps, int(reached),
            env.trap_hits, env.traffic_hits,
            f"{metrics['pg_loss']:.4f}", f"{metrics['vf_loss']:.4f}",
            f"{metrics['entropy']:.4f}", f"{metrics['clip_frac']:.3f}",
            f"{metrics['lr']:.6f}", f"{metrics['ent_coef']:.5f}",
            eval_avg_r if isinstance(eval_avg_r, str) else f"{eval_avg_r:.1f}",
            eval_sr if isinstance(eval_sr, str) else f"{eval_sr:.2f}",
            eval_trap if isinstance(eval_trap, str) else f"{eval_trap:.1f}",
            eval_traffic if isinstance(eval_traffic, str) else f"{eval_traffic:.1f}",
        ])

        # ── console ──
        if ep % LOG_EVERY == 0:
            window = min(LOG_EVERY, len(recent_rewards))
            avg_r = np.mean(recent_rewards[-window:])
            avg_len = np.mean(recent_lengths[-window:])
            avg_sr = np.mean(recent_successes[-window:])
            elapsed = time.time() - t_start

            print(
                f"[Phase {phase_label}] "
                f"Ep {ep:>5}/{total_episodes} │ "
                f"R={avg_r:>7.1f} │ "
                f"Len={avg_len:>5.1f} │ "
                f"SR={avg_sr:>5.1%} │ "
                f"Traps={env.trap_hits} │ "
                f"Traffic={env.traffic_hits} │ "
                f"PG={metrics['pg_loss']:>7.4f} │ "
                f"VF={metrics['vf_loss']:>8.2f} │ "
                f"H={metrics['entropy']:.3f} │ "
                f"Clip={metrics['clip_frac']:.2f} │ "
                f"LR={metrics['lr']:.1e} │ "
                f"{elapsed:.0f}s"
            )

        # ── checkpoint ──
        if ep % SAVE_EVERY == 0:
            p = f"checkpoints/{ckpt_prefix}_ep{ep}.pt"
            agent.save(p)
            print(f"  ↳ Saved checkpoint: {p}")

    log_file.close()

    # Final checkpoint
    final_path = f"checkpoints/{ckpt_prefix}_final.pt"
    agent.save(final_path)
    print(f"Phase {phase_label} done.  Log: {log_path}  Ckpt: {final_path}\n")
    return agent


# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ── Verify stage 1 checkpoint exists ──
    if not os.path.exists(STAGE1_CHECKPOINT):
        print(f"ERROR: Stage 1 checkpoint not found at '{STAGE1_CHECKPOINT}'")
        print(f"Run train_ppo.py (stage 1) first, or update STAGE1_CHECKPOINT path.")
        sys.exit(1)

    # ─────────────── Phase A: static traps + traffic ───────────────
    print("=" * 65)
    print("  STAGE 2  —  PHASE A : Static Traps + Traffic")
    print("=" * 65)

    env_a = GridWorldStage2(
        size=GRID_SIZE, max_steps=MAX_STEPS, rng_seed=SEED,
        trap_mode="static",
    )

    # Create agent with stage 2 obs dim, then transplant stage 1 weights
    agent = PPOAgent(
        env_a.OBS_DIM, 5,
        **PPO_CFG,
        total_updates=PHASE_A_EPISODES,
    )
    agent = load_stage1_into_stage2(agent, STAGE1_CHECKPOINT, old_obs_dim=10)

    # Quick sanity check: run one greedy eval before training starts
    # to see how much navigation skill survived the transplant
    pre_r, pre_sr, _, _ = greedy_eval(env_a, agent, EVAL_EPISODES)
    print(f"  Pre-training eval (transplanted weights):")
    print(f"    Avg reward: {pre_r:.1f}  |  Success rate: {pre_sr:.0%}")
    print()

    agent = train_phase(
        env_a, agent,
        total_episodes=PHASE_A_EPISODES,
        log_path="logs/stage2_phaseA_log.csv",
        ckpt_prefix="stage2_phaseA",
        phase_label="A",
    )

    # ─────────────── Phase B: dynamic traps + traffic ──────────────
    print("=" * 65)
    print("  STAGE 2  —  PHASE B : Dynamic Traps + Traffic")
    print("=" * 65)

    env_b = GridWorldStage2(
        size=GRID_SIZE, max_steps=MAX_STEPS, rng_seed=SEED,
        trap_mode="dynamic",
    )

    # Continue from Phase A checkpoint — same agent, reset annealing
    # schedule for Phase B (fresh LR / entropy ramp over B episodes).
    agent._update_count = 0
    agent.total_updates = PHASE_B_EPISODES
    agent.lr_start = PPO_CFG["lr"]
    agent.lr_end = PPO_CFG["lr_end"]
    agent.ent_coef_start = PPO_CFG["ent_coef"]
    agent.ent_coef_end = PPO_CFG["ent_coef_end"]

    agent = train_phase(
        env_b, agent,
        total_episodes=PHASE_B_EPISODES,
        log_path="logs/stage2_phaseB_log.csv",
        ckpt_prefix="stage2_phaseB",
        phase_label="B",
    )

    print("Stage 2 training complete.")
    print("  Phase A log : logs/stage2_phaseA_log.csv")
    print("  Phase B log : logs/stage2_phaseB_log.csv")
    print("  Final ckpt  : checkpoints/stage2_phaseB_final.pt")