"""
Training script for Stage 4: Police learn to catch frozen thief.

Loads frozen thief from Stage 3 checkpoint.
Creates 2 independent PPO agents for police.
Each police has its own buffer, network, optimizer, and update cycle.
Both train simultaneously against the same frozen thief.

Produces:
    checkpoints/stage4_police0_final.pt
    checkpoints/stage4_police1_final.pt
    logs/stage4_log.csv
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

from env.grid_world_stage4 import GridWorldStage4
from agents.ppo_agent import PPOAgent, PPOPolicy

# ══════════════════════════════════════════════════════════
#  Hyperparameters
# ══════════════════════════════════════════════════════════

SEED = 42
GRID_SIZE = 15
MAX_STEPS = 200

THIEF_CHECKPOINT = "checkpoints/stage3_final.pt"
THIEF_OBS_DIM = 40    # stage 3 observation size

TOTAL_EPISODES = 500000

EVAL_EVERY = 5000
EVAL_EPISODES = 10
LOG_EVERY = 5000
SAVE_EVERY = 10000

GUI = False
RENDER_EVERY = 50
PAUSE = 0.02

if GUI:
    from utils.visualize_stage3 import show_grid_s3, reset_view_s3

# PPO config for police — larger hidden layer for 235-dim input
POLICE_PPO_CFG = dict(
    clip_eps=0.2,
    gamma=0.99,
    gae_lambda=0.95,
    vf_coef=0.5,
    vf_clip=3.0,
    ent_coef=0.05,       # more exploration — police start clueless
    ent_coef_end=0.02,
    max_grad_norm=0.5,
    n_epochs=4,
    batch_size=64,
    min_batch_size=8,
    lr=3e-4,
    lr_end=3e-5,
)

POLICE_HIDDEN = 256   # bigger network for 235-dim obs


# ══════════════════════════════════════════════════════════
#  Load frozen thief
# ══════════════════════════════════════════════════════════

def load_frozen_thief(checkpoint_path, obs_dim=40, n_actions=5):
    """Load thief policy from Stage 3 checkpoint. Returns the policy network."""
    print(f"  Loading frozen thief: {checkpoint_path}")
    policy = PPOPolicy(obs_dim, n_actions, hidden=128)
    ckpt = torch.load(checkpoint_path, weights_only=True)
    policy.load_state_dict(ckpt["policy_state"])
    policy.eval()
    print(f"  ✓ Thief policy loaded (obs={obs_dim}, frozen)\n")
    return policy


# ══════════════════════════════════════════════════════════
#  Greedy evaluation
# ══════════════════════════════════════════════════════════

def greedy_eval(env, agents, n_episodes=10):
    """Run greedy evaluation for police agents."""
    catches = 0
    escapes = 0
    trap_deaths = 0
    timeouts = 0
    total_steps = 0

    for _ in range(n_episodes):
        obs_list = env.reset()
        done = False
        while not done:
            actions = []
            for i, agent in enumerate(agents):
                state_t = torch.tensor(obs_list[i], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = agent.policy(state_t)
                actions.append(logits.argmax(dim=-1).item())

            obs_list, rewards, done = env.step(actions)

        total_steps += env.steps
        if env.caught_by_police:
            catches += 1
        elif env.agent_pos == env.goal:
            escapes += 1
        elif env.trap_hits > 0:
            trap_deaths += 1
        else:
            timeouts += 1

    n = max(n_episodes, 1)
    return dict(
        catch_rate=catches / n,
        escape_rate=escapes / n,
        trap_rate=trap_deaths / n,
        timeout_rate=timeouts / n,
        avg_steps=total_steps / n,
    )


# ══════════════════════════════════════════════════════════
#  Training loop
# ══════════════════════════════════════════════════════════

def train(env, agents, total_episodes, log_path):

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "episode", "ep_length", "outcome",
        "catcher_idx", "had_cctv_sighting",
        "police0_reward", "police1_reward",
        "police0_pg_loss", "police0_vf_loss", "police0_entropy",
        "police1_pg_loss", "police1_vf_loss", "police1_entropy",
        "lr",
        "eval_catch_rate", "eval_escape_rate",
        "eval_trap_rate", "eval_timeout_rate", "eval_avg_steps",
    ])

    recent_catches = []
    recent_escapes = []
    t_start = time.time()

    for ep in range(1, total_episodes + 1):

        obs_list = env.reset()
        done = False
        ep_rewards = [0.0] * env.n_police

        render_this = GUI and (ep % RENDER_EVERY == 0)
        path = [env.agent_pos] if render_this else None

        while not done:
            # Get actions from both police
            actions = []
            log_probs = []
            values = []
            for i, agent in enumerate(agents):
                action, lp, val = agent.select_action(obs_list[i])
                actions.append(action)
                log_probs.append(lp)
                values.append(val)

            # Step environment
            next_obs_list, rewards, done = env.step(actions)

            # Store transitions for each police agent
            for i, agent in enumerate(agents):
                agent.store_transition(
                    obs_list[i], actions[i], log_probs[i],
                    rewards[i], values[i], done,
                )
                ep_rewards[i] += rewards[i]

            obs_list = next_obs_list

            # Render
            if render_this:
                path.append(env.agent_pos)
                traffic_pos = [car.pos for car in env.traffic_cars]
                show_grid_s3(
                    env.render(), env.goal,
                    agent_pos=env.agent_pos,
                    path=path,
                    traps=env.traps,
                    traffic_positions=traffic_pos,
                    police_positions=env.police_positions,
                    cctv_cells=env.cctv_cells,
                    title=(f'Stage 4 Ep {ep} | Step {env.steps} | '
                           f'R0={ep_rewards[0]:.0f} R1={ep_rewards[1]:.0f}'),
                    pause=PAUSE,
                )

        # PPO update for each police agent
        all_metrics = []
        for i, agent in enumerate(agents):
            metrics = agent.update(
                last_state=obs_list[i], last_done=done,
            )
            all_metrics.append(metrics)

        if render_this:
            reset_view_s3()

        # Determine outcome
        if env.caught_by_police:
            outcome = "caught"
        elif env.agent_pos == env.goal:
            outcome = "escaped"
        elif env.trap_hits > 0:
            outcome = "trap"
        else:
            outcome = "timeout"

        recent_catches.append(1.0 if outcome == "caught" else 0.0)
        recent_escapes.append(1.0 if outcome == "escaped" else 0.0)

        # ── eval ──
        eval_data = {}
        if ep % EVAL_EVERY == 0:
            eval_data = greedy_eval(env, agents, EVAL_EPISODES)

        # ── CSV ──
        had_sighting = 1 if env.last_cctv_sighting is not None else 0
        m0, m1 = all_metrics[0], all_metrics[1]
        log_writer.writerow([
            ep, env.steps, outcome,
            env.catcher_idx, had_sighting,
            f"{ep_rewards[0]:.1f}", f"{ep_rewards[1]:.1f}",
            f"{m0['pg_loss']:.4f}", f"{m0['vf_loss']:.4f}", f"{m0['entropy']:.4f}",
            f"{m1['pg_loss']:.4f}", f"{m1['vf_loss']:.4f}", f"{m1['entropy']:.4f}",
            f"{m0['lr']:.6f}",
            eval_data.get("catch_rate", ""),
            eval_data.get("escape_rate", ""),
            eval_data.get("trap_rate", ""),
            eval_data.get("timeout_rate", ""),
            eval_data.get("avg_steps", ""),
        ])

        # ── console ──
        if ep % LOG_EVERY == 0:
            window = min(LOG_EVERY, len(recent_catches))
            catch_rate = np.mean(recent_catches[-window:])
            escape_rate = np.mean(recent_escapes[-window:])
            elapsed = time.time() - t_start

            print(
                f"Ep {ep:>6}/{total_episodes} │ "
                f"Catch={catch_rate:>5.1%} │ "
                f"Escape={escape_rate:>5.1%} │ "
                f"Len={env.steps:>3d} │ "
                f"CCTV={'Y' if had_sighting else 'N'} │ "
                f"P0: PG={m0['pg_loss']:>7.4f} VF={m0['vf_loss']:>6.2f} "
                f"H={m0['entropy']:.3f} │ "
                f"P1: PG={m1['pg_loss']:>7.4f} VF={m1['vf_loss']:>6.2f} "
                f"H={m1['entropy']:.3f} │ "
                f"LR={m0['lr']:.1e} │ "
                f"{elapsed:.0f}s"
            )

        # ── checkpoint ──
        if ep % SAVE_EVERY == 0:
            for i, agent in enumerate(agents):
                p = f"checkpoints/stage4_police{i}_ep{ep}.pt"
                agent.save(p)
            print(f"  ↳ Saved police checkpoints (ep {ep})")

    log_file.close()

    # Final saves
    for i, agent in enumerate(agents):
        p = f"checkpoints/stage4_police{i}_final.pt"
        agent.save(p)
        print(f"  Saved: {p}")

    print(f"\nStage 4 training complete.")
    print(f"  Log: {log_path}")


# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if not os.path.exists(THIEF_CHECKPOINT):
        print(f"ERROR: Thief checkpoint not found at '{THIEF_CHECKPOINT}'")
        print(f"Run train_ppo_stage3.py first.")
        sys.exit(1)

    print("=" * 65)
    print("  STAGE 4  —  Police Training (Frozen Thief)")
    print("=" * 65)

    # Load frozen thief
    thief_policy = load_frozen_thief(THIEF_CHECKPOINT, obs_dim=THIEF_OBS_DIM)

    # Create environment
    env = GridWorldStage4(
        thief_policy=thief_policy,
        size=GRID_SIZE, max_steps=MAX_STEPS, rng_seed=SEED,
    )

    print(f"  Police obs dim: {env.POLICE_OBS_DIM}")
    print(f"  Police count:   {env.n_police}")
    print(f"  Hidden size:    {POLICE_HIDDEN}")
    print(f"  CCTV cameras:   {sorted(env.cctv_cells)}")
    print()

    # Create 2 independent police PPO agents
    police_agents = []
    for i in range(env.n_police):
        agent = PPOAgent(
            env.POLICE_OBS_DIM, 5,
            **POLICE_PPO_CFG,
            total_updates=TOTAL_EPISODES,
        )
        # Replace default 128-hidden network with 256-hidden
        agent.policy = PPOPolicy(env.POLICE_OBS_DIM, 5, hidden=POLICE_HIDDEN)
        agent.optimizer = torch.optim.Adam(
            agent.policy.parameters(), lr=POLICE_PPO_CFG["lr"], eps=1e-5,
        )
        police_agents.append(agent)
        print(f"  Police {i}: PPOPolicy({env.POLICE_OBS_DIM} → {POLICE_HIDDEN} → 5)")

    # Pre-training eval
    print()
    eval_data = greedy_eval(env, police_agents, EVAL_EPISODES)
    print(f"  Pre-training eval:")
    print(f"    Catch: {eval_data['catch_rate']:.0%}  "
          f"Escape: {eval_data['escape_rate']:.0%}  "
          f"Trap: {eval_data['trap_rate']:.0%}  "
          f"Timeout: {eval_data['timeout_rate']:.0%}")
    print()

    train(
        env, police_agents,
        total_episodes=TOTAL_EPISODES,
        log_path="logs/stage4_log.csv",
    )