"""
Training script for Stage 5: Adversarial Co-Training.

Loads thief from Stage 3 checkpoint (40-dim obs, 128-hidden).
Loads 2 police from Stage 4 checkpoints (244-dim obs, 256-hidden).
All three agents train simultaneously — nobody is frozen.

The main analytical output: thief success rate and police catch rate
plotted on the same graph over training, showing adversarial cycling.

Produces:
    checkpoints/stage5_thief_final.pt
    checkpoints/stage5_police0_final.pt
    checkpoints/stage5_police1_final.pt
    logs/stage5_log.csv
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

from env.grid_world_stage5 import GridWorldStage5
from agents.ppo_agent import PPOAgent, PPOPolicy

# ══════════════════════════════════════════════════════════
#  Hyperparameters
# ══════════════════════════════════════════════════════════

SEED = 42
GRID_SIZE = 15
MAX_STEPS = 200

# Checkpoints to load
THIEF_CHECKPOINT = "checkpoints/stage3_final.pt"
POLICE0_CHECKPOINT = "checkpoints/stage4_police0_final.pt"
POLICE1_CHECKPOINT = "checkpoints/stage4_police1_final.pt"

TOTAL_EPISODES = 200000

EVAL_EVERY = 5000
EVAL_EPISODES = 5
LOG_EVERY = 5000
SAVE_EVERY = 50000

GUI = False
RENDER_EVERY = 50
PAUSE = 0.02

if GUI:
    from utils.visualize_stage3 import show_grid_s3, reset_view_s3

# Thief PPO config — same entropy/vf_clip as stage 3
THIEF_PPO_CFG = dict(
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
    lr=5e-5,          # lower LR — fine-tuning, not learning from scratch
    lr_end=5e-6,
)

# Police PPO config — same as stage 4
POLICE_PPO_CFG = dict(
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
    lr=1e-4,
    lr_end=1e-5,
)

THIEF_HIDDEN = 128
POLICE_HIDDEN = 256


# ══════════════════════════════════════════════════════════
#  Load checkpoints
# ══════════════════════════════════════════════════════════

def load_thief(checkpoint_path, obs_dim, hidden):
    """Load thief agent from Stage 3 checkpoint."""
    agent = PPOAgent(obs_dim, 5, **THIEF_PPO_CFG, total_updates=TOTAL_EPISODES)
    agent.policy = PPOPolicy(obs_dim, 5, hidden=hidden)
    agent.optimizer = torch.optim.Adam(
        agent.policy.parameters(), lr=THIEF_PPO_CFG["lr"], eps=1e-5,
    )
    ckpt = torch.load(checkpoint_path, weights_only=True)
    agent.policy.load_state_dict(ckpt["policy_state"])
    # Fresh optimizer — adversarial landscape is different from stage 3
    print(f"  ✓ Thief loaded from {checkpoint_path}")
    return agent


def load_police(checkpoint_path, obs_dim, hidden, idx):
    """Load one police agent from Stage 4 checkpoint."""
    agent = PPOAgent(obs_dim, 5, **POLICE_PPO_CFG, total_updates=TOTAL_EPISODES)
    agent.policy = PPOPolicy(obs_dim, 5, hidden=hidden)
    agent.optimizer = torch.optim.Adam(
        agent.policy.parameters(), lr=POLICE_PPO_CFG["lr"], eps=1e-5,
    )
    ckpt = torch.load(checkpoint_path, weights_only=True)
    agent.policy.load_state_dict(ckpt["policy_state"])
    print(f"  ✓ Police {idx} loaded from {checkpoint_path}")
    return agent


# ══════════════════════════════════════════════════════════
#  Greedy evaluation
# ══════════════════════════════════════════════════════════

def greedy_eval(env, thief_agent, police_agents, n_episodes=5):
    catches = 0
    escapes = 0
    trap_deaths = 0
    timeouts = 0
    total_steps = 0

    for _ in range(n_episodes):
        thief_obs, police_obs = env.reset()
        done = False
        while not done:
            # Thief greedy
            t_state = torch.tensor(thief_obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                t_logits, _ = thief_agent.policy(t_state)
            thief_action = t_logits.argmax(dim=-1).item()

            # Police greedy
            police_actions = []
            for i, agent in enumerate(police_agents):
                p_state = torch.tensor(police_obs[i], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    p_logits, _ = agent.policy(p_state)
                police_actions.append(p_logits.argmax(dim=-1).item())

            thief_obs, _, police_obs, _, done = env.step(thief_action, police_actions)

        total_steps += env.steps
        if env.caught_by_police:
            catches += 1
        elif env.thief_reached_goal:
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

def train(env, thief_agent, police_agents, total_episodes, log_path):

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "episode", "ep_length", "outcome",
        "thief_reward", "police0_reward", "police1_reward",
        "thief_pg_loss", "thief_vf_loss", "thief_entropy",
        "police0_pg_loss", "police0_vf_loss", "police0_entropy",
        "police1_pg_loss", "police1_vf_loss", "police1_entropy",
        "thief_lr", "police_lr",
        "eval_catch_rate", "eval_escape_rate",
        "eval_trap_rate", "eval_timeout_rate", "eval_avg_steps",
    ])

    recent_catches = []
    recent_escapes = []
    recent_traps = []
    t_start = time.time()

    for ep in range(1, total_episodes + 1):

        thief_obs, police_obs = env.reset()
        done = False
        thief_ep_reward = 0.0
        police_ep_rewards = [0.0, 0.0]

        render_this = GUI and (ep % RENDER_EVERY == 0)
        path = [env.agent_pos] if render_this else None

        while not done:
            # ── thief selects action ──
            t_action, t_lp, t_val = thief_agent.select_action(thief_obs)

            # ── police select actions ──
            p_actions, p_lps, p_vals = [], [], []
            for i, agent in enumerate(police_agents):
                a, lp, v = agent.select_action(police_obs[i])
                p_actions.append(a)
                p_lps.append(lp)
                p_vals.append(v)

            # ── step environment ──
            (next_thief_obs, thief_reward,
             next_police_obs, police_rewards, done) = env.step(t_action, p_actions)

            # ── store thief transition ──
            thief_agent.store_transition(
                thief_obs, t_action, t_lp, thief_reward, t_val, done,
            )
            thief_ep_reward += thief_reward

            # ── store police transitions ──
            for i, agent in enumerate(police_agents):
                agent.store_transition(
                    police_obs[i], p_actions[i], p_lps[i],
                    police_rewards[i], p_vals[i], done,
                )
                police_ep_rewards[i] += police_rewards[i]

            thief_obs = next_thief_obs
            police_obs = next_police_obs

            # ── render ──
            if render_this:
                path.append(env.agent_pos)
                show_grid_s3(
                    env.render(), env.goal,
                    agent_pos=env.agent_pos,
                    path=path,
                    traps=env.traps,
                    traffic_positions=[car.pos for car in env.traffic_cars],
                    police_positions=env.police_positions,
                    cctv_cells=env.cctv_cells,
                    title=(f'Stage 5 Ep {ep} | Step {env.steps} | '
                           f'TR={thief_ep_reward:.0f}'),
                    pause=PAUSE,
                )

        # ── PPO updates for ALL agents ──
        t_metrics = thief_agent.update(last_state=thief_obs, last_done=done)
        p_metrics = []
        for i, agent in enumerate(police_agents):
            m = agent.update(last_state=police_obs[i], last_done=done)
            p_metrics.append(m)

        if render_this:
            reset_view_s3()

        # ── determine outcome ──
        if env.caught_by_police:
            outcome = "caught"
        elif env.thief_reached_goal:
            outcome = "escaped"
        elif env.trap_hits > 0:
            outcome = "trap"
        else:
            outcome = "timeout"

        recent_catches.append(1.0 if outcome == "caught" else 0.0)
        recent_escapes.append(1.0 if outcome == "escaped" else 0.0)
        recent_traps.append(1.0 if outcome == "trap" else 0.0)

        # ── eval ──
        eval_data = {}
        if ep % EVAL_EVERY == 0:
            eval_data = greedy_eval(env, thief_agent, police_agents, EVAL_EPISODES)

        # ── CSV ──
        m0, m1 = p_metrics[0], p_metrics[1]
        log_writer.writerow([
            ep, env.steps, outcome,
            f"{thief_ep_reward:.1f}",
            f"{police_ep_rewards[0]:.1f}", f"{police_ep_rewards[1]:.1f}",
            f"{t_metrics['pg_loss']:.4f}", f"{t_metrics['vf_loss']:.4f}",
            f"{t_metrics['entropy']:.4f}",
            f"{m0['pg_loss']:.4f}", f"{m0['vf_loss']:.4f}", f"{m0['entropy']:.4f}",
            f"{m1['pg_loss']:.4f}", f"{m1['vf_loss']:.4f}", f"{m1['entropy']:.4f}",
            f"{t_metrics['lr']:.6f}", f"{m0['lr']:.6f}",
            eval_data.get("catch_rate", ""),
            eval_data.get("escape_rate", ""),
            eval_data.get("trap_rate", ""),
            eval_data.get("timeout_rate", ""),
            eval_data.get("avg_steps", ""),
        ])

        # ── console ──
        if ep % LOG_EVERY == 0:
            window = min(LOG_EVERY, len(recent_catches))
            catch_r = np.mean(recent_catches[-window:])
            escape_r = np.mean(recent_escapes[-window:])
            trap_r = np.mean(recent_traps[-window:])
            elapsed = time.time() - t_start

            print(
                f"Ep {ep:>6}/{total_episodes} │ "
                f"Escape={escape_r:>5.1%} │ "
                f"Catch={catch_r:>5.1%} │ "
                f"Trap={trap_r:>4.1%} │ "
                f"Len={env.steps:>3d} │ "
                f"T: H={t_metrics['entropy']:.3f} │ "
                f"P0: H={m0['entropy']:.3f} │ "
                f"P1: H={m1['entropy']:.3f} │ "
                f"{elapsed:.0f}s"
            )

        # ── checkpoint ──
        if ep % SAVE_EVERY == 0:
            thief_agent.save(f"checkpoints/stage5_thief_ep{ep}.pt")
            for i, agent in enumerate(police_agents):
                agent.save(f"checkpoints/stage5_police{i}_ep{ep}.pt")
            print(f"  ↳ Saved all checkpoints (ep {ep})")

    log_file.close()

    # Final saves
    thief_agent.save("checkpoints/stage5_thief_final.pt")
    for i, agent in enumerate(police_agents):
        agent.save(f"checkpoints/stage5_police{i}_final.pt")

    print(f"\nStage 5 adversarial training complete.")
    print(f"  Log: {log_path}")
    print(f"  Thief:   checkpoints/stage5_thief_final.pt")
    print(f"  Police0: checkpoints/stage5_police0_final.pt")
    print(f"  Police1: checkpoints/stage5_police1_final.pt")


# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Verify checkpoints exist
    for label, path in [("Thief", THIEF_CHECKPOINT),
                        ("Police 0", POLICE0_CHECKPOINT),
                        ("Police 1", POLICE1_CHECKPOINT)]:
        if not os.path.exists(path):
            print(f"ERROR: {label} checkpoint not found: {path}")
            sys.exit(1)

    print("=" * 65)
    print("  STAGE 5  —  Adversarial Co-Training")
    print("=" * 65)
    print()

    env = GridWorldStage5(size=GRID_SIZE, max_steps=MAX_STEPS, rng_seed=SEED)
    print(f"  Thief obs:  {env.THIEF_OBS_DIM}")
    print(f"  Police obs: {env.POLICE_OBS_DIM}")
    print(f"  CCTV cells: {sorted(env.cctv_cells)}")
    print()

    # Load all agents
    thief = load_thief(THIEF_CHECKPOINT, env.THIEF_OBS_DIM, THIEF_HIDDEN)
    police = [
        load_police(POLICE0_CHECKPOINT, env.POLICE_OBS_DIM, POLICE_HIDDEN, 0),
        load_police(POLICE1_CHECKPOINT, env.POLICE_OBS_DIM, POLICE_HIDDEN, 1),
    ]

    # Pre-training eval
    print()
    eval_data = greedy_eval(env, thief, police, EVAL_EPISODES)
    print(f"  Pre-training eval:")
    print(f"    Escape: {eval_data['escape_rate']:.0%}  "
          f"Catch: {eval_data['catch_rate']:.0%}  "
          f"Trap: {eval_data['trap_rate']:.0%}  "
          f"Timeout: {eval_data['timeout_rate']:.0%}")
    print()

    train(env, thief, police, TOTAL_EPISODES, "logs/stage5_log.csv")