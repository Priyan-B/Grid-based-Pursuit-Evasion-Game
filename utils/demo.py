"""
Demo runner — loads a checkpoint and runs visual episodes.

This is the presentation/debugging tool. No training happens here.
Just loads a trained agent, runs episodes with full GUI visualisation,
and prints per-episode stats.

Usage:
    python demo.py                           # uses defaults at top of file
    python demo.py --stage 3 --episodes 5    # command-line override

Configure the section below or use command-line args.
"""

import os
import sys
import argparse
import numpy as np
import torch
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.ppo_agent import PPOAgent

# ══════════════════════════════════════════════════════════
#  Configuration — edit these or use command-line args
# ══════════════════════════════════════════════════════════

DEFAULTS = dict(
    stage=3,
    checkpoint="training/checkpoints/stage3_final.pt",
    episodes=5,
    seed=None,          # None = random each run, int = deterministic
    pause=0.05,         # seconds per frame (slower = easier to watch)
    grid_size=15,
    max_steps=200,
)


# ══════════════════════════════════════════════════════════
#  Environment + visualiser loader per stage
# ══════════════════════════════════════════════════════════

def make_env(stage, grid_size=15, max_steps=200, rng_seed=42):
    """Create the correct environment for the given stage."""
    if stage == 1:
        from env.grid_world import GridWorld
        return GridWorld(size=grid_size, max_steps=max_steps, rng_seed=rng_seed)
    elif stage == 2:
        from env.grid_world_stage2 import GridWorldStage2
        return GridWorldStage2(
            size=grid_size, max_steps=max_steps, rng_seed=rng_seed,
            trap_mode="dynamic",
        )
    elif stage == 3:
        from env.grid_world_stage3 import GridWorldStage3
        return GridWorldStage3(
            size=grid_size, max_steps=max_steps, rng_seed=rng_seed,
        )
    else:
        raise ValueError(f"Unknown stage: {stage}")


def get_show_fn(stage):
    """Return the correct visualisation function for the stage."""
    if stage == 1:
        from utils.visualize import show_grid, reset_view
        return show_grid, reset_view
    elif stage == 2:
        from utils.visualize_stage2 import show_grid_s2, reset_view_s2
        return show_grid_s2, reset_view_s2
    elif stage == 3:
        from utils.visualize_stage3 import show_grid_s3, reset_view_s3
        return show_grid_s3, reset_view_s3
    else:
        raise ValueError(f"Unknown stage: {stage}")


def render_frame(stage, env, show_fn, path, ep, ep_reward, pause):
    """Call the correct visualiser with the right arguments per stage."""
    if stage == 1:
        show_fn(
            env.render(), env.goal,
            agent_pos=env.agent_pos,
            path=path,
            title=f'Demo Ep {ep}  |  Step {env.steps}  |  R={ep_reward:.0f}',
            pause=pause,
        )
    elif stage == 2:
        traffic_pos = [car.pos for car in env.traffic_cars]
        show_fn(
            env.render(), env.goal,
            agent_pos=env.agent_pos,
            path=path,
            traps=env.traps,
            traffic_positions=traffic_pos,
            title=f'Demo Ep {ep}  |  Step {env.steps}  |  R={ep_reward:.0f}',
            pause=pause,
        )
    elif stage == 3:
        traffic_pos = [car.pos for car in env.traffic_cars]
        police_pos = [pc.pos for pc in env.police_cars]
        show_fn(
            env.render(), env.goal,
            agent_pos=env.agent_pos,
            path=path,
            traps=env.traps,
            traffic_positions=traffic_pos,
            police_positions=police_pos,
            cctv_cells=env.cctv_cells,
            title=f'Demo Ep {ep}  |  Step {env.steps}  |  R={ep_reward:.0f}',
            pause=pause,
        )


# ══════════════════════════════════════════════════════════
#  Main demo loop
# ══════════════════════════════════════════════════════════

def run_demo(stage, checkpoint, episodes, seed, pause, grid_size, max_steps):

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Create environment
    env = make_env(stage, grid_size=grid_size, max_steps=max_steps)

    print(f"\n{'=' * 55}")
    print(f"  DEMO  —  Stage {stage}")
    print(f"{'=' * 55}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Obs dim:    {env.OBS_DIM}")
    print(f"  Episodes:   {episodes}")
    print(f"  Pause:      {pause}s per frame")
    if stage >= 3:
        print(f"  CCTV cells: {sorted(env.cctv_cells)}")
    print()

    # Load agent
    agent = PPOAgent(env.OBS_DIM, 5)
    agent.load(checkpoint)
    agent.policy.eval()
    print(f"  ✓ Loaded checkpoint\n")

    # Get visualiser
    show_fn, reset_fn = get_show_fn(stage)

    # Run episodes
    stats = []
    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        ep_reward = 0.0
        path = [env.agent_pos]

        while not done:
            # Greedy action (argmax, no sampling)
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, _ = agent.policy(state_t)
            action = logits.argmax(dim=-1).item()

            state, reward, done = env.step(action)
            ep_reward += reward
            path.append(env.agent_pos)

            # Render
            render_frame(stage, env, show_fn, path, ep, ep_reward, pause)

        # Episode done — print summary
        reached = env.agent_pos == env.goal
        outcome = "✓ GOAL" if reached else "✗ TIMEOUT"

        extra = ""
        if stage >= 2:
            if env.trap_hits > 0:
                outcome = "✗ TRAP DEATH"
            extra += f"  Traps={env.trap_hits}  Traffic={env.traffic_hits}"
        if stage >= 3:
            if env.caught_by_police:
                outcome = "✗ CAUGHT"
            extra += f"  CCTV={len(env.cctv_log)}"

        print(
            f"  Episode {ep}/{episodes}  |  "
            f"{outcome:14s}  |  "
            f"Steps={env.steps:3d}  |  "
            f"Reward={ep_reward:7.1f}"
            f"{extra}"
        )

        stats.append(dict(
            reached=reached,
            steps=env.steps,
            reward=ep_reward,
            caught=getattr(env, 'caught_by_police', False),
            trap_death=getattr(env, 'trap_hits', 0) > 0,
        ))

        reset_fn()

        # Small pause between episodes
        time.sleep(0.5)

    # Summary
    n = len(stats)
    successes = sum(1 for s in stats if s["reached"])
    catches = sum(1 for s in stats if s["caught"])
    traps = sum(1 for s in stats if s["trap_death"])
    avg_r = np.mean([s["reward"] for s in stats])
    avg_len = np.mean([s["steps"] for s in stats])

    print(f"\n{'─' * 55}")
    print(f"  Summary: {successes}/{n} goals  |  "
          f"{catches}/{n} caught  |  {traps}/{n} trap deaths")
    print(f"  Avg reward: {avg_r:.1f}  |  Avg steps: {avg_len:.1f}")
    print(f"{'─' * 55}\n")

    # Keep window open
    print("  Close the plot window to exit.")
    import matplotlib.pyplot as plt
    plt.ioff()
    plt.show()


# ══════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo runner for trained agents")
    parser.add_argument("--stage", type=int, default=DEFAULTS["stage"],
                        help="Stage number (1, 2, or 3)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULTS["checkpoint"],
                        help="Path to .pt checkpoint file")
    parser.add_argument("--episodes", type=int, default=DEFAULTS["episodes"],
                        help="Number of episodes to run")
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"],
                        help="Random seed (omit for random)")
    parser.add_argument("--pause", type=float, default=DEFAULTS["pause"],
                        help="Seconds per frame")
    parser.add_argument("--grid-size", type=int, default=DEFAULTS["grid_size"])
    parser.add_argument("--max-steps", type=int, default=DEFAULTS["max_steps"])
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    run_demo(
        stage=args.stage,
        checkpoint=args.checkpoint,
        episodes=args.episodes,
        seed=args.seed,
        pause=args.pause,
        grid_size=args.grid_size,
        max_steps=args.max_steps,
    )