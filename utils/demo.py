"""
Demo runner — loads checkpoints and runs visual episodes.

This is the presentation/debugging tool. No training happens here.
Just loads trained agents, runs episodes with full GUI visualisation,
and prints per-episode stats.

Usage:
    # Stage 1-3: single agent
    python demo.py --stage 3 --checkpoint checkpoints/stage3_final.pt
    python demo.py --stage 2 --checkpoint checkpoints/stage2_phaseB_final.pt
    python demo.py --stage 1 --checkpoint checkpoints/ppo_final.pt

    # Stage 4: multi-agent (thief + 2 police)
    python demo.py --stage 4 --thief-ckpt checkpoints/stage3_final.pt \
                              --police0-ckpt checkpoints/stage4_police0_final.pt \
                              --police1-ckpt checkpoints/stage4_police1_final.pt

    # Options
    python demo.py --stage 3 --pause 0.1 --episodes 3
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

from agents.ppo_agent import PPOAgent, PPOPolicy

# ══════════════════════════════════════════════════════════
#  Configuration defaults
# ══════════════════════════════════════════════════════════

DEFAULTS = dict(
    stage=3,
    checkpoint="checkpoints/stage3_final.pt",
    thief_ckpt="checkpoints/stage3_final.pt",
    police0_ckpt="checkpoints/stage4_police0_final.pt",
    police1_ckpt="checkpoints/stage4_police1_final.pt",
    episodes=5,
    seed=None,
    pause=0.05,
    grid_size=15,
    max_steps=200,
)


# ══════════════════════════════════════════════════════════
#  Environment + visualiser loader per stage
# ══════════════════════════════════════════════════════════

def make_env(stage, grid_size=15, max_steps=200, rng_seed=42,
             thief_policy=None):
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
    elif stage == 4:
        from env.grid_world_stage4 import GridWorldStage4
        assert thief_policy is not None, "Stage 4 requires thief_policy"
        return GridWorldStage4(
            thief_policy=thief_policy,
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
    elif stage in (3, 4):
        # Stage 4 reuses stage 3 visualiser
        from utils.visualize_stage3 import show_grid_s3, reset_view_s3
        return show_grid_s3, reset_view_s3
    else:
        raise ValueError(f"Unknown stage: {stage}")


def render_frame(stage, env, show_fn, path, ep, info_str, pause):
    """Call the correct visualiser with the right arguments per stage."""
    if stage == 1:
        show_fn(
            env.render(), env.goal,
            agent_pos=env.agent_pos,
            path=path,
            title=f'Demo Ep {ep}  |  {info_str}',
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
            title=f'Demo Ep {ep}  |  {info_str}',
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
            title=f'Demo Ep {ep}  |  {info_str}',
            pause=pause,
        )
    elif stage == 4:
        traffic_pos = [car.pos for car in env.traffic_cars]
        show_fn(
            env.render(), env.goal,
            agent_pos=env.agent_pos,
            path=path,
            traps=env.traps,
            traffic_positions=traffic_pos,
            police_positions=env.police_positions,
            cctv_cells=env.cctv_cells,
            title=f'Demo Ep {ep}  |  {info_str}',
            pause=pause,
        )


# ══════════════════════════════════════════════════════════
#  Stage 1-3 demo (single agent)
# ══════════════════════════════════════════════════════════

def run_demo_single(stage, checkpoint, episodes, seed, pause,
                    grid_size, max_steps):

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

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

    agent = PPOAgent(env.OBS_DIM, 5)
    agent.load(checkpoint)
    agent.policy.eval()
    print(f"  ✓ Loaded checkpoint\n")

    show_fn, reset_fn = get_show_fn(stage)

    stats = []
    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        ep_reward = 0.0
        path = [env.agent_pos]

        while not done:
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, _ = agent.policy(state_t)
            action = logits.argmax(dim=-1).item()

            state, reward, done = env.step(action)
            ep_reward += reward
            path.append(env.agent_pos)

            info = f'Step {env.steps}  |  R={ep_reward:.0f}'
            render_frame(stage, env, show_fn, path, ep, info, pause)

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
            f"Reward={ep_reward:7.1f}{extra}"
        )
        stats.append(dict(
            reached=reached, steps=env.steps, reward=ep_reward,
            caught=getattr(env, 'caught_by_police', False),
            trap_death=getattr(env, 'trap_hits', 0) > 0,
        ))
        reset_fn()
        time.sleep(0.5)

    _print_summary(stats)
    _keep_window_open()


# ══════════════════════════════════════════════════════════
#  Stage 4 demo (multi-agent: frozen thief + 2 police)
# ══════════════════════════════════════════════════════════

def run_demo_stage4(thief_ckpt, police0_ckpt, police1_ckpt,
                    episodes, seed, pause, grid_size, max_steps):

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    print(f"\n{'=' * 55}")
    print(f"  DEMO  —  Stage 4 (Police vs Frozen Thief)")
    print(f"{'=' * 55}")
    print(f"  Thief ckpt:   {thief_ckpt}")
    print(f"  Police 0 ckpt: {police0_ckpt}")
    print(f"  Police 1 ckpt: {police1_ckpt}")
    print(f"  Episodes:     {episodes}")
    print(f"  Pause:        {pause}s per frame")
    print()

    # Load frozen thief
    thief_policy = PPOPolicy(40, 5, hidden=128)
    ckpt = torch.load(thief_ckpt, weights_only=True)
    thief_policy.load_state_dict(ckpt["policy_state"])
    thief_policy.eval()
    print(f"  ✓ Loaded frozen thief")

    # Create environment
    env = make_env(4, grid_size=grid_size, max_steps=max_steps,
                   thief_policy=thief_policy)
    print(f"  Police obs dim: {env.POLICE_OBS_DIM}")
    print(f"  CCTV cells:     {sorted(env.cctv_cells)}")

    # Load 2 police agents
    police_agents = []
    for i, ckpt_path in enumerate([police0_ckpt, police1_ckpt]):
        agent = PPOAgent(env.POLICE_OBS_DIM, 5)
        # Recreate with 256 hidden (matching training)
        agent.policy = PPOPolicy(env.POLICE_OBS_DIM, 5, hidden=256)
        ckpt = torch.load(ckpt_path, weights_only=True)
        agent.policy.load_state_dict(ckpt["policy_state"])
        agent.policy.eval()
        police_agents.append(agent)
        print(f"  ✓ Loaded police {i}")

    print()
    show_fn, reset_fn = get_show_fn(4)

    stats = []
    for ep in range(1, episodes + 1):
        obs_list = env.reset()
        done = False
        path = [env.agent_pos]

        while not done:
            # Police actions (greedy)
            actions = []
            for i, agent in enumerate(police_agents):
                state_t = torch.tensor(
                    obs_list[i], dtype=torch.float32
                ).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = agent.policy(state_t)
                actions.append(logits.argmax(dim=-1).item())

            obs_list, rewards, done = env.step(actions)
            path.append(env.agent_pos)

            # Render
            cctv_str = "CCTV!" if env.last_cctv_sighting is not None else ""
            info = f'Step {env.steps}  |  {cctv_str}'
            render_frame(4, env, show_fn, path, ep, info, pause)

        # Determine outcome
        if env.caught_by_police:
            outcome = f"✗ CAUGHT by Police {env.catcher_idx}"
        elif env.agent_pos == env.goal:
            outcome = "✓ THIEF ESCAPED"
        elif env.trap_hits > 0:
            outcome = "✗ THIEF TRAP DEATH"
        else:
            outcome = "✗ TIMEOUT"

        had_cctv = "Yes" if env.last_cctv_sighting is not None else "No"
        print(
            f"  Episode {ep}/{episodes}  |  "
            f"{outcome:26s}  |  "
            f"Steps={env.steps:3d}  |  "
            f"CCTV={had_cctv}"
        )
        stats.append(dict(
            reached=(env.agent_pos == env.goal),
            steps=env.steps,
            reward=sum(rewards),
            caught=env.caught_by_police,
            trap_death=env.trap_hits > 0,
        ))
        reset_fn()
        time.sleep(0.5)

    _print_summary(stats)
    _keep_window_open()


# ══════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════

def _print_summary(stats):
    n = len(stats)
    successes = sum(1 for s in stats if s["reached"])
    catches = sum(1 for s in stats if s["caught"])
    traps = sum(1 for s in stats if s["trap_death"])
    avg_len = np.mean([s["steps"] for s in stats])

    print(f"\n{'─' * 55}")
    print(f"  Summary: {successes}/{n} thief goals  |  "
          f"{catches}/{n} caught  |  {traps}/{n} trap deaths")
    print(f"  Avg steps: {avg_len:.1f}")
    print(f"{'─' * 55}\n")


def _keep_window_open():
    print("  Close the plot window to exit.")
    import matplotlib.pyplot as plt
    plt.ioff()
    plt.show()


# ══════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo runner for trained agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Stage 1:  python demo.py --stage 1 --checkpoint checkpoints/ppo_final.pt
  Stage 2:  python demo.py --stage 2 --checkpoint checkpoints/stage2_phaseB_final.pt
  Stage 3:  python demo.py --stage 3 --checkpoint checkpoints/stage3_final.pt
  Stage 4:  python demo.py --stage 4 --thief-ckpt checkpoints/stage3_final.pt \\
                --police0-ckpt checkpoints/stage4_police0_final.pt \\
                --police1-ckpt checkpoints/stage4_police1_final.pt
        """,
    )
    parser.add_argument("--stage", type=int, default=DEFAULTS["stage"],
                        help="Stage number (1, 2, 3, or 4)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULTS["checkpoint"],
                        help="Checkpoint path (stages 1-3)")
    parser.add_argument("--thief-ckpt", type=str, default=DEFAULTS["thief_ckpt"],
                        help="Thief checkpoint (stage 4)")
    parser.add_argument("--police0-ckpt", type=str, default=DEFAULTS["police0_ckpt"],
                        help="Police 0 checkpoint (stage 4)")
    parser.add_argument("--police1-ckpt", type=str, default=DEFAULTS["police1_ckpt"],
                        help="Police 1 checkpoint (stage 4)")
    parser.add_argument("--episodes", type=int, default=DEFAULTS["episodes"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--pause", type=float, default=DEFAULTS["pause"])
    parser.add_argument("--grid-size", type=int, default=DEFAULTS["grid_size"])
    parser.add_argument("--max-steps", type=int, default=DEFAULTS["max_steps"])
    args = parser.parse_args()

    if args.stage <= 3:
        if not os.path.exists(args.checkpoint):
            print(f"ERROR: Checkpoint not found: {args.checkpoint}")
            sys.exit(1)
        run_demo_single(
            stage=args.stage,
            checkpoint=args.checkpoint,
            episodes=args.episodes,
            seed=args.seed,
            pause=args.pause,
            grid_size=args.grid_size,
            max_steps=args.max_steps,
        )
    elif args.stage == 4:
        for label, path in [("Thief", args.thief_ckpt),
                            ("Police 0", args.police0_ckpt),
                            ("Police 1", args.police1_ckpt)]:
            if not os.path.exists(path):
                print(f"ERROR: {label} checkpoint not found: {path}")
                sys.exit(1)
        run_demo_stage4(
            thief_ckpt=args.thief_ckpt,
            police0_ckpt=args.police0_ckpt,
            police1_ckpt=args.police1_ckpt,
            episodes=args.episodes,
            seed=args.seed,
            pause=args.pause,
            grid_size=args.grid_size,
            max_steps=args.max_steps,
        )
    else:
        print(f"ERROR: Unknown stage {args.stage}")
        sys.exit(1)