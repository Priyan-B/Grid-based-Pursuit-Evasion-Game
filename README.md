# Adversarial Pursuit-Evasion Game on a City Grid

A multi-stage reinforcement learning project where a **thief agent** learns to navigate a procedurally generated 15×15 city grid, avoid hazards, and evade **police agents** that learn to intercept using CCTV surveillance data. Both sides are trained using Proximal Policy Optimization (PPO) and evolve through increasingly complex stages, culminating in simultaneous adversarial co-training.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Setup and Installation](#setup-and-installation)
4. [The Environment](#the-environment)
5. [The PPO Algorithm](#the-ppo-algorithm)
6. [Training Stages](#training-stages)
7. [Running Training](#running-training)
8. [Running the Demo](#running-the-demo)
9. [Plotting Results](#plotting-results)
10. [Observation Spaces](#observation-spaces)
11. [Weight Transplant Technique](#weight-transplant-technique)
12. [Hyperparameter Reference](#hyperparameter-reference)

---

## Project Overview

The project builds a reinforcement learning environment in which agents learn through pure trial and error — no hand-coded rules, no pathfinding algorithms. A neural network receives numerical observations about the agent's surroundings and outputs action probabilities. Through hundreds of thousands of episodes of reward feedback, the network weights converge on effective strategies.

The project is structured as five progressive stages:

| Stage | What Trains | New Mechanics | Thief Obs | Key Result |
|-------|------------|---------------|-----------|------------|
| 1 | Thief | Basic navigation | 10-dim | 95%+ success rate |
| 2 | Thief | Traps (lethal) + Traffic | 28-dim | 90%+ SR, zero trap deaths |
| 3 | Thief | Random police + silent CCTV | 40-dim | ~64% SR against 3 random cops |
| 4 | Police (×2) | Thief frozen, police learn | 244-dim | ~29% catch rate |
| 5 | All agents | Adversarial co-training | Both | Adversarial cycling dynamics |

Each stage loads the previous checkpoint and expands the observation space while preserving all previously learned behavior through a surgical weight transplant technique.

---

## Project Structure

```
.
├── agents/
│   └── ppo_agent.py              # PPO algorithm (shared across ALL stages)
│
├── env/
│   ├── grid_world.py             # Stage 1: base city grid
│   ├── grid_world_stage2.py      # Stage 2: + traps + traffic
│   ├── grid_world_stage3.py      # Stage 3: + random police + CCTV
│   ├── grid_world_stage4.py      # Stage 4: multi-agent, frozen thief
│   └── grid_world_stage5.py      # Stage 5: adversarial co-training
│
├── training/
│   ├── train_ppo.py              # Stage 1 training script
│   ├── train_ppo_stage2.py       # Stage 2 (Phase A + Phase B)
│   ├── train_ppo_stage3.py       # Stage 3 training script
│   ├── train_ppo_stage4.py       # Stage 4 police training
│   ├── train_ppo_stage5.py       # Stage 5 adversarial training
│   ├── plot_logs.py              # Stage 1 plotting
│   ├── plot_logs_stage2.py       # Stage 2 plotting
│   ├── plot_logs_stage3.py       # Stage 3 plotting
│   ├── plot_logs_stage4.py       # Stage 4 plotting
│   └── plot_logs_stage5.py       # Stage 5 plotting (adversarial curves)
│
├── utils/
│   ├── visualize.py              # Stage 1 GUI renderer
│   ├── visualize_stage2.py       # Stage 2 GUI renderer (+ traps/traffic)
│   ├── visualize_stage3.py       # Stage 3+ GUI renderer (+ police/CCTV)
│   └── demo.py                   # Standalone demo for all stages (1-5)
│
├── checkpoints/                  # Saved .pt model files
├── logs/                         # CSV training logs and PNG plots
├── requirements.txt
└── README.md
```

---

## Setup and Installation

```bash
git clone https://github.com/Priyan-B/Grid-based-Pursuit-Evasion-Game.git
cd Grid-based-Pursuit-Evasion-Game
pip install -r requirements.txt
```

Requirements: `torch`, `numpy`, `matplotlib`

Python 3.8+ recommended. No GPU required — all networks are small enough to train efficiently on CPU.

---

## The Environment

The city is a 15×15 grid generated procedurally with a fixed seed for reproducibility. The outer ring is always walls. Interior cells are a mix of walls (buildings) and roads, with structured corridors every 3 rows/columns. A BFS flood-fill ensures all road cells are connected — no isolated pockets.

```
Grid (. = road, # = wall):
 0  ###############
 1  #.....##......#
 2  #.............#
 3  ##..#.##....#.#
 4  #...#.........#
 5  #.............#
 6  #...#..#.....##
 7  #.....#......##
 8  #.............#
 9  #.....#.......#
10  ##.....#....#.#
11  #.............#
12  #..#..#..##..##
13  #..#.....#..###
14  ###############
```

The goal (safe zone) is at position (11, 13), near the bottom-right corner. The thief spawns randomly on any road cell except the goal at the start of each episode.

---

## The PPO Algorithm

All agents across all stages use the identical PPO implementation in `agents/ppo_agent.py`. Nothing about the algorithm changes between stages — only the observation space, reward function, and which agents are frozen vs training.

**Network architecture:**

```
observation → Linear(obs_dim, hidden) → Tanh
            → Linear(hidden, hidden)  → Tanh
            → policy_head: Linear(hidden, 5)   → 5 action logits
            → value_head:  Linear(hidden, 1)   → state value V(s)
```

The thief uses hidden=128. Police use hidden=256 (larger observation space).

**Core PPO update:**

Each episode, the agent collects a rollout of (state, action, log_prob, reward, value, done) tuples. At the end:

1. **GAE** computes advantages: how much better each action was compared to the value estimate.

   ```
   δ_t = r_t + γ·V(s_{t+1})·(1-done) - V(s_t)
   A_t = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...
   ```

2. **Advantage normalization**: advantages are standardized to mean=0, std=1.

3. **Clipped surrogate loss** prevents the policy from changing too much per update:

   ```
   ratio = π_new(a|s) / π_old(a|s)
   L_clip = max(-A·ratio, -A·clamp(ratio, 1-ε, 1+ε))
   ```

4. **Clipped value loss** prevents the value head from overshooting:

   ```
   V_clipped = V_old + clamp(V_new - V_old, -vf_clip, +vf_clip)
   L_value = 0.5 · max((V_new - R)², (V_clipped - R)²)
   ```

5. **Entropy bonus** encourages exploration by penalizing overly deterministic policies.

6. **Total loss**: `L = L_clip + vf_coef·L_value - ent_coef·entropy`

The data is shuffled into minibatches and this update runs for n_epochs passes. Learning rate and entropy coefficient are linearly annealed over training.

---

## Training Stages

### Stage 1 — Basic Navigation

The thief learns to navigate from a random spawn point to the goal.

**Observation (10-dim):** 3×3 local patch around the thief (walls=1.0, roads=0.0, goal=0.5) + normalized Manhattan distance to goal.

**Reward:** +100 reaching goal, -0.1 per step, -1.0 bumping walls, +1.0/-1.0 for moving closer/farther from goal.

**Result:** 95%+ success rate after 100k episodes.

### Stage 2 — Traps and Traffic

Two training phases:

**Phase A (Static Traps):** 6 traps placed randomly at episode start, fixed all episode. Stepping on a trap is lethal — episode ends immediately with -50 reward. Traffic cars patrol contiguous road segments vertically and horizontally, moving every 3 thief steps. Hitting traffic causes a -3 penalty and 2-step stun.

**Phase B (Dynamic Traps):** Same as Phase A, but traps teleport to new positions every 15 steps mid-episode.

**Observation (28-dim):** Previous 10 dims + 3×3 trap overlay + 3×3 traffic overlay. The Stage 1 checkpoint is transplanted into the larger network — wall and distance weights are copied exactly, new channels initialized near-zero.

**Result:** 90%+ success rate, near-zero trap deaths.

### Stage 3 — Police and CCTV

3 police cars perform random walks on road cells. They walk through traps and traffic freely. If a police car and the thief occupy the same cell, the thief is caught (episode ends, -100 reward).

6 CCTV cameras are placed at fixed intersections with 3×3 detection vision. When the thief passes within 1 cell of a camera, the position and timestep are silently logged. The thief has no knowledge of camera locations.

**Observation (40-dim):** Previous 28 dims + 3×3 police overlay + 3 normalized police distances.

**Result:** ~64% success rate against random police.

### Stage 4 — Police Learn

The thief is frozen at its Stage 3 checkpoint. 2 police agents, each with their own PPO network, train against the frozen competent thief.

**Police observation (244-dim):**

| Component | Dims | Description |
|-----------|------|-------------|
| Full grid | 225 | 15×15 flattened (wall=1, road=0) |
| Own position | 2 | Normalized (row/size, col/size) |
| Goal position | 2 | Where the thief is heading |
| CCTV sighting | 2 | Last known thief position (live updates) |
| Has sighting | 1 | Flag: have we spotted the thief? |
| Staleness | 1 | Steps since last sighting, normalized |
| Teammate position | 2 | Relative to self (encourages spreading) |
| Local thief detection | 9 | 3×3 vision — can see thief nearby |

Police combine two information sources: CCTV for long-range tracking ("thief was at X, Y steps ago") and 3×3 local vision for short-range pursuit ("thief is right next to me").

**Police reward:** +100 catching thief, +50 teammate caught thief, -50 thief escaped, -30 timeout.

**Result:** ~29% catch rate at 500k episodes.

### Stage 5 — Adversarial Co-Training

Both thief (from Stage 3) and police (from Stage 4) are loaded and train simultaneously. Nobody is frozen. Each episode, all three agents select actions, the environment steps, and all three receive observations and rewards. All three do PPO updates at episode end.

**The key analytical output:** A plot showing thief escape rate and police catch rate on the same graph over training. The expected pattern is adversarial cycling — one side improves, forcing the other to adapt, and back and forth.

---

## Running Training

Training scripts are run from the project root directory. Each stage requires the previous stage's checkpoint.

```bash
# Stage 1: Basic navigation
python training/train_ppo.py

# Stage 2: Traps + Traffic (loads Stage 1 checkpoint)
python training/train_ppo_stage2.py

# Stage 3: Police + CCTV (loads Stage 2 checkpoint)
python training/train_ppo_stage3.py

# Stage 4: Police training (loads Stage 3 checkpoint)
python training/train_ppo_stage4.py

# Stage 5: Adversarial (loads Stage 3 thief + Stage 4 police)
python training/train_ppo_stage5.py
```

Training parameters (episode count, eval frequency, checkpoint frequency) are configured at the top of each training script. Set `GUI = True` to watch episodes live during training (requires a display).

**Expected checkpoint chain:**

```
ppo_final.pt (Stage 1)
    → stage2_phaseB_final.pt (Stage 2)
        → stage3_final.pt (Stage 3)
            → stage4_police0_final.pt, stage4_police1_final.pt (Stage 4)
                → stage5_thief_final.pt, stage5_police0_final.pt,
                  stage5_police1_final.pt (Stage 5)
```

---

## Running the Demo

The demo script loads trained checkpoints and runs visual episodes with full GUI rendering. No training occurs — purely for presentation and debugging.

```bash
# Stage 1: Watch the thief navigate
python utils/demo.py --stage 1 --checkpoint checkpoints/ppo_final.pt

# Stage 2: Watch trap and traffic avoidance
python utils/demo.py --stage 2 --checkpoint checkpoints/stage2_phaseB_final.pt

# Stage 3: Watch thief evade random police
python utils/demo.py --stage 3 --checkpoint checkpoints/stage3_final.pt

# Stage 4: Watch trained police chase frozen thief
python utils/demo.py --stage 4 \
    --thief-ckpt checkpoints/stage3_final.pt \
    --police0-ckpt checkpoints/stage4_police0_final.pt \
    --police1-ckpt checkpoints/stage4_police1_final.pt

# Stage 5: Watch adversarial agents compete
python utils/demo.py --stage 5 \
    --thief-ckpt checkpoints/stage5_thief_final.pt \
    --police0-ckpt checkpoints/stage5_police0_final.pt \
    --police1-ckpt checkpoints/stage5_police1_final.pt

# Options
python utils/demo.py --stage 3 --pause 0.1 --episodes 10
```

The demo runs greedy (argmax) actions with no exploration noise. Each episode prints outcome, steps, and reward. A summary is shown at the end. The plot window stays open until manually closed.

**Visual elements:**

| Symbol | Color | Entity |
|--------|-------|--------|
| Red circle | #e74c3c | Thief |
| Dark square | #2c3e50 | Police |
| Orange diamond | #e67e22 | Trap |
| Yellow square | #f1c40f | Traffic |
| Green square + star | #2ecc71 | Goal (safe zone) |
| Purple tint | #9b59b6 | CCTV camera |
| Blue line | #3498db | Thief's path |
| Dark gray | #393940 | Wall / Building |
| Light gray | #edede6 | Road |

---

## Plotting Results

Each stage has its own plotting script that reads the CSV log and generates training curves.

```bash
python training/plot_logs.py                    # Stage 1
python training/plot_logs_stage2.py             # Stage 2 (Phase A + B)
python training/plot_logs_stage3.py             # Stage 3
python training/plot_logs_stage4.py             # Stage 4
python training/plot_logs_stage5.py             # Stage 5 (adversarial plot)
```

Plots are saved as PNG files in the `logs/` directory. The Stage 5 plot is the main analytical output of the project — it shows thief escape rate and police catch rate on the same axes over training.

---

## Observation Spaces

### Thief Observation (40-dim, Stage 3+)

```
Index   Dims  Description
[0:9]     9   3×3 local patch (wall=1.0, road=0.0, goal=0.5)
[9:18]    9   3×3 trap overlay (1.0 if trap present)
[18:27]   9   3×3 traffic overlay (1.0 if traffic car present)
[27]      1   Normalized Manhattan distance to goal
[28:37]   9   3×3 police overlay (1.0 if police in cell)
[37:40]   3   Normalized distance to each police car (nearest first)
```

Stages 1 and 2 use subsets: Stage 1 uses [0:9] + [27] = 10 dims, Stage 2 uses [0:27] + [27] = 28 dims. The observation layout is designed so each stage's features occupy the same indices as the previous stage, enabling clean weight transplants.

### Police Observation (244-dim, Stage 4+)

```
Index       Dims  Description
[0:225]     225   Full 15×15 grid (wall=1, road=0)
[225:227]     2   Own position (row/size, col/size)
[227:229]     2   Goal position (row/size, col/size)
[229:231]     2   Last CCTV sighting position (-1,-1 if none)
[231]         1   Has-sighting flag (0 or 1)
[232]         1   Sighting staleness (steps_since / max_steps)
[233:235]     2   Relative position to teammate (dr/size, dc/size)
[235:244]     9   3×3 local thief detection
```

---

## Weight Transplant Technique

When the observation space grows between stages, the agent's first neural network layer changes shape. Rather than training from scratch, the previous stage's weights are surgically copied into the new, larger layer:

1. Create the new (larger) network
2. Initialize all weights near zero (std=0.01)
3. Copy the previous stage's weights into the correct column positions
4. All other layers (hidden, policy head, value head) are identical in shape — copy directly
5. The optimizer starts fresh (Adam moments have wrong shape for the resized layer)

**Critical detail:** When new observation channels are inserted between existing ones (not appended), the column mapping must account for the shift. For example, Stage 1→2: the distance feature moves from index 9 to index 27 because trap and traffic channels are inserted in between.

On the first forward pass after transplant, the new channels contribute near-zero activation (small weights × input ≈ 0), so the agent behaves almost identically to before. Through training, the near-zero weights gradually grow to encode the new features.

---

## Hyperparameter Reference

### PPO Parameters (stable across all stages)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| clip_eps | 0.2 | PPO clipping range — limits policy change per update |
| gamma | 0.99 | Discount factor — effective horizon ~100 steps |
| gae_lambda | 0.95 | GAE bias-variance tradeoff |
| vf_coef | 0.5 | Value loss weight in total loss |
| vf_clip | 3.0 | Value head clipping — prevents gradient explosions |
| max_grad_norm | 0.5 | Gradient clipping threshold |
| n_epochs | 4 | PPO passes over each rollout |
| batch_size | 64 | Minibatch size |
| min_batch_size | 8 | Skip batches smaller than this (prevents NaN) |

### Entropy and Learning Rate (tuned per stage)

| Stage | ent_coef | ent_coef_end | lr | lr_end |
|-------|----------|--------------|-----|--------|
| 1 | 0.02 | 0.001 | 3e-4 | 3e-5 |
| 2 | 0.02 | 0.001 | 3e-4 | 3e-5 |
| 3 | 0.05 | 0.02 | 3e-4 | 3e-5 |
| 4 | 0.05 | 0.02 | 3e-4 | 3e-5 |
| 5 | 0.05 | 0.02 | 1e-4 | 1e-5 |

The entropy floor (ent_coef_end) is critical. Values below 0.02 for Stages 3+ cause policy collapse — the agent becomes fully deterministic, locks into a single path, and cannot recover when that path fails.

### Environment Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| Grid size | 15×15 | City dimensions |
| Max steps | 200 | Episode timeout |
| N_TRAPS | 6 | Lethal traps per episode |
| N_TRAFFIC | 5 | Traffic cars |
| TRAFFIC_MOVE_INTERVAL | 3 | Traffic moves every N thief steps |
| TRAP_SHUFFLE_INTERVAL | 15 | Dynamic traps teleport every N steps |
| STUN_DURATION | 2 | Steps frozen after traffic hit |
| N_POLICE (Stage 3) | 3 | Random police in thief training |
| N_POLICE (Stage 4-5) | 2 | Learning police agents |
| CCTV cameras | 6 | Fixed positions, 3×3 detection vision |

### Reward Functions

**Thief:**

| Event | Reward |
|-------|--------|
| Reached goal | +100 |
| Caught by police | -100 |
| Stepped on trap | -50 |
| Hit traffic | -3 |
| Moved closer to goal | +1 |
| Moved away from goal | -1 |
| Bumped wall | -1 |
| Stayed still | -0.3 |
| Per step | -0.1 |

**Police (Stage 4-5):**

| Event | Reward |
|-------|--------|
| Caught thief (own catch) | +100 |
| Teammate caught thief | +50 |
| Thief reached goal | -50 |
| Timeout | -30 |
| Per step | -0.1 |