# Adversarial Pursuit-Evasion Game on a City Grid

A multi-stage reinforcement learning project where a thief agent learns to navigate a 15×15 city grid, avoid traps and traffic, and evade police agents that learn to intercept using CCTV surveillance. Both sides are trained using PPO and evolve through 5 stages, culminating in simultaneous adversarial co-training.

# Final Report:
**[Link to FAI Final Paper.pdf](FAI%20Final%20Paper.pdf)**

---

## Project Structure

```
.
├── agents/
│   └── ppo_agent.py                # PPO algorithm (shared across all stages)
├── env/
│   ├── grid_world.py               # Stage 1: base city grid
│   ├── grid_world_stage2.py        # Stage 2: + traps + traffic
│   ├── grid_world_stage3.py        # Stage 3: + random police + CCTV
│   ├── grid_world_stage4.py        # Stage 4: multi-agent, frozen thief
│   └── grid_world_stage5.py        # Stage 5: adversarial co-training
├── training/
│   ├── train_ppo.py                # Stage 1 training
│   ├── train_ppo_stage2.py         # Stage 2 training (Phase A + B)
│   ├── train_ppo_stage3.py         # Stage 3 training
│   ├── train_ppo_stage4.py         # Stage 4 police training
│   ├── train_ppo_stage5.py         # Stage 5 adversarial training
│   ├── plot_logs.py                # Stage 1 plotting
│   ├── plot_logs_stage2.py         # Stage 2 plotting
│   ├── plot_logs_stage3.py         # Stage 3 plotting
│   ├── plot_logs_stage4.py         # Stage 4 plotting
│   ├── plot_logs_stage5.py         # Stage 5 adversarial plot
│   ├── checkpoints/                # Saved model files (.pt)
│   └── logs/                       # CSV logs and PNG plots
├── utils/
│   ├── demo.py                     # Visual demo runner (all stages)
│   ├── visualize.py                # Stage 1 GUI renderer
│   ├── visualize_stage2.py         # Stage 2 GUI renderer
│   └── visualize_stage3.py         # Stage 3-5 GUI renderer
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/Priyan-B/Grid-based-Pursuit-Evasion-Game.git
cd Grid-based-Pursuit-Evasion-Game
pip install -r requirements.txt
```

Requires: `torch`, `numpy`, `matplotlib`. Python 3.8+. No GPU needed.

---

## Training

Each stage loads the previous stage's checkpoint. Run from the project root:

```bash
# Stage 1: Thief learns basic navigation
python3 training/train_ppo.py

# Stage 2: Thief learns trap avoidance + traffic dodging
python3 training/train_ppo_stage2.py

# Stage 3: Thief learns to evade random police (CCTV collects data silently)
python3 training/train_ppo_stage3.py

# Stage 4: Police learn to catch the frozen thief using CCTV + local vision
python3 training/train_ppo_stage4.py

# Stage 5: Both sides train simultaneously — adversarial co-training
python3 training/train_ppo_stage5.py
```

Checkpoints are saved to `training/checkpoints/`. Training logs are saved to `training/logs/`.

**Checkpoint chain:**

```
ppo_final.pt → stage2_phaseB_final.pt → stage3_final.pt
                                              ↓
                               stage4_police0_final.pt
                               stage4_police1_final.pt
                                              ↓
                               stage5_thief_final.pt
                               stage5_police0_final.pt
                               stage5_police1_final.pt
```

---

## Running the Demo

The demo loads trained checkpoints and runs visual episodes with full GUI. No training occurs.

```bash
# Stage 1: Basic navigation
python3 utils/demo.py --stage 1 --checkpoint training/checkpoints/ppo_final.pt

# Stage 2: Traps + traffic avoidance
python3 utils/demo.py --stage 2 --checkpoint training/checkpoints/stage2_phaseB_final.pt

# Stage 3: Evading random police
python3 utils/demo.py --stage 3 --checkpoint training/checkpoints/stage3_final.pt

# Stage 4: Trained police vs frozen thief
python3 utils/demo.py --stage 4 \
    --thief-ckpt training/checkpoints/stage3_final.pt \
    --police0-ckpt training/checkpoints/stage4_police0_final.pt \
    --police1-ckpt training/checkpoints/stage4_police1_final.pt

# Stage 5: Adversarial — both sides trained
python3 utils/demo.py --stage 5 \
    --thief-ckpt training/checkpoints/stage5_thief_final.pt \
    --police0-ckpt training/checkpoints/stage5_police0_final.pt \
    --police1-ckpt training/checkpoints/stage5_police1_final.pt
```

Options: `--episodes 10`, `--pause 0.1` (slower playback), `--seed 42` (reproducible).

---

## Plotting Results

```bash
python3 training/plot_logs.py              # Stage 1 curves
python3 training/plot_logs_stage2.py       # Stage 2 curves
python3 training/plot_logs_stage3.py       # Stage 3 curves
python3 training/plot_logs_stage4.py       # Stage 4 police curves
python3 training/plot_logs_stage5.py       # Stage 5 adversarial dynamics
```

Plots are saved as PNGs in `training/logs/`.

---

## Stages Overview

**Stage 1 — Navigation:** Thief learns to reach the goal on a city grid with walls. Observation: 3×3 local view + distance to goal (10-dim).

**Stage 2 — Traps + Traffic:** Lethal traps (episode ends on contact) and bouncing traffic cars are added. Traps start static, then become dynamic (teleport mid-episode). Observation grows to 28-dim with trap and traffic overlays.

**Stage 3 — Police + CCTV:** 3 random-walking police are added. Getting caught ends the episode. 6 CCTV cameras silently log the thief's position when it passes within their 3×3 detection range. Observation grows to 40-dim.

**Stage 4 — Police Learn:** Thief is frozen. 2 police agents train with their own PPO networks. Police see the full grid map, CCTV sighting data (updated live), teammate position, and a 3×3 local thief detection. Observation: 244-dim.

**Stage 5 — Adversarial:** All agents unfreeze and train simultaneously. The thief adapts to smarter police, police adapt to an evasive thief. The main output is the adversarial dynamics plot showing escape rate vs catch rate over training.

---

[//]: # (## Results)

[//]: # ()
[//]: # (| Stage | Result |)

[//]: # (|-------|--------|)

[//]: # (| 1 | 95%+ thief success rate |)

[//]: # (| 2 | 90%+ SR, near-zero trap deaths |)

[//]: # (| 3 | 64% SR against 3 random police |)

[//]: # (| 4 | 29% police catch rate |)

[//]: # (| 5 | 65.6% escape / 32.1% catch &#40;adversarial&#41; |)