"""
Main orchestration script that runs Stage 1 and Stage 2 training sequentially.

Execution flow:
  1. Runs train_ppo.py (Stage 1: basic navigation with walls)
  2. Then runs train_ppo_stage2.py (Stage 2: navigation with traps and traffic)

This ensures that Stage 1 completes fully before Stage 2 begins,
and Stage 2 loads the Stage 1 checkpoint as its starting point.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_training_pipeline():
    """Run the complete training pipeline: Stage 1 → Stage 2"""

    project_root = Path(__file__).parent.parent
    training_dir = project_root / "training"

    print("=" * 70)
    print("  STARTING GRID-BASED PURSUIT-EVASION TRAINING PIPELINE")
    print("=" * 70)
    print()

    # ─────────────────────────────────────────────────────────
    #  STAGE 1: Basic Navigation with Walls
    # ─────────────────────────────────────────────────────────
    print("▶ STAGE 1: Training basic navigation (walls only)...")
    print("-" * 70)

    stage1_script = training_dir / "train_ppo.py"

    try:
        result = subprocess.run(
            [sys.executable, str(stage1_script)],
            cwd=str(project_root),
            check=True,
            text=True,
        )
        print()
        print("✓ STAGE 1 COMPLETED SUCCESSFULLY")
        print(f"  Checkpoint saved to: checkpoints/ppo_final.pt")
        print(f"  Logs saved to: logs/training_log.csv")
    except subprocess.CalledProcessError as e:
        print()
        print("✗ STAGE 1 FAILED")
        print(f"  Error: {e}")
        sys.exit(1)

    print()
    print("=" * 70)

    # ─────────────────────────────────────────────────────────
    #  STAGE 2: Traps and Traffic
    # ─────────────────────────────────────────────────────────
    print("▶ STAGE 2: Training with traps and traffic...")
    print("-" * 70)

    stage2_script = training_dir / "train_ppo_stage2.py"

    try:
        result = subprocess.run(
            [sys.executable, str(stage2_script)],
            cwd=str(project_root),
            check=True,
            text=True,
        )
        print()
        print("✓ STAGE 2 COMPLETED SUCCESSFULLY")
        print(f"  Phase A checkpoint: checkpoints/stage2_phaseA_final.pt")
        print(f"  Phase B checkpoint: checkpoints/stage2_phaseB_final.pt")
        print(f"  Phase A logs: logs/stage2_phaseA_log.csv")
        print(f"  Phase B logs: logs/stage2_phaseB_log.csv")
    except subprocess.CalledProcessError as e:
        print()
        print("✗ STAGE 2 FAILED")
        print(f"  Error: {e}")
        sys.exit(1)

    print()
    print("=" * 70)
    print("  ✓ COMPLETE TRAINING PIPELINE FINISHED")
    print("=" * 70)

if __name__ == "__main__":
    run_training_pipeline()