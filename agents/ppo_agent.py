"""
PPO Agent with clipped surrogate objective, GAE, entropy bonus,
multi-epoch minibatch updates, gradient clipping, and LR annealing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

def _ortho_init(layer, gain=np.sqrt(2)):
    """Orthogonal initialisation — standard for PPO networks."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class PPOPolicy(nn.Module):
    """Actor-Critic with shared trunk, split heads."""

    def __init__(self, state_size: int, action_size: int, hidden: int = 128):
        super().__init__()

        self.shared = nn.Sequential(
            _ortho_init(nn.Linear(state_size, hidden)),
            nn.Tanh(),
            _ortho_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
        )

        # Policy head — small init so early actions are near-uniform
        self.policy_head = _ortho_init(nn.Linear(hidden, action_size), gain=0.01)
        # Value head — small init so early values are near zero
        self.value_head = _ortho_init(nn.Linear(hidden, 1), gain=1.0)

    def forward(self, state):
        x = self.shared(state)
        return self.policy_head(x), self.value_head(x)

    def get_action_and_value(self, state, action=None):
        """
        If action is None  → sample a new action.
        If action is given → evaluate that action (for the PPO update).
        Returns: action, log_prob, entropy, value
        """
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        return (
            action,
            dist.log_prob(action),
            dist.entropy(),
            value.squeeze(-1),
        )


# ---------------------------------------------------------------------------
# Rollout Buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores one rollout and computes GAE returns on demand."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def store(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    @property
    def size(self):
        return len(self.rewards)

    def compute_gae(self, gamma: float, gae_lambda: float, last_value: float, last_done: bool):
        """
        Generalized Advantage Estimation (Schulman et al., 2016).

        Computes per-step advantages and discounted returns using the
        TD-residual formula:
            δ_t     = r_t + γ · V(s_{t+1}) · (1 - done_{t+1}) − V(s_t)
            A_t^GAE = Σ_{l=0}^{T-t} (γλ)^l · δ_{t+l}

        Returns are then  A_t + V(s_t)  so the value head regresses to them.
        """
        T = self.size
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(last_done)
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - float(self.dones[t + 1])

            delta = (
                self.rewards[t]
                + gamma * next_value * next_non_terminal
                - self.values[t]
            )
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        returns = advantages + np.array(self.values, dtype=np.float32)
        return advantages, returns

    def to_tensors(self, advantages, returns):
        """Pack everything into tensors for minibatch sampling."""
        return dict(
            states=torch.tensor(np.array(self.states), dtype=torch.float32),
            actions=torch.tensor(np.array(self.actions), dtype=torch.long),
            old_log_probs=torch.tensor(
                np.array(self.log_probs), dtype=torch.float32
            ),
            old_values=torch.tensor(
                np.array(self.values), dtype=torch.float32
            ),
            advantages=torch.tensor(advantages, dtype=torch.float32),
            returns=torch.tensor(returns, dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent:
    """
    Proximal Policy Optimisation (clip variant).

    Key hyper-parameters and what they do:
    ─────────────────────────────────────────────────────────────────────
    clip_eps        PPO clipping range ε.  Keeps the policy update
                    conservative: ratio r_t(θ) is clamped to [1-ε, 1+ε].
    gamma           Discount factor for future rewards.
    gae_lambda      λ for GAE.  λ=1 → Monte-Carlo returns (high variance),
                    λ=0 → one-step TD (high bias).  0.95 is a good default.
    vf_coef         Weight of value-function loss in the total loss.
    vf_clip         Clip range for the value function.  Prevents the value
                    head from overshooting wildly, which would blow up the
                    shared backbone and kill the policy head with NaNs.
    ent_coef        Entropy bonus coefficient — encourages exploration.
    max_grad_norm   Gradient clipping threshold (L2 norm).
    n_epochs        How many passes over the rollout per update.
    batch_size      Minibatch size within each epoch.
    min_batch_size  Minimum minibatch size — skip smaller tail batches
                    to avoid NaN from std() on 1-element tensors.
    lr              Initial learning rate.
    lr_end          Final learning rate (for linear annealing).
    total_updates   Total number of update() calls expected during
                    training; used to schedule LR and entropy annealing.
    ent_coef_end    Final entropy coefficient (linearly decayed).
    ─────────────────────────────────────────────────────────────────────
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        # --- core PPO ---
        clip_eps: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        vf_coef: float = 0.5,
        vf_clip: float = 10.0,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        # --- update schedule ---
        n_epochs: int = 4,
        batch_size: int = 64,
        min_batch_size: int = 8,
        # --- optimiser ---
        lr: float = 3e-4,
        lr_end: float = 3e-5,
        # --- annealing ---
        total_updates: int = 1000,
        ent_coef_end: float = 0.001,
    ):
        self.policy = PPOPolicy(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)

        # Store all hypers
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.vf_coef = vf_coef
        self.vf_clip = vf_clip
        self.ent_coef_start = ent_coef
        self.ent_coef_end = ent_coef_end
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.min_batch_size = min_batch_size

        self.lr_start = lr
        self.lr_end = lr_end
        self.total_updates = total_updates

        self.buffer = RolloutBuffer()
        self._update_count = 0

    # ----- action selection (used during rollout collection) -----

    @torch.no_grad()
    def select_action(self, state):
        """
        Sample action from current policy.
        Returns: action (int), log_prob (float), value (float)
        """
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action, log_prob, _, value = self.policy.get_action_and_value(state_t)
        return action.item(), log_prob.item(), value.item()

    @torch.no_grad()
    def estimate_value(self, state):
        """Bootstrap value for the last state in a rollout."""
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        _, value = self.policy(state_t)
        return value.item()

    # ----- buffer helpers -----

    def store_transition(self, state, action, log_prob, reward, value, done):
        self.buffer.store(state, action, log_prob, reward, value, done)

    # ----- PPO update -----

    def update(self, last_state, last_done):
        """
        Run the full PPO update:
        1. Compute GAE advantages + returns from the rollout buffer.
        2. Normalise advantages GLOBALLY over the whole rollout (not
           per-minibatch, which caused NaN on single-element batches).
        3. For n_epochs, shuffle data into minibatches and take
           clipped-objective gradient steps.
        4. Anneal LR and entropy coefficient.
        5. Clear buffer.

        Returns a dict of training metrics for logging.
        """
        self._update_count += 1

        # --- schedule: linear annealing of LR and entropy coef ---
        frac = min(self._update_count / max(self.total_updates, 1), 1.0)
        new_lr = self.lr_start + frac * (self.lr_end - self.lr_start)
        for pg in self.optimizer.param_groups:
            pg["lr"] = new_lr
        self.ent_coef = self.ent_coef_start + frac * (
            self.ent_coef_end - self.ent_coef_start
        )

        # --- compute GAE ---
        last_value = self.estimate_value(last_state)
        advantages, returns = self.buffer.compute_gae(
            self.gamma, self.gae_lambda, last_value, last_done
        )

        # --- pack into tensors ---
        batch = self.buffer.to_tensors(advantages, returns)
        B = batch["states"].shape[0]

        # --- GLOBAL advantage normalisation (safe even for B=1) ---
        adv = batch["advantages"]
        if B > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        batch["advantages"] = adv

        # --- tracking ---
        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_ent = 0.0
        total_clipfrac = 0.0
        n_updates = 0

        for _epoch in range(self.n_epochs):
            indices = np.arange(B)
            np.random.shuffle(indices)

            for start in range(0, B, self.batch_size):
                end = start + self.batch_size
                mb_idx = indices[start:end]

                # --- skip tiny tail minibatches to avoid degenerate grads ---
                if len(mb_idx) < self.min_batch_size:
                    continue

                mb_states = batch["states"][mb_idx]
                mb_actions = batch["actions"][mb_idx]
                mb_old_lp = batch["old_log_probs"][mb_idx]
                mb_old_val = batch["old_values"][mb_idx]
                mb_adv = batch["advantages"][mb_idx]
                mb_ret = batch["returns"][mb_idx]

                # --- evaluate actions under current policy ---
                _, new_log_prob, entropy, new_value = (
                    self.policy.get_action_and_value(mb_states, mb_actions)
                )

                # --- PPO clipped surrogate objective ---
                log_ratio = new_log_prob - mb_old_lp
                ratio = torch.exp(log_ratio)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # --- CLIPPED value loss ---
                # This is what prevents the NaN cascade. Without it:
                #   return ≈ 100, old_value ≈ 0 → MSE ≈ 10,000
                #   → huge gradient through shared trunk → weights explode
                #   → policy head outputs NaN logits.
                # With clipping, the value head can only move ±vf_clip per
                # update, so the gradient stays bounded.
                vf_unclipped = (new_value - mb_ret).pow(2)
                vf_clipped_pred = mb_old_val + torch.clamp(
                    new_value - mb_old_val,
                    -self.vf_clip, self.vf_clip,
                )
                vf_clipped = (vf_clipped_pred - mb_ret).pow(2)
                vf_loss = 0.5 * torch.max(vf_unclipped, vf_clipped).mean()

                # --- entropy bonus ---
                ent_loss = entropy.mean()

                # --- total loss ---
                loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # --- track metrics ---
                with torch.no_grad():
                    clip_frac = (
                        (torch.abs(ratio - 1.0) > self.clip_eps).float().mean()
                    )
                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_ent += ent_loss.item()
                total_clipfrac += clip_frac.item()
                n_updates += 1

        self.buffer.clear()

        # Guard against zero updates (very short episode, all batches skipped)
        if n_updates == 0:
            return {
                "pg_loss": 0.0, "vf_loss": 0.0, "entropy": 0.0,
                "clip_frac": 0.0, "lr": new_lr, "ent_coef": self.ent_coef,
            }

        return {
            "pg_loss": total_pg_loss / n_updates,
            "vf_loss": total_vf_loss / n_updates,
            "entropy": total_ent / n_updates,
            "clip_frac": total_clipfrac / n_updates,
            "lr": new_lr,
            "ent_coef": self.ent_coef,
        }

    # ----- persistence -----

    def save(self, path: str):
        torch.save({
            "policy_state": self.policy.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "update_count": self._update_count,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, weights_only=True)
        self.policy.load_state_dict(ckpt["policy_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self._update_count = ckpt["update_count"]