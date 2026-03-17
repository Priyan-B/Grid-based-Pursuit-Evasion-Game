import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PPOPolicy(nn.Module):

    def __init__(self, state_size, action_size):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.policy_head = nn.Linear(64, action_size)
        self.value_head = nn.Linear(64, 1)

    def forward(self, state):

        x = self.shared(state)

        action_logits = self.policy_head(x)
        value = self.value_head(x)

        return action_logits, value

class PPOAgent:

    def __init__(self, state_size, action_size):

        self.policy = PPOPolicy(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.001)

        self.gamma = 0.99

    def select_action(self, state):

        state = torch.tensor(state, dtype=torch.float32)

        logits, value = self.policy(state)

        dist = Categorical(logits=logits)

        action = dist.sample()

        log_prob = dist.log_prob(action)

        return action.item(), log_prob, value

    def update(self, log_probs, values, rewards):

        returns = []
        G = 0

        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns)

        values = torch.stack(values).squeeze()

        advantage = returns - values.detach()

        log_probs = torch.stack(log_probs)

        policy_loss = -(log_probs * advantage).mean()

        value_loss = (returns - values).pow(2).mean()

        loss = policy_loss + 0.5 * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()