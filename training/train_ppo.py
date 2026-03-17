from env.grid_world import GridWorld
from agents.ppo_agent import PPOAgent
from utils.visualize import show_grid, reset_view

import numpy as np


env = GridWorld(size=15)

state_size = env.OBS_DIM   # 9 (3x3 local grid)
action_size = 5            # up, down, left, right, stay

agent = PPOAgent(state_size, action_size)

episodes = 2000

# ---- visualisation control ----
# Show the agent moving live every RENDER_EVERY episodes.
# Set to 1 to watch every episode (slow), 10-50 for a good balance.
RENDER_EVERY = 20
PAUSE = 0.02              # seconds per frame when rendering


for ep in range(episodes):

    state = env.reset()
    done = False

    log_probs = []
    values = []
    rewards = []

    total_reward = 0

    render_this_ep = (ep % RENDER_EVERY == 0)
    path = [env.agent_pos] if render_this_ep else None

    while not done:

        action, log_prob, value = agent.select_action(state)

        next_state, reward, done = env.step(action)

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)

        state = next_state
        total_reward += reward

        # ---- live render ----
        if render_this_ep:
            path.append(env.agent_pos)
            show_grid(
                env.render(), env.goal,
                agent_pos=env.agent_pos,
                path=path,
                title=f'Episode {ep+1}  |  Step {env.steps}  |  R={total_reward:.0f}',
                pause=PAUSE
            )

    agent.update(log_probs, values, rewards)

    if render_this_ep:
        reset_view()

    if (ep + 1) % 100 == 0:
        print(f"Episode {ep+1:>5}/{episodes}  Reward: {total_reward:.1f}")

print("\nTraining done.")