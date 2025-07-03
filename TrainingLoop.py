import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from snake_env import SnakeGame
import matplotlib.pyplot as plt

from DQNAgent import DQNAgent

env = SnakeGame(grid_size=10,)
state_dim = env.grid_size * env.grid_size
action_dim = 4

agent = DQNAgent(state_dim, action_dim)
episodes = 10000
target_update_interval = 10
scores = []

for episode in range(episodes):
    state = env.reset()
    total_reward = 0

    for t in range(200):  # Max 200 steps
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember((state, action, reward, next_state, float(done)))
        agent.train_step()

        state = next_state
        total_reward += reward

        if done:
            break

    if episode % target_update_interval == 0:
        agent.update_target()

    scores.append(total_reward)
    print(f"Episode {episode} | Score: {env.score} | Epsilon: {agent.epsilon:.3f}")

env.close()

torch.save(agent.policy_net.state_dict(), 
    r"C:\Users\kleinds\OneDrive - AbbVie Inc (O365)\Documents\DHBW\6_Semester\Aktuelle Data Science Entwicklungen\Implementierung\SnakeByPaulAndDavid\dqn_snake1.pth")


plt.plot(scores)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Performance')
plt.grid()
plt.show()