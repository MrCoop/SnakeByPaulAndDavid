import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from snake_env import SnakeGame

class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)


class DQNAgent:
    def __init__(self, state_size, action_size, hidden_dim=128, gamma=0.99, lr=1e-3,
                 buffer_size=10000, batch_size=64, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.memory = deque(maxlen=buffer_size)

        self.policy_net = DQN(state_size, hidden_dim, action_size).to(self.device)
        self.target_net = DQN(state_size, hidden_dim, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.policy_net(state).argmax().item()

    def remember(self, transition):
        self.memory.append(transition)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].detach().unsqueeze(1)
        target_q = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon reduzieren
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay



env = SnakeGame(grid_size=10)
state_dim = env.grid_size * env.grid_size
action_dim = 4

agent = DQNAgent(state_dim, action_dim)
episodes = 300
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