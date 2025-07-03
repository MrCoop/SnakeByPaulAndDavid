import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from snake_env import SnakeGame

from DQN import DQN

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

        states = torch.FloatTensor(np.array(states)).to(self.device)
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