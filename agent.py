from random import randint, choice
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Agent:
    def __init__(self, state_size=2, action_size=5):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=1000)
        self.batch_size = 32
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.tau = 0.01

        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        self.action_map = np.linspace(0, 2, action_size)

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if random.random() <= self.epsilon:
            discrete_action = random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                discrete_action = q_values.argmax().item()

        # Convert discrete action to continuous water amount
        return np.array([self.action_map[discrete_action]])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample minibatch
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*minibatch)

        # Convert to tensors
        states_np = np.array(states)
        states = torch.FloatTensor(states_np)

        actions = torch.LongTensor([np.where(self.action_map == action[0])[0][0]
                                    for action in actions])
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        # Get current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Get next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]

        # Compute target Q values
        target_q_values = rewards + self.gamma * next_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network
        self.update_target_network()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def update_target_network(self):
        """Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, policy_param in zip(self.target_net.parameters(),
                                              self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data +
                                    (1.0 - self.tau) * target_param.data)