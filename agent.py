from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
from model import DQN
from settings import config


class Agent:
    def __init__(self, gamma, epsilon_init, max_memory, batch_size, lr):
        """ Agent class for the DQN algorithm
        """
        self.gamma = gamma  # Discount factor, how much we value future rewards
        self.epsilon = epsilon_init  # Exploration vs exploitation trade-off
        self.memory = deque(maxlen=max_memory)
        self.batch_size = batch_size
        self.model = DQN(
            n_observations=13,
            n_actions=3,
        ).to(config['device'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        """ Store the current step in the memory
        """
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self, state, action, reward, next_state, done):
        """ Train the model given a single step or a batch of steps
        """
        state = torch.tensor(state, dtype=torch.float32,
                             device=config['device'])
        action = torch.tensor(action, dtype=torch.float32,
                              device=config['device'])
        reward = torch.tensor(reward, dtype=torch.float32,
                              device=config['device'])
        next_state = torch.tensor(next_state, dtype=torch.float32,
                                  device=config['device'])

        if len(next_state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            done = (done,)

        prediction = self.model(state)
        target = prediction.clone()

        for i in range(len(done)):
            with torch.no_grad():
                Q_new = reward[i]
                if not done[i]:
                    Q_new = reward[i] + (self.gamma *
                                         torch.max(self.model(next_state[i])))

            target[i][action[i].argmax().item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()
        if config['debug']:
            print('Loss:', loss)
        self.optimizer.step()

    def train_short_memory(self, state, action, reward, next_state, done):
        """ Train the model given a single step
        """
        self.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        """ Train the model given a batch of steps
        """
        mini_sample = (
            self.memory
            if len(self.memory) < self.batch_size
            else random.sample(self.memory, self.batch_size)
        )

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.train_step(states, actions, rewards, next_states, dones)

    def get_action(self, state) -> list[float]:
        """ Get the action to take given the current state, based on the
            epsilon-greedy policy
        """
        action = [0, 0, 0]

        if random.uniform(0, 1) < self.epsilon:
            choice = random.randint(0, 2)
            action[choice] = 1
            if config['debug']:
                print(f"Random action: {action}")
        else:
            with torch.no_grad():
                state = torch.tensor(
                    state,
                    dtype=torch.float32,
                    device=config['device'],
                )
                prediction = self.model(state)
                choice = prediction.argmax().item()
                action[choice] = 1
                if config['debug']:
                    print(f"Predicted action: {action}")

        return action
