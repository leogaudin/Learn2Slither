from collections import deque
import random
import torch
from model import DQN
from trainer import Trainer
from settings import device


class Agent:
    def __init__(self, gamma, epsilon, max_memory, batch_size, lr):
        self.gamma = gamma  # Discount factor, how much we value future rewards
        self.epsilon = epsilon  # Exploration vs exploitation trade-off
        self.memory = deque(maxlen=max_memory)
        self.batch_size = batch_size
        self.model = DQN(
            n_observations=9,
            n_actions=3,
        ).to(device)
        self.trainer = Trainer(self.model, lr, gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        mini_sample = (
            self.memory
            if len(self.memory) < self.batch_size
            else random.sample(self.memory, self.batch_size)
        )

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def get_action(self, state) -> list[float]:
        action = [0, 0, 0]

        if random.uniform(0, 1) < self.epsilon:
            choice = random.randint(0, 2)
            action[choice] = 1
            print(f"Random action: {action}")
        else:
            state = torch.tensor(
                state,
                dtype=torch.float32
            ).to(device)
            prediction = self.model(state)
            choice = torch.argmax(prediction).item()
            action[choice] = 1
            print(f"Predicted action: {action}")

        return action
