from collections import deque
import random


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size) if len(self.memory) >= batch_size else self.memory

    def __len__(self):
        return len(self.memory)
