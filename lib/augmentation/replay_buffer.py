import copy
import random

import torch.nn as nn


class ReplayBuffer(nn.Module):
    def __init__(self, decay_rate=0.9, buffer_size=-1):
        super().__init__()
        self.decay_rate = decay_rate
        self.buffer = nn.ModuleList([])
        self.priority = []
        self.buffer_size = buffer_size

    def store(self, augmentation):
        self.buffer.append(copy.deepcopy(augmentation))
        self.priority.append(1)
        self.priority = list(map(lambda x: self.decay_rate * x, self.priority)) # decay
        if self.buffer_size > 0 and len(self.priority) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]
            self.priority = self.priority[-self.buffer_size:]

    def sampling(self, n_samples, latest_aug=None):
        if latest_aug is not None:
            buffer = list(self.buffer._modules.values()) + [latest_aug]
            priority = self.priority + [1]
        else:
            buffer = self.buffer
            priority = self.priority
        return random.choices(buffer, priority, k=n_samples)

    def __len__(self):
        return len(self.buffer)

    def initialize(self, length, module):
        # This function must be called before the "load_state_dict" function.
        # placeholder to load state dict
        self.buffer = nn.ModuleList([copy.deepcopy(module) for _ in range(length)])
        self.priority = [self.decay_rate**(i+1) for i in reversed(range(length))]
