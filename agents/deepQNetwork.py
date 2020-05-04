import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    # remembers a transition
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batchSize):
        return random.sample(self.memory, batchSize)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, inputs, outputs, kernelSize):
        super(DQN, self).__init__()
        self.kernelSize = kernelSize
        self.padding = kernelSize // 2

        self.dropout1 = nn.Dropout(p=0.75)
        # self.dropout1 = nn.Dropout(p=0.4)
        self.conv1 = nn.Conv1d(inputs, 32, kernel_size=self.kernelSize, stride=1, padding=self.padding)
        self.bn1 = nn.BatchNorm1d(32)

        self.dropout2 = nn.Dropout(p=0.85)
        # self.dropout2 = nn.Dropout(p=0.1)
        self.conv2 = nn.Conv1d(32, 128, kernel_size=self.kernelSize, stride=1, padding=self.padding)
        self.bn2 = nn.BatchNorm1d(128)

        self.dropout3 = nn.Dropout(p=0.90)
        # self.dropout3 = nn.Dropout(p=0.01)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=self.kernelSize, stride=1, padding=self.padding)
        self.bn3 = nn.BatchNorm1d(256)

        self.head = nn.Linear(256, outputs)

        self.to(device)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(self.dropout1(x))))
        x = F.relu(self.bn2(self.conv2(self.dropout2(x))))
        x = F.relu(self.bn3(self.conv3(self.dropout3(x))))

        preHead = x.view(x.size(0), -1)
        return nn.Softmax(dim=1)(self.head(preHead))