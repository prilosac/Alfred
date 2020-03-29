import random
from collections import namedtuple

import torch
import torch.nn as nn
# import torch.optim as optim
import torch.nn.functional as F
# import torchvision.transforms as T

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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



# from collections import deque
# import numpy as np

class DQN(nn.Module):
    def __init__(self, inputs, outputs, kernelSize):
        super(DQN, self).__init__()
        self.kernelSize = kernelSize
        self.padding = kernelSize // 2

        self.dropout1 = nn.Dropout(p=0.75)
        # self.conv1 = nn.Conv1d(inputs, 16, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv1d(inputs, 16, kernel_size=self.kernelSize, stride=1, padding=self.padding)
        self.bn1 = nn.BatchNorm1d(16)
        self.dropout2 = nn.Dropout(p=0.75)
        # self.conv2 = nn.Conv1d(16, 32, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=self.kernelSize, stride=1, padding=self.padding)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(p=0.75)
        # self.conv3 = nn.Conv1d(32, 32, kernel_size=1, stride=1)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=self.kernelSize, stride=1, padding=self.padding)
        self.bn3 = nn.BatchNorm1d(32)

        # def conv2d_size_out(size, kernel_size = 1, stride = 1):
        #     return (size - (kernel_size - 1) - 1) // stride  + 1
        # convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(inputs)))
        # linear_input_size = convw * 32
        self.head = nn.Linear(32, outputs)

        # self.to(device)

    # def forward(self, x):
    #     x = F.relu(self.conv1(x))
    #     return F.relu(self.conv2(x))

    def forward(self, x):
        # print(x)
        x = F.relu(self.bn1(self.conv1(self.dropout1(x))))
        # x = F.relu(self.conv1(x))
        x = F.relu(self.bn2(self.conv2(self.dropout2(x))))
        x = F.relu(self.bn3(self.conv3(self.dropout3(x))))
        # return x
        # print(x)

        # print(torch.mean(x.view(x.size(0), -1), 0, keepdim=True).shape)
        # print(self.head(x.view(x.size(0), -1)))
        # print(x.shape)
        preHead = x.view(x.size(0), -1)
        # preHead = torch.mean(x.view(x.size(0), -1), 0, keepdim=True)
        return nn.Softmax(dim=1)(self.head(preHead))







    # def __init__(self, state_size, action_size):
    #     # this is the SHAPE of the state (same shape as state
    #     # output from console)
    #     self.state_size = state_size
    #     # shape of action space, i.e. object with 1 boolean field
    #     # for each button and a number field for analog inputs
    #     self.action_size = action_size
    #     self.memory = deque(maxlen=2000)
    #     self.gamma = 0.95 # discount
    #     self.epsilon = 1.0 # exploration
    #     self.epsilon_min = 0.01
    #     self.epsilon_decay = 0.995
    #     self.learning_rate = 0.001
    #     self.model = self._build_model()

#     def _build_model(self):
#         # Build the Neural Network
#         return []

#     def remember(self, state, action, reward, new_state, done):
#         self.memory.append((state, action, reward, new_state, done))

#     def getAction(self, state):
#         if np.random.rand() <= self.epsilon:
#             # Do a random action
#             # return random.rand_range(self.action_size)
#             return
#         action = self.model.predict(state)
#         return np.argmax(action[0])