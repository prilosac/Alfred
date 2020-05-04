from os import path
import random
from collections import namedtuple
from math import exp
import numpy as np
import util
import copy
from agents.deepQNetwork import DQN, ReplayMemory
import torch.optim as optim
import torch
import torch.nn.functional as F
import p3.state as p3state

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 100

class QLearningAgent:
    def __init__(self, charActions, learningRate=0.1, discountRate=0.95, explorationRate=1.0, explorationDecay=0.0005, explorationRateMin=0.01, model="nosave"):
        """Initialize here"""
        self.actions = charActions
        self.learningRate = learningRate
        self.discountRate = discountRate
        self.explorationRateMax = explorationRate
        self.explorationRate = explorationRate
        self.explorationDecay = explorationDecay
        self.explorationRateMin = explorationRateMin
        self.predictionFrames = 120
        self.kernelSize = 5
        

        # self.QValues = util.myDict()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policyNet = DQN(15, len(self.actions), self.kernelSize).to(self.device)
        self.policyNet = self.policyNet.double()
        self.targetNet = DQN(15, len(self.actions), self.kernelSize).to(self.device)
        self.targetNet = self.targetNet.double()
        self.compareDict = None
        self.previousModelDelta = {}

        if model != "nosave" and model != "test" and path.exists("models/"+model):
            self.policyNet.load_state_dict(torch.load("models/" + model))
        
        self.targetNet.load_state_dict(self.policyNet.state_dict())
        self.targetNet.eval()

        self.optimizer = optim.Adam(self.policyNet.parameters(), self.learningRate)
        self.memory = ReplayMemory(1000)

        self.rewardMemory = util.StateMemory(self.predictionFrames*2)

    def getAction(self, state):
        if(random.random() <= self.explorationRate):
            return self.randomAction()
        return self.policy(state)

    def randomAction(self):
        # Returns a list of button inputs (may be of size 1).
        # return self.actions[np.random.randint(0, len(self.actions))]
        return random.randint(0, len(self.actions)-1)

    def policy(self, state):
        # Exit training mode to get an answer based on policy, then return to training mode\
        self.policyNet.train(mode=False)

        states = [state]
        for i in range(self.kernelSize-1):
            # index will be one less than size, and the last element is just the current state, so add 2 to i to make sure the element exists
            if len(self.memory) >= (i+2):
                s = self.memory.memory[len(self.memory)-(i+2)][2]
                states.append(s)
            else:
                states.append(copy.deepcopy(state))

        predictState = torch.unsqueeze(torch.cat(states), 2)
        policyAns = self.policyNet(predictState)

        self.policyNet.train(mode=True)

        avgPolicyAns = torch.mean(policyAns.detach(), 0, keepdim=True)

        return self.actions.index(np.random.choice(self.actions, p=torch.squeeze(avgPolicyAns).detach().numpy()))

    def getQValue(self, state, action):
        # Return predicted Q values for state
        # Model input: state
        # Model output: Array of Q values for single state
        return self.QValues[(state, tuple(action))]

    def getValue(self, state):
        return max([self.getQValue(state, action) for action in self.actions])        

    def getReward(self):
        newState = self.rewardMemory.last()
        oldState = self.rewardMemory.first()

        p3p = newState.players[2].__dict__['percent'] - oldState.players[2].__dict__['percent']
        p1p = newState.players[0].__dict__['percent'] - oldState.players[0].__dict__['percent']
        p3s = oldState.players[2].__dict__['stocks'] - newState.players[2].__dict__['stocks']
        p1s = oldState.players[0].__dict__['stocks'] - newState.players[0].__dict__['stocks']

        ans = (0.0075*(p1p) - 0.0075*(p3p))

        # don't punish the same death more than once
        # just because it occurs over multiple frames
        # don't reward the same kill more than once 
        # just because it occurs over multiple frames
        if self.rewardMemory.died(2):
            ans -= 1.0
        if self.rewardMemory.died(0):
            ans += 1.0

        return ans

    def update(self, oldState, newState, actionIndex):
        action = self.actions[actionIndex]

        self.predictFuture(newState, self.predictionFrames)
        reward = self.getReward()
        reward = torch.tensor([reward], device=self.device)

        self.memory.push(self.getStateRep(oldState), torch.tensor(actionIndex), self.getStateRep(newState), torch.tensor(reward, dtype=torch.double))

        # shift exploration_rate toward zero (less gambling)
        if self.explorationRate > self.explorationRateMin:
            self.explorationRate = self.explorationRateMin + (self.explorationRateMax - \
                self.explorationRateMin) * \
                    exp(-1. * newState.frame / self.explorationDecay)

        self.optimize_model()

        # Update the target network, copying all weights and biases in DQN
        if oldState.frame % 100 == 0:
            # if self.compareDict is not None:
            # self.compare_models(self.policyNet.state_dict(), self.targetNet.state_dict())
            self.targetNet.load_state_dict(self.policyNet.state_dict())

    def optimize_model(self):
        
        if self.memory.position % BATCH_SIZE != 0 or len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),  device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.stack(batch.action, dim=0)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken. These are the actions which would've been taken for each batch state according to policy_net print(state_batch.shape)
        state_action_values = self.policyNet(torch.unsqueeze(state_batch, 2)).gather(1, torch.unsqueeze(action_batch, 1))

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device, dtype=torch.double)
        next_state_values[non_final_mask] = self.targetNet(torch.unsqueeze(non_final_next_states, 2)).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.discountRate) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policyNet.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
            else:
                continue
        self.optimizer.step()

    def printState(self, state):
        print("Player 1")
        print("---------------------")
        for attr, value in state.players[0].__dict__.items():
            print(attr, value)
        print("---------------------")
        print("Player 3")
        print("---------------------")
        for attr, value in state.players[2].__dict__.items():
            print(attr, value)
        for attr, value in state.__dict__.items():
            if(attr != "players"):
                print(attr, value)
        print("---------------------")

    def printStocks(self, state):
        p1Stocks = state.players[0].__dict__['stocks']
        p1Percent = state.players[0].__dict__['percent']
        p3Stocks = state.players[2].__dict__['stocks']
        p3Percent = state.players[2].__dict__['percent']
        print("Player 1    |    Alfred")
        print("------------|----------")
        print("   ", p1Stocks, "      |", "   ", p3Stocks)
        print("   ", p1Percent, "    |", "   ", p3Percent)
        print("------------|----------")

    def getStateRep(self, state):
        qState = (
            state.players[0].__dict__['stocks'],
            state.players[0].__dict__['percent'],
            state.players[0].__dict__['cursor_x'],
            state.players[0].__dict__['cursor_y'],
            state.players[0].__dict__['pos_x'],
            state.players[0].__dict__['pos_y'],
            state.players[0].__dict__['action_state'].value,
            state.players[2].__dict__['stocks'],
            state.players[2].__dict__['percent'],
            state.players[2].__dict__['cursor_x'],
            state.players[2].__dict__['cursor_y'],
            state.players[2].__dict__['pos_x'],
            state.players[2].__dict__['pos_y'],
            state.players[2].__dict__['action_state'].value,
            state.__dict__['stage'].value,
        )
        qState = torch.unsqueeze(torch.tensor(list(qState), dtype=torch.float64, device=self.device), 0)
        qState = F.normalize(qState)
        m = qState.mean(1, keepdim=True)
        s = qState.std(1, unbiased=False, keepdim=True)
        qState -= m
        qState /= s
        return qState

    def predictFuture(self, state, frames):
        # p1/p3 - Player 1 / Player 3
        # v - Velocity
        # air - Air
        # att - Attack
        # pos - Position
        if frames <= 0:
            return

        predictedState = copy.deepcopy(state)
        p1airvx = state.players[0].__dict__['self_air_vel_x']
        p1airvy = state.players[0].__dict__['self_air_vel_y']
        p1attvx = state.players[0].__dict__['attack_vel_x']
        p1attvy = state.players[0].__dict__['attack_vel_y']
        p1posx = state.players[0].__dict__['pos_x']
        p1posy = state.players[0].__dict__['pos_y']

        p3airvx = state.players[2].__dict__['self_air_vel_x']
        p3airvy = state.players[2].__dict__['self_air_vel_y']
        p3attvx = state.players[2].__dict__['attack_vel_x']
        p3attvy = state.players[2].__dict__['attack_vel_y']
        p3posx = state.players[2].__dict__['pos_x']
        p3posy = state.players[2].__dict__['pos_y']

        if p3posy < 0.0:
            predictedState.players[2].__dict__['stocks'] = np.max(predictedState.players[2].__dict__['stocks']-1, 0)
            predictedState.players[2].__dict__['action_state'] = p3state.ActionState.DeadDown
        predictedState.players[2].__dict__['pos_x'] += (p3airvx + p3attvx)*2
        predictedState.players[2].__dict__['pos_y'] += (p3airvy + p3attvy)*2

        if p1posy < 0.0:
            predictedState.players[0].__dict__['stocks'] = np.max(predictedState.players[0].__dict__['stocks']-1, 0)
            predictedState.players[0].__dict__['action_state'] = p3state.ActionState.DeadDown
        predictedState.players[0].__dict__['pos_x'] += (p1airvx + p1attvx)*2
        predictedState.players[0].__dict__['pos_y'] += (p1airvy + p1attvy)*2

        self.rewardMemory.push(predictedState)
        self.predictFuture(predictedState, frames-1)

    def compare_models(self, model_1, model_2):
        print('------------------------------')
        models_differ = 0
        deltas_differ = 0
        runningDelta = 0
        runningDeltaCount = 0
        for key_item_1, key_item_2 in zip(model_1.items(), model_2.items()):
            if not ('weight' in key_item_1[0] and 'weight' in key_item_2[0]):
                continue

            diff = abs(key_item_1[1] - key_item_2[1])
            runningDelta += diff.sum().item()
            runningDeltaCount += float(diff.numel())
            if torch.allclose(key_item_1[1], key_item_2[1]):
                pass
            # if torch.allclose(key_item_1.grad, key_item_2.grad, rtol=1e-03, atol=1e-05):
            #     pass
            else:
                models_differ += 1
                if (key_item_1[0] == key_item_2[0]):
                    print('Mismtach found at', key_item_1[0])
                    # print('Mismtach of', diff, 'found at', key_item_1[0])
                    pass
                else:
                    raise Exception

            if key_item_1[0] in self.previousModelDelta:
                # print(self.previousModelDelta)
                if torch.allclose(diff, self.previousModelDelta[key_item_1[0]]):
                    pass
                else:
                    deltas_differ += 1
                    print('Mismatch found at', key_item_1[0], '[delta]')
            else:
                deltas_differ = -1
            
            self.previousModelDelta[key_item_1[0]] = diff

        # filemode = 'w'
        # if path.exists('./weightAverages.txt'):
        #     filemode = 'a'
        
        # with open('weightAverages.txt', filemode) as f:
        #     f.write(str(runningDelta/runningDeltaCount) + ' ')
        #     f.close

        if models_differ == 0:
            print('Models match within tolerance!')
        if deltas_differ == 0:
            print('Model deltas match within tolerance!')