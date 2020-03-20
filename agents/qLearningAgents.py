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

BATCH_SIZE = 20

class QLearningAgent:
    def __init__(self, charActions, learningRate=0.1, discountRate=0.95, explorationRate=1.0, explorationDecay=0.0005, explorationRateMin=0.01, model="nosave"):
        """Initialize here"""
        # self.lastState = None
        self.actions = charActions
        self.learningRate = learningRate
        self.discountRate = discountRate
        self.explorationRateMax = explorationRate
        self.explorationRate = explorationRate
        self.explorationDecay = explorationDecay
        self.explorationRateMin = explorationRateMin
        self.predictionFrames = 120
        

        # self.QValues = util.myDict()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policyNet = DQN(15, len(self.actions)).to(self.device)
        self.policyNet = self.policyNet.double()
        self.targetNet = DQN(15, len(self.actions)).to(self.device)
        self.targetNet = self.targetNet.double()

        if model != "nosave" and path.exists("models/"+model):
            self.policyNet.load_state_dict(torch.load("models/" + model))
            # print(self.policyNet.state_dict())

        self.targetNet.load_state_dict(self.policyNet.state_dict())
        self.targetNet.eval()

        # self.optimizer = optim.RMSprop(self.policyNet.parameters())
        # self.optimizer = optim.RMSprop(self.policyNet.parameters(), lr=self.learningRate, weight_decay=2)
        self.optimizer = optim.Adam(self.policyNet.parameters(), self.learningRate)
        self.memory = ReplayMemory(1000)

        # self.lastObs = None
        self.rewardMemory = util.StateMemory(self.predictionFrames*2)

    def getAction(self, state):
        if(random.random() <= self.explorationRate):
            # print("random")
            return self.randomAction()
        # print("policy")
        return self.policy(state)

    def randomAction(self):
        # print("random")
        # Returns a list of button inputs (may be of size 1).
        # return self.actions[np.random.randint(0, len(self.actions))]
        # return random.choice(self.actions)
        return random.randint(0, len(self.actions)-1)

    def policy(self, state):
        # print("policy")
        # return (1, None, [])
        # ans = [None, -float("inf")]
        # for action in self.actions:
        #     tempQVal = self.getQValue(state, action)
        #     if tempQVal >= ans[1]:
        #         ans[0] = action
        #         ans[1] = tempQVal
        # return ans[0]
        # return the action with the highet Q from the neural net prediction
        # return self.actions[self.policyNet(torch.squeeze(torch.tensor(list(state)), 0)).max(1)[1].view(1, 1)]
        # return self.actions[self.policyNet(torch.unsqueeze(torch.tensor(list(state)), 1)).max(1)[1].view(1, 1)]

        # print(torch.unsqueeze(torch.unsqueeze(torch.tensor(list(state)), 1), 0))

        # Exit training mode to get an answer based on policy, then return to training mode\
        self.policyNet.train(mode=False)
        policyAns = self.policyNet(torch.unsqueeze(state, 2))
        # policyAns = self.policyNet(torch.stack(self.predictMemory.memory, dim=2))
        self.policyNet.train(mode=True)

        # print(np.random.choice(self.actions, p=torch.squeeze(policyAns.detach())))

        # print(torch.squeeze(policyAns))
        # print(policyAns.max(1)[1].view(1, 1).item(), torch.squeeze(policyAns)[policyAns.max(1)[1].view(1, 1).item()])
        return self.actions.index(np.random.choice(self.actions, p=torch.squeeze(policyAns.detach())))
        # return policyAns.max(1)[1].view(1, 1).item()

        # return self.actions[self.policyNet(torch.tensor(list(state))).max(1)[1].view(1, 1)]


     # Ask model to estimate Q value for specific state (inference)
    def getQValue(self, state, action):
        # Return predicted Q values for state
        # Model input: state
        # Model output: Array of Q values for single state
        
        # return self.session.run(self.model_output, feed_dict={self.model_input: self.to_one_hot(state)})[0]
        # print(type(state))
        # print(type(tuple(action)))
        return self.QValues[(state, tuple(action))]

    def getValue(self, state):
        # return np.amax([self.getQValue(state, action) for action in self.actions])
        return max([self.getQValue(state, action) for action in self.actions])        

    # def getReward(self, oldState, newState):
    def getReward(self):
        newState = self.rewardMemory.last()
        oldState = self.rewardMemory.first()
        # print([i.frame for i in self.rewardMemory.memory])
        # print("Reward over frames ", oldState.frame, " - ", newState.frame)
        # print(len(self.rewardMemory))

        p3p = newState.players[2].__dict__['percent'] - oldState.players[2].__dict__['percent']
        p1p = newState.players[0].__dict__['percent'] - oldState.players[0].__dict__['percent']
        p3s = oldState.players[2].__dict__['stocks'] - newState.players[2].__dict__['stocks']
        p1s = oldState.players[0].__dict__['stocks'] - newState.players[0].__dict__['stocks']
        # ans = (p1s-p3s) + (0.0075*(p1p) - 0.0075*(p3p))
        ans = (0.0075*(p1p) - 0.0075*(p3p))

        # print(newState.players[2].__dict__['percent'], oldState.players[2].__dict__['percent'], newState.players[0].__dict__['percent'], oldState.players[0].__dict__['percent'], oldState.players[2].__dict__['stocks'], newState.players[2].__dict__['stocks'], oldState.players[0].__dict__['stocks'], newState.players[0].__dict__['stocks'], " | oldFrame: ", oldState.frame, " | newFrame: ", newState.frame)
        # print("% 1  |  Stock 1  |  % 3  |  Stock 3" )
        # print(p1p, p1s, p3p, p3s)

        # don't punish the same death more than once
        # just because it occurs over multiple frames
        # don't reward the same kill more than once 
        # just because it occurs over multiple frames
        if self.rewardMemory.died(2):
            ans -= 1.0
        if self.rewardMemory.died(0):
            ans += 1.0
        
        # if(newState.players[2].__dict__['action_state'] == state.ActionState.Guard):
        #         ans -= 5
        return ans

    def update(self, oldState, newState, actionIndex):
        # self.lastObs = newState
        # self.rewardMemory.push(copy.deepcopy(newState))
        
        action = self.actions[actionIndex]

        # qState = self.getStateRep(oldState)

        
        # key = (qState, tuple(action))
        # if(key not in self.QValues.keys()):
        #     self.QValues[(qState, tuple(action))] = 0

        # reward = self.getReward(oldState, newState)
        self.predictFuture(newState, self.predictionFrames)
        reward = self.getReward()
        reward = torch.tensor([reward], device=self.device)

        # Store the transition in memory
        # print(actionIndex)

        # self.memory.push(torch.unsqueeze(torch.tensor(list(self.getStateRep(oldState)), dtype=torch.float64, device=self.device), 0), torch.tensor(actionIndex), torch.unsqueeze(torch.tensor(list(self.getStateRep(newState)), dtype=torch.float64, device=self.device), 0), torch.tensor(reward, dtype=torch.double))
        self.memory.push(self.getStateRep(oldState), torch.tensor(actionIndex), self.getStateRep(newState), torch.tensor(reward, dtype=torch.double))

        # if newState.frame % 500 == 0:
        #     self.printStocks(newState)
        # if(reward != 0.0):
        #     print("Reward: ", reward)

        # sample = reward + self.discountRate*self.getValue(newState)

        # self.QValues[(qState, tuple(action))] = (1-self.learningRate)*self.getQValue(qState, action) + self.learningRate*sample
        
        # # Train our model with new data
        # # self.train(oldState, action, reward, newState)

        # Finally shift our exploration_rate toward zero (less gambling)
        if self.explorationRate > self.explorationRateMin:
            # self.explorationRate -= self.explorationDecay
            self.explorationRate = self.explorationRateMin + (self.explorationRateMax - \
                self.explorationRateMin) * \
                    exp(-1. * newState.frame / self.explorationDecay)
            # print(newState.frame)    
            # print(self.explorationRate)
        
        
        # Perform one step of the optimization (on the target network)
        self.optimize_model()

        # Update the target network, copying all weights and biases in DQN
        if oldState.frame % 100 == 0:
            self.targetNet.load_state_dict(self.policyNet.state_dict())

    # def train(self, oldState, action, reward, newState):
    #     # Ask the model for the Q values of the old state (inference)
    #     oldStateQValues = self.getQValues(oldState)

    #     # Ask the model for the Q values of the new state (inference)
    #     newStateQValues = self.getQValues(newState)

    #     # Real Q value for the action we took. This is what we will train towards.
    #     old_state_Q_values[action] = reward + self.discount * np.amax(new_state_Q_values)
        
    #     # Setup training data
    #     training_input = self.to_one_hot(old_state)
    #     target_output = [old_state_Q_values]
    #     training_data = {self.model_input: training_input, self.target_output: target_output}

    #     # Train
    #     self.session.run(self.optimizer, feed_dict=training_data)

    def optimize_model(self):
        
        if self.memory.position % BATCH_SIZE != 0 or len(self.memory) < BATCH_SIZE:
            return
        # if len(self.memory) < BATCH_SIZE:
        #     return
        print("running optimization")
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),  device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        # action_batch = torch.cat(batch.action)
        action_batch = torch.stack(batch.action, dim=0)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policyNet(torch.unsqueeze(state_batch, 2)).gather(1, torch.unsqueeze(action_batch, 1))

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device, dtype=torch.double)
        next_state_values[non_final_mask] = self.targetNet(torch.unsqueeze(non_final_next_states, 2)).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.discountRate) + reward_batch

        # Compute Huber loss
        print(state_action_values.shape)
        print(expected_state_action_values.unsqueeze(1).shape)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policyNet.parameters():
            if param.grad is not None:
                # print(param.grad)
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
            state.players[0].__dict__['action_state'].value,
            state.__dict__['stage'].value,
        )
        qState = torch.unsqueeze(torch.tensor(list(qState), dtype=torch.float64, device=self.device), 0)
        qState = F.normalize(qState)
        # print(qState)
        m = qState.mean(1, keepdim=True)
        s = qState.std(1, unbiased=False, keepdim=True)
        # print(m)
        # print(s)
        qState -= m
        qState /= s
        # print(qState)
        # print("------------------")
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

        # print("----------------------------------")
        # print(predictedState.players[2].__dict__)
        # print(state.players[2].__dict__)
        # print("Falco")
        # print("x: ", state.players[0].__dict__['pos_x'], " y: ", state.players[0].__dict__['pos_y'])
        # print("self_air_vel_x: ", state.players[0].__dict__['self_air_vel_x'], " self_air_vel_y: ", state.players[0].__dict__['self_air_vel_y'])
        # print("attack_vel_x: ", state.players[0].__dict__['attack_vel_x'], " attack_vel_y: ", state.players[0].__dict__['attack_vel_y'])
        # print("Yoshi")
        # print("x: ", state.players[2].__dict__['pos_x'], " y: ", state.players[2].__dict__['pos_y'])
        # print("self_air_vel_x: ", state.players[2].__dict__['self_air_vel_x'], " self_air_vel_y: ", state.players[2].__dict__['self_air_vel_y'])
        # print("attack_vel_x: ", state.players[2].__dict__['attack_vel_x'], " attack_vel_y: ", state.players[2].__dict__['attack_vel_y'])

        # return