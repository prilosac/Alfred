import p3.pad
import math
import copy
import torch
import numpy as np
from p3.state import ActionState

class Yoshi:
    def __init__(self, agent, pad, agentOptions):
        self.possibleActions = []
        self.learn = agentOptions['learn']

        with open("inputs/test3.txt", "r") as inputs:
            lines = inputs.readlines()
            for line in lines:
                self.possibleActions.append(eval(line))
        
        # self.possibleActions = [
        #     [(0, pad.tilt_stick, [p3.pad.Stick.MAIN, 0.5, 0.75]), (1, pad.press_button, [p3.pad.Button.A]), (1, pad.release_button, [p3.pad.Button.A]), (1, None, [])],
        #     [(0, pad.tilt_stick, [p3.pad.Stick.MAIN, 0.0, 0.5]), (5, pad.tilt_stick, [p3.pad.Stick.MAIN, 0.5, 0.5]), (1, None, [])],
        #     [(0, pad.tilt_stick, [p3.pad.Stick.MAIN, 1.0, 0.5]), (5, pad.tilt_stick, [p3.pad.Stick.MAIN, 0.5, 0.5]), (1, None, [])],
        #     [(0, pad.tilt_stick, [p3.pad.Stick.MAIN, 0.3, 1.0]), (0, pad.press_button, [p3.pad.Button.B]), (1, pad.release_button, [p3.pad.Button.B]), (0, pad.tilt_stick, [p3.pad.Stick.MAIN, 0.5, 0.5]), (1, None, [])],
        #     [(0, pad.tilt_stick, [p3.pad.Stick.MAIN, 0.5, 0.0]), (0, pad.press_button, [p3.pad.Button.A]), (1, pad.release_button, [p3.pad.Button.A]), (0, pad.tilt_stick, [p3.pad.Stick.MAIN, 0.5, 0.5]), (1, None, [])]
        # ]
        
        self.action_list = []
        self.last_action = 0
        self.lastState = None
        # self.lastState = torch.unsqueeze(torch.unsqueeze(torch.tensor(()), 1), 0)
        # self.lastState = torch.tensor(0)
        # self.lastAction = ['(1, None, [])']
        self.lastAction = 0
        self.selected = False

        self.agent = agent(self.possibleActions, agentOptions['learningRate'], agentOptions['discountRate'], agentOptions['explorationRate'], agentOptions['explorationDecay'], agentOptions['explorationRateMin'], agentOptions['model'])

    def pick_self(self, state, pad):
        if self.selected:
            self.lastState = state
            # Release buttons and lazilly rotate the c stick.
            pad.release_button(p3.pad.Button.A)
            # pad.tilt_stick(p3.pad.Stick.MAIN, 0.5, 0.5)
            # angle = (state.frame % 240) / 240.0 * 2 * math.pi
            # pad.tilt_stick(p3.pad.Stick.C, 0.4 * math.cos(angle) + 0.5, 0.4 * math.sin(angle) + 0.5)

        else:
            # Go to yoshi and press A
            target_x = 4.5
            target_y = 18.5
            # >>> choose fox >>>
            # target_x = -23.5
            # target_y = 11.5
            # <<< choose fox <<<

            # print("x: ", state.players[2].cursor_x, "y: ", state.players[2].cursor_y)
            dx = target_x - state.players[2].cursor_x
            dy = target_y - state.players[2].cursor_y
            mag = math.sqrt(dx * dx + dy * dy)
            # if state.frame % 200 == 0:
                # self.printState(state)
            if mag < 1.3:
                pad.press_button(p3.pad.Button.A)
                self.selected = True
                pad.tilt_stick(p3.pad.Stick.MAIN, 0.5, 0.5)
            else:
                pad.tilt_stick(p3.pad.Stick.MAIN, 0.5 * (dx / mag) + 0.5, 0.5 * (dy / mag) + 0.5)

    def advance(self, state, pad):
        # if(state.frame % 200 == 0):
                # self.printState(state)
        # self.printState(state)
        if state.frame > self.last_action:
            self.agent.rewardMemory.push(copy.deepcopy(state))
        while self.action_list:
            wait, func, args = self.action_list[0]
            if state.frame - self.last_action < wait:
                return
            # elif (state.frame - self.last_action) < 3:
            #     return
            else:
                # self.agent.update(self.lastState, state, self.lastAction)
                self.action_list.pop(0)
                if func is not None:
                    func(*args)
                self.last_action = state.frame
        else:
            if (state.frame - self.last_action) < 3:
                return
            # Eventually this will point at some decision-making thing.
            # self.utilt(pad)
            # if(state.players[2].__dict__['action_state'] == ActionState.Wait):
            #     self.wavedash(pad)
            # print(self.agent.getAction(state))
            qState = self.agent.getStateRep(state)

            if self.learn:
                self.agent.update(self.lastState, state, self.lastAction)
                nextAction = self.agent.getAction(qState)
            else:
                nextAction = self.agent.policy(qState)
            # nextAction = self.agent.getAction(qState)
            # print(nextAction)
            self.lastState = copy.deepcopy(state)
            self.lastAction = nextAction
            # if isinstance(nextAction, int):
            #     self.addAction(self.possibleActions[nextAction], pad)
            # else:
            #     print(nextAction.detach())
            #     self.addAction(np.random.choice(self.possibleActions, p=nextAction.detach()), pad)
            self.addAction(self.possibleActions[nextAction], pad)

    def addAction(self, action, pad):
        for gameInput in action:
            if type(gameInput) is str:
                self.action_list.append(eval(gameInput))
            else:
                self.action_list.append(gameInput)

    def utilt(self, pad):
        self.action_list.append((0, pad.tilt_stick, [p3.pad.Stick.MAIN, 0.5, 0.8]))
        self.action_list.append((0, pad.press_button, [p3.pad.Button.A]))
        self.action_list.append((1, pad.release_button, [p3.pad.Button.A]))
        self.action_list.append((1, None, []))

    def wavedash(self, pad):
        self.action_list.append((0, pad.press_button, [p3.pad.Button.Y]))
        self.action_list.append((1, pad.release_button, [p3.pad.Button.Y]))
        self.action_list.append((6, pad.tilt_stick, [p3.pad.Stick.MAIN, 0.8, 0.2]))
        self.action_list.append((0, pad.press_button, [p3.pad.Button.L]))
        self.action_list.append((1, pad.release_button, [p3.pad.Button.L]))
        self.action_list.append((1, pad.tilt_stick, [p3.pad.Stick.MAIN, 0.5, 0.5]))
        self.action_list.append((1, None, []))

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
