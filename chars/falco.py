import p3.pad
import math
import copy
import torch
from p3.state import ActionState
import time

class Falco:
    def __init__(self, agent, pad, agentOptions):
        self.possibleActions = []
        self.learn = agentOptions['learn']

        with open("inputs/testnorm.txt", "r") as inputs:
            lines = inputs.readlines()
            for line in lines:
                self.possibleActions.append(eval(line))
        
        self.action_list = []
        self.last_action = 0
        self.lastState = None
        self.lastAction = 0
        self.selected = False

        self.active = False
        self.is_bot = False
        self.bot_level_selected = False
        self.bot_level_set = False

        self.agent = agent(self.possibleActions, agentOptions['learningRate'], agentOptions['discountRate'], agentOptions['explorationRate'], agentOptions['explorationDecay'], agentOptions['explorationRateMin'], agentOptions['model'])

    def pick_self(self, state, pad):
        if self.selected:
            self.lastState = state
            pad.release_button(p3.pad.Button.A)
        else:
            # Go to falco and press A
            target_x = -29.5
            target_y = 11.5
            moved = self.move_stick(state, pad, target_x, target_y)
            if moved:
                pad.press_button(p3.pad.Button.A)
                self.selected = True
                # the sleep is so that the inputs don't get sent at the same time in which case the reset overrides the button press
                time.sleep(.02)
                pad.reset()

    def set_bot_level(self, state, pad, level=9):
        if self.bot_level_set:
            self.lastState = state
            pad.release_button(p3.pad.Button.A)
        else:
            target_x = 0.0
            target_y = 0.0
            if not self.active:
                # move cursor above player row to activate p2
                target_x = -15.0
                target_y = 1.5
                moved = self.move_stick(state, pad, target_x, target_y)
                if moved:
                    pad.reset()
                    self.active = True

            elif not self.is_bot:
                target_x = -30.0
                target_y = -2.2
                moved = self.move_stick(state, pad, target_x, target_y)
                if moved:
                    pad.press_button(p3.pad.Button.A)
                    time.sleep(0.05)
                    pad.reset()
                    self.is_bot = True

            elif not self.bot_level_selected:
                # Go to bot level and press A
                target_x = -30.9
                target_y = -15.12
                moved = self.move_stick(state, pad, target_x, target_y)
                if moved:
                    pad.press_button(p3.pad.Button.A)
                    time.sleep(0.05)
                    # pad.release_button(p3.pad.Button.A)
                    pad.reset()
                    self.bot_level_selected = True

            elif not self.bot_level_set:
                target_x = -31.9 + float(level)*1.1
                target_y = -15.12
                moved = self.move_stick(state, pad, target_x, target_y)
                if moved:
                    pad.press_button(p3.pad.Button.A)
                    time.sleep(.02)
                    pad.reset()
                    self.bot_level_set = True

    def move_stick(self, state, pad, target_x, target_y):
        # print("-----P2 Inputs-----")
        # print("X: ", state.players[1].cursor_x, "    Y: ", state.players[1].cursor_y)
        dx = target_x - state.players[0].cursor_x
        dy = target_y - state.players[0].cursor_y
        mag = math.sqrt(dx * dx + dy * dy)

        # print(mag)
        if mag >= 1.3:
            # x = [dx, 0]
            # y = [0, dy]
            # direction_vector = [dx, dy]
            
            # x_component = dx * ((sum([x[i] * direction_vector[i] for i in range(len(x))])) / (dx*mag))
            # y_component = dy * ((sum([y[i] * direction_vector[i] for i in range(len(y))])) / (dy*mag))
            # x_contrib = x_component / mag
            # y_contrib = y_component / mag
            # print("X: ", '{:.2f}'.format(x_component), " Y: ", '{:.2f}'.format(y_component))
            # print("X_contrib: ", '{:.2f}'.format(x_contrib), " Y_contrib: ", '{:.2f}'.format(y_contrib))
            # print(mag)
            # print("-----")

            pad.tilt_stick(p3.pad.Stick.MAIN, 0.5 * (dx / mag) + 0.5, 0.5 * (dy / mag) + 0.5)
            # pad.tilt_stick(p3.pad.Stick.MAIN, 0.5 * x_contrib + 0.5, 0.5 * y_contrib + 0.5)

            # print("-----dx, dy-----")
            # print("dx: ", dx, "    dy: ", dy)
            # print("-----Tilt-----")
            # print("X: ", 0.5 * x_contrib + 0.5, "    Y: ", 0.5 * y_contrib + 0.5 + 0.5)
            # print("X: ", 0.5 * (dx / mag) + 0.5, "    Y: ", 0.5 * (dy / mag) + 0.5)
            # print("=================")

        return mag < 1.3

    def advance(self, state, pad):
        while self.action_list:
            wait, func, args = self.action_list[0]
            if state.frame - self.last_action < wait:
                return
            else:
                self.action_list.pop(0)
                if func is not None:
                    func(*args)
                self.last_action = state.frame
        else:
            if (state.frame - self.last_action) < 3:
                return
            qState = self.agent.getStateRep(state)
            
            if self.learn:
                self.agent.update(self.lastState, state, self.lastAction)
                nextAction = self.agent.getAction(qState)
            else:
                nextAction = self.agent.policy(qState)

            self.lastState = copy.deepcopy(state)
            self.lastAction = nextAction
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
