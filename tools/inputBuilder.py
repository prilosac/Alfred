import numpy
import enum
import itertools
from collections.abc import Iterable
import sys

args = sys.argv[1:]
outfile = args[0] or "test.txt"

@enum.unique
class Button(enum.Enum):
    A = 0
    B = 1
    X = 2
    Y = 3
    Z = 4
    START = 5
    L = 6
    R = 7
    D_UP = 8
    D_DOWN = 9
    D_LEFT = 10
    D_RIGHT = 11

@enum.unique
class Trigger(enum.Enum):
    L = 0
    R = 1

@enum.unique
class Stick(enum.Enum):
    MAIN = 0
    C = 1

def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

# (0, pad.ACTION_FUNCTION, [p3.pad.INPUT_TYPE.INPUT[, x-val, y-val]])
Sticks = ["Stick.MAIN", "Stick.C"]
Triggers = ["Trigger.L", "Trigger.R"]
# Buttons = ["Button.A", "Button.B", "Button.X", "Button.Y", "Button.Z", "Button.L", "Button.R", "Button.D_UP"]
Buttons = ["Button.A", "Button.B", "Button.X", "Button.Y", "Button.Z", "Button.L", "Button.R"]
ButtonActions = ["pad.press_button", "pad.release_button"]

wavedashRight = ('(0, pad.press_button, [p3.pad.Button.Y])', '(1, pad.release_button, [p3.pad.Button.Y])', '(6, pad.tilt_stick, [p3.pad.Stick.MAIN, 0.8, 0.2])', '(0, pad.press_button, [p3.pad.Button.L])', '(1, pad.release_button, [p3.pad.Button.L])', '(1, pad.tilt_stick, [p3.pad.Stick.MAIN, 0.5, 0.5])', '(10, None, [])')
wavedashLeft = ('(0, pad.press_button, [p3.pad.Button.Y])', '(1, pad.release_button, [p3.pad.Button.Y])', '(6, pad.tilt_stick, [p3.pad.Stick.MAIN, 0.2, 0.2])', '(0, pad.press_button, [p3.pad.Button.L])', '(1, pad.release_button, [p3.pad.Button.L])', '(1, pad.tilt_stick, [p3.pad.Stick.MAIN, 0.5, 0.5])', '(10, None, [])')
taunt = ('(0, pad.press_button, [p3.pad.Button.D_UP])', '(2, pad.release_button, [p3.pad.Button.D_UP])', '(1, None, [])')

# def createButtonInputs():
#     ans = []
#     for button in Buttons:
#         for action in ButtonActions:
#             currInput = "(0, {}, [p3.pad.{}])".format(action, button)
#             # print(currInput)
#             # file.write("[" + currInput + "]\n")
#             ans.append(currInput)

#     ans.append('(1, None, [])')
#     # print(ans)
#     return ans

def createButtonInputs():
    ans = []
    for button in Buttons:
        currInput = "(0, {}, [p3.pad.{}])".format(ButtonActions[0], button)
        # currNullInput = "(2, {}, [p3.pad.{}])".format(ButtonActions[1], button)
        currNullInput = "(2, pad.reset, [])"
        ans.append([currInput, currNullInput])
        # ans.append(currNullInput)

    ans.append('(1, None, [])')
    # print(ans)
    return ans

def createStickInputs():
    ans = []
    for stick in Sticks:
        x = 0.0
        y = 0.0
        for i in range(6):
            y = 0.0
            for j in range(6):
                currInput = "(0, pad.tilt_stick, [p3.pad.{}, {}, {}])".format(stick, x, y)
                # currNullInput = "(1, pad.tilt_stick, [p3.pad.{}, {}, {}])".format(stick, 0.5, 0.5)
                currNullInput = "(2, pad.reset, [])"

                y += 0.2
                # ans.append(currInput)
                ans.append([currInput, currNullInput])
            x += 0.2
            
    ans.append('(1, None, [])')
    return ans

# file = open("inputs/yoshi.txt", "w+")
# file = open("inputs/old.txt", "w+")
file = open("inputs/"+outfile, "w+")

file.write(str(list(wavedashLeft)) + "\n")
file.write(str(list(wavedashRight)) + "\n")
file.write(str(list(taunt)) + "\n")

for i in range(3):
    allInputs = ["(1, None, [])"]
    possibleButtonInputs = createButtonInputs()
    possibleStickInputs = createStickInputs()
    allInputs += possibleButtonInputs + possibleStickInputs
    currCombos = itertools.combinations(allInputs, i)
    # currCombos = list(itertools.product(possibleButtonInputs, possibleStickInputs))
    for combo in currCombos:
        # The below code attemps to skip inputs with triggers
        # and the main stick being pressed at the same time
        # to urge Yoshi not to perma-roll. This is so far
        # unsuccessful

        # stickFlag = False
        # triggerFlag = False
        # skip = False
        # if len(combo) > 0:
        #     for action in combo:
        #         if ("p3.pad.Stick.MAIN" in action):
        #             if triggerFlag:
        #                 skip = True
        #             stickFlag = True
        #         if("p3.pad.Button.L" in action or \
        #             "p3.pad.Button.R" in action):
        #             if stickFlag:
        #                 skip = True
        #             triggerFlag = True
        #     #     print(action)
        #     # print(combo)
        
        # if not skip:
        #     combo = flatten(combo)
        #     file.write(str(list(combo)) + "\n")

        combo = flatten(combo)
        file.write(str(list(combo)) + "\n")
