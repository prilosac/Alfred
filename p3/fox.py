import p3.pad
import math

class Fox:
    def __init__(self):
        self.action_list = []
        self.last_action = 0
        self.selected = False

    def pick_fox(self, state, pad):
        if self.selected:
            # Release buttons and lazilly rotate the c stick.
            pad.release_button(p3.pad.Button.A)
            pad.tilt_stick(p3.pad.Stick.MAIN, 0.5, 0.5)
            angle = (state.frame % 240) / 240.0 * 2 * math.pi
            pad.tilt_stick(p3.pad.Stick.C, 0.4 * math.cos(angle) + 0.5, 0.4 * math.sin(angle) + 0.5)
        else:
            # Go to fox and press A
            target_x = -23.5
            target_y = 11.5
            dx = target_x - state.players[2].cursor_x
            dy = target_y - state.players[2].cursor_y
            mag = math.sqrt(dx * dx + dy * dy)
            if mag < 0.3:
                pad.press_button(p3.pad.Button.A)
                self.selected = True
            else:
                pad.tilt_stick(p3.pad.Stick.MAIN, 0.5 * (dx / mag) + 0.5, 0.5 * (dy / mag) + 0.5)

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
            # Eventually this will point at some decision-making thing.
            self.shinespam(pad)

    def shinespam(self, pad):
        self.action_list.append((0, pad.tilt_stick, [p3.pad.Stick.MAIN, 0.5, 0.0]))
        self.action_list.append((0, pad.press_button, [p3.pad.Button.B]))
        self.action_list.append((1, pad.release_button, [p3.pad.Button.B]))
        self.action_list.append((0, pad.tilt_stick, [p3.pad.Stick.MAIN, 0.5, 0.5]))
        self.action_list.append((0, pad.press_button, [p3.pad.Button.X]))
        self.action_list.append((1, pad.release_button, [p3.pad.Button.X]))
        self.action_list.append((1, None, []))
