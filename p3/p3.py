import os.path
import os, signal, platform
import subprocess
import time

import p3.fox
import p3.memory_watcher
import p3.menu_manager
import p3.pad
import p3.state
import p3.state_manager
import p3.stats
import chars.fox
import chars.yoshi
import chars.falco
import agents.qLearningAgents
import torch
from dolphin import start

startTime = time.time()
runTime = 3600# 10800 # 21600 # 3600 # 1 hour # 5400 # 1.5 hours

def find_dolphin_dir():
    """Attempts to find the dolphin user directory. None on failure."""
    candidates = ['~/.dolphin-emu', '~/.local/share/.dolphin-emu', '~/.local/share/dolphin-emu', '~/Library/Application Support/Dolphin']
    # candidates = ['~/dolphin-test/build/Binaries/Sys', '~/dolphin-emu-nogui/build/Binaries/Sys', '~/dolphin-gurvan/build/Binaries/Sys']
    
    for candidate in candidates:
        path = os.path.expanduser(candidate)
        print(path)
        if os.path.isdir(path):
            return path
    return None

def write_locations(dolphin_dir, locations):
    """Writes out the locations list to the appropriate place under dolphin_dir."""
    path = dolphin_dir + '/MemoryWatcher/Locations.txt'
    with open(path, 'w') as f:
        f.write('\n'.join(locations))

        dolphin_dir = find_dolphin_dir()
        if dolphin_dir is None:
            print('Could not detect dolphin directory.')
            return

def run(char, state, sm, mw, pad, stats):
    mm = p3.menu_manager.MenuManager()
    while True:
        if time.time() > (startTime + runTime):
            raise KeyboardInterrupt
        last_frame = state.frame
        res = next(mw)
        if res is not None:
            sm.handle(*res)
        if state.frame > last_frame:
            stats.add_frames(state.frame - last_frame)
            start = time.time()
            make_action(state, pad, mm, char)
            stats.add_thinking_time(time.time() - start)
            stats.add_metrics(state)
            

def make_action(state, pad, mm, char):
    if state.menu == p3.state.Menu.Game:
        char.advance(state, pad)
    elif state.menu == p3.state.Menu.Characters:
        char.pick_self(state, pad)
        mm.press_start_lots(state, pad)
    elif state.menu == p3.state.Menu.Stages:
        # Handle this once we know where the cursor position is in memory.
        pad.tilt_stick(p3.pad.Stick.C, 0.5, 0.5)
        mm.press_start_lots(state, pad)
    elif state.menu == p3.state.Menu.PostGame:
        mm.press_start_lots(state, pad)

def main(charString, agentString, lr=0.1, dr=0.95, er=1.0, ed=0.0005, emin=0.01, model="nosave", learn=True, selfSelect=False, selfChar="Falco", level=9, default=True, headless=False):
    dolphin_dir = find_dolphin_dir()
    if dolphin_dir is None:
        print('Could not find dolphin config dir.')
        return

    state = p3.state.State()
    sm = p3.state_manager.StateManager(state)
    write_locations(dolphin_dir, sm.locations())

    stats = p3.stats.Stats()

    charSwitcher={
        "Yoshi": chars.yoshi.Yoshi,
        "Fox": chars.fox.Fox,
        "Falco": chars.falco.Falco
    }
    agentSwitcher={
        "Q": agents.qLearningAgents.QLearningAgent
    }
    agent = agentSwitcher.get(agentString)
    agentOptions = {
        "learningRate": lr,
        "discountRate": dr,
        "explorationRate": er,
        "explorationDecay": ed,
        "explorationRateMin": emin,
        "model": model,
        "learn": learn
    }

    dolphinPid = None
    process = None

    try:
        print('Start dolphin now. Press ^C to stop p3.')
        pad_path = dolphin_dir + '/Pipes/p3'
        mw_path = dolphin_dir + '/MemoryWatcher/MemoryWatcher'
        pad = p3.pad.Pad(pad_path)
        char = charSwitcher.get(charString)(agent, pad, agentOptions)

        time.sleep(1)
        start(default, headless)

        if selfSelect:
            pad_enemy_path = dolphin_dir + '/Pipes/p2'
            pad_enemy = p3.pad.Pad(pad_enemy_path)
            char_enemy = charSwitcher.get(selfChar)(agent, pad_enemy, agentOptions)

            print(mw_path)
            with pad_enemy as pad_e, p3.memory_watcher.MemoryWatcher(mw_path) as mw_e:
                mm = p3.menu_manager.MenuManager()
                while not char_enemy.selected:
                    last_frame = state.frame
                    res_e = next(mw_e)
                    if res_e is not None:
                        # print(res_e)
                        sm.handle(*res_e)
                    if state.frame > last_frame:
                        if not char_enemy.bot_level_set:
                            char_enemy.set_bot_level(state, pad_e, level=level)
                        else:
                            char_enemy.pick_self(state, pad_e)
                char_enemy.pick_self(state, pad_e)
        
        with pad as pad, p3.memory_watcher.MemoryWatcher(mw_path) as mw:
            run(char, state, sm, mw, pad, stats)
    except KeyboardInterrupt:
        print('Stopped')
        # stats.save_row_results(model)
        # stats.save_readable_results(model)
        print(stats)
        if(model != "nosave" and learn):
            torch.save(char.agent.policyNet.state_dict(), "models/" + model)

if __name__ == '__main__':
    main("Fox", "Q")
