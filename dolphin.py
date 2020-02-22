import platform
import os.path as path
import os, subprocess

gale01_ini = """
[Gecko_Enabled]
$Netplay Community Settings
[Controls]
PadType0 = 6
PadType1 = 7
PadType2 = 6
PadType3 = 0
"""

pipe_config = """
Buttons/A = `Button A`
Buttons/B = `Button B`
Buttons/X = `Button X`
Buttons/Y = `Button Y`
Buttons/Z = `Button Z`
Buttons/Start = `Button START`
Main Stick/Up = `Axis MAIN Y +`
Main Stick/Down = `Axis MAIN Y -`
Main Stick/Left = `Axis MAIN X -`
Main Stick/Right = `Axis MAIN X +`
Main Stick/Modifier = Shift_L
Main Stick/Modifier/Range = 50.000000000000000
C-Stick/Up = `Axis C Y +`
C-Stick/Down = `Axis C Y -`
C-Stick/Left = `Axis C X -`
C-Stick/Right = `Axis C X +`
C-Stick/Modifier = Control_L
C-Stick/Modifier/Range = 50.000000000000000
Triggers/L = `Button L`
Triggers/R = `Button R`
D-Pad/Up = `Button D_UP`
D-Pad/Down = `Button D_DOWN`
D-Pad/Left = `Button D_LEFT`
D-Pad/Right = `Button D_RIGHT`
Main Stick/Dead Zone = 5.0000000000000000
C-Stick/Dead Zone = 9.0000000000000000
Triggers/L-Analog = `Axis L +`
Triggers/R-Analog = `Axis R +`
"""

def set_config(dolphin_dir):
    config_dir = path.expanduser('~/.config/dolphin-emu')
    # config_dir = dolphin_dir + '/Config'
    if not path.exists(config_dir):
        os.makedirs(config_dir)
    mw_dir = dolphin_dir + '/MemoryWatcher'
    if not path.exists(mw_dir):
        os.makedirs(mw_dir)
    pipes_dir = dolphin_dir + '/Pipes'
    if not path.exists(pipes_dir):
        os.makedirs(pipes_dir)
    with open(config_dir + '/GCPadNew.ini', 'w') as f:
        config_1 = "[GCPad1]\n"
        config_1 += "Device = Pipe/0/p2"
        config_1 += pipe_config
        config_3 = "[GCPad3]\n"
        config_3 += "Device = Pipe/0/p3"
        config_3 += pipe_config
        config = config_1 + config_3
        f.write(config)

    game_dir = ""
    if platform.system() == "Darwin":
        # game_dir = '/Users/lucasteixeira/Dolphin Games/Super Smash Bros. Melee (v1.02).iso'
        game_dir = '/Users/lucasteixeira/Dolphin Games/'
    elif platform.system() == "Linux":
        # game_dir = '/home/prilo/DolphinGames/Super Smash Bros. Melee (v1.02).iso'
        game_dir = path.expanduser('~/DolphinGames/')

    if not path.exists(config_dir + '/Dolphin.ini'):
        dolphin_ini_init = open(config_dir + '/Dolphin.ini', 'w')
        dolphin_ini_init.close

    with open('./config/Dolphin.ini', 'r') as f:
        dolphin_ini = f.read()

    with open(config_dir + '/Dolphin.ini', 'w') as f:
      config_args = dict(
        iso_path = game_dir,
        gfx_backend = "Null"
      )
      f.write(dolphin_ini.format(**config_args))


    game_settings = dolphin_dir + '/GameSettings'
    with open(game_settings + '/GALE01.ini', 'w') as f:
      f.write(gale01_ini)

def start(default, headless):
    args = []
    if platform.system() == "Darwin":
        print("Note: Custom and Headless flags do not apply on Mac OS")
        args = ['/usr/bin/open', '-n', '-a' '/Applications/Dolphin.app', '-e', '/Users/lucasteixeira/Dolphin Games/Super Smash Bros. Melee (v1.02).iso']
    elif platform.system() == "Linux":
        if default:
            args = ['dolphin-emu', '-e', path.expanduser('~/DolphinGames/Super Smash Bros. Melee (v1.02).iso')]
        elif headless:
            args = [path.expanduser('~/dolphin-emu-nogui/build/Binaries/dolphin-emu-nogui'), '-e', path.expanduser('~/DolphinGames/Super Smash Bros. Melee (v1.02).iso')]
            # args = [path.expanduser('~/dolphin-emu-nogui/build/Binaries/dolphin-emu-nogui'), '-e', path.expanduser('~/DolphinGames/Super Smash Bros. Melee (v1.02).iso'), '-u', path.expanduser('~/dolphin-emu-nogui/build/Binaries/Sys')]
        elif not headless:
            args = [path.expanduser('~/dolphin-gurvan/build/Binaries/dolphin-emu'), '-e', path.expanduser('~/DolphinGames/Super Smash Bros. Melee (v1.02).iso')]
            # args = [path.expanduser('~/dolphin-test/build/Binaries/dolphin-emu'), '-u', path.expanduser('~/dolphin-test/build/Binaries/Sys'), '-e', path.expanduser('~/DolphinGames/Super Smash Bros. Melee (v1.02).iso')]

    else:
        sys.exit("Platform not recognized:")
    # process = subprocess.run(['/usr/bin/open', '-n', '-a' '/Applications/Dolphin.app', '-e', '/Users/lucasteixeira/Dolphin Games/Super Smash Bros. Melee (v1.02).iso'], check=True)
    return subprocess.Popen(args)