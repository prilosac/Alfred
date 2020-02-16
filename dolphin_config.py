gale01_ini = """
[Gecko_Enabled]
$Netplay Community Settings
"""

pipe_config = """
Device = Pipe/1/p2
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
    config_dir = dolphin_dir + '/Config'
    with open(config_dir + 'GCPadNew.ini', 'w') as f:
        config_1 = "[GCPad1]\n"
        config_1 += "Device = Pipe/1/p2\n"
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
        game_dir = '/home/prilo/DolphinGames/'

    with open(datapath + '/Dolphin.ini', 'r') as f:
        dolphin_ini = f.read()

    with open(configDir + '/Dolphin.ini', 'w') as f:
      config_args = dict(
        iso_path=game_dir,
      )
      f.write(dolphin_ini.format(**config_args))


    game_settings = dolphin_dir + '/GameSettings'
    with open(game_settings + '/GALE01.ini', 'w') as f:
      f.write(gale01_ini)