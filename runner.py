# import p3 as p3
# import p3.fox as fox
# import chars.yoshi as yoshi
import p3.p3 as p3
# from agents import qLearningAgents as qla
import sys, platform
from dolphin import set_config, start

learn = True
default = True
headless = False
args = sys.argv[1:]

if args.__contains__('-p') or args.__contains__('--play'):
    learn = False
if args.__contains__('-c') or args.__contains__('--custom'):
    default = False
if args.__contains__('-h') or args.__contains__('--headless'):
    headless = True

if platform.system() == "Linux":
    dolphin_dir = p3.find_dolphin_dir()
    set_config(dolphin_dir)
    # start(default, headless)

# CharString, Agent, Learning Rate, Discount Rate,
# Exploration Rate, Exploration Discount, Model
# Min Exploration Rate
p3.main("Yoshi", "Q", lr=0.1, dr=0.95, er=1.0, ed=20000, emin=0.01, model="3-19-2020-0", learn=learn, selfSelect=True, level=9, default=default, headless=headless)
# p3.main("Yoshi", "Q", lr=0.1, dr=0.95, er=1.0, ed=20000, emin=0.01, model="nosave", learn=learn, selfSelect=True, level=9, default=default, headless=headless)