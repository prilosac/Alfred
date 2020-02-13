# import p3 as p3
# import p3.fox as fox
# import chars.yoshi as yoshi
import p3.p3 as p3
# from agents import qLearningAgents as qla
import sys

learn = True
args = sys.argv[1:]
if args.__contains__('-p') or args.__contains__('--play'):
    learn = False

# CharString, Agent, Learning Rate, Discount Rate,
# Exploration Rate, Exploration Discount, Model
# Min Exploration Rate
p3.main("Yoshi", "Q", lr=0.1, dr=0.95, er=1.0, ed=20000, emin=0.01, model="2-11-2020-Adam", learn=learn, selfSelect=True, level=9)