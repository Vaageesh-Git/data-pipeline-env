from env import DataPipelineEnv
from models import Action
import sys

try:
    e = DataPipelineEnv()
    obs = e.reset(0)
    print("Reset OK")
    obs, r, d, i = e.step(Action(command="write", path="solution.py", content=""))
    print("Write OK", r)
    obs, r, d, i = e.step(Action(command="run"))
    print("Run OK", r)
    obs, r, d, i = e.step(Action(command="submit"))
    print("Submit OK", r)
except Exception as ex:
    print("ERROR:", type(ex), ex)

