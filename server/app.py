# app.py
from fastapi import FastAPI, HTTPException
from models import Action, Observation, State
from env import DataPipelineEnv
import uvicorn

app = FastAPI(title="DataPipe-Sandbox OpenEnv")

# Global environment instance
# In a production environment with multiple users, you would use 
# a dictionary mapping session_ids to Env instances.
env = DataPipelineEnv()

@app.get("/")
async def root():
    return {"message": "DataPipe-Sandbox is running. Use /reset to start."}

@app.post("/reset", response_model=Observation)
async def reset(task_id: int = 0):
    """
    Resets the environment to a specific task.
    OpenEnv Spec: Must return the initial Observation.
    """
    try:
        observation = env.reset(task_id=task_id)
        return observation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
async def step(action: Action):
    """
    Executes one action in the environment.
    OpenEnv Spec: Returns (observation, reward, done, info).
    """
    try:
        observation, reward, done, info = env.step(action)
        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state", response_model=State)
async def get_state():
    """
    Returns the current internal state of the environment.
    """
    try:
        return env.state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()