# app.py
from typing import Any, Optional

from fastapi import Body, FastAPI, HTTPException, Query, Request
from models import Action, Observation, State
from env import DataPipelineEnv
from tasks import TASKS
import uvicorn

app = FastAPI(title="DataPipe-Sandbox OpenEnv", version="1.0.0")

# Global environment instance
# In a production environment with multiple users, you would use 
# a dictionary mapping session_ids to Env instances.
env = DataPipelineEnv()

@app.get("/")
async def root():
    return {"message": "DataPipe-Sandbox is running. Use /reset to start."}

@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/metadata")
async def metadata():
    return {
        "name": "datapipe-sandbox-v1",
        "description": "Data pipeline execution and grading environment.",
        "task_count": len(TASKS),
        "tasks": [serialize_task(task) for task in TASKS],
    }


@app.get("/schema")
async def schema():
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": State.model_json_schema(),
    }


@app.post("/mcp")
async def mcp(payload: dict[str, Any] = Body(default_factory=dict)):
    request_id = payload.get("id") if isinstance(payload, dict) else None
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "tools": [],
        },
    }


def serialize_task(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": task["id"],
        "name": task["name"],
        "description": task["description"],
        "entrypoint": task["entrypoint"],
        "output_file": task["output_file"],
        "input_files": task["input_files"],
        "has_grader": callable(task.get("grader")),
        "grader": "grade_submission" if callable(task.get("grader")) else None,
    }


@app.get("/tasks")
async def list_tasks():
    return [serialize_task(task) for task in TASKS]


@app.get("/tasks/{task_id}")
async def get_task(task_id: int):
    if task_id < 0 or task_id >= len(TASKS):
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")
    return serialize_task(TASKS[task_id])


async def resolve_task_id(request: Request, query_task_id: Optional[int]) -> int:
    if query_task_id is not None:
        return query_task_id

    try:
        payload = await request.json()
    except Exception:
        payload = {}

    if isinstance(payload, dict):
        value = payload.get("task_id", payload.get("id", 0))
        try:
            return int(value)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="task_id must be an integer")

    return 0


@app.post("/reset", response_model=Observation)
async def reset(
    request: Request,
    task_id: Optional[int] = Query(default=None),
):
    """
    Resets the environment to a specific task.
    OpenEnv Spec: Must return the initial Observation.
    """
    try:
        selected_task_id = await resolve_task_id(request, task_id)
        observation = env.reset(task_id=selected_task_id)
        return observation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
async def step(payload: dict[str, Any] = Body(...)):
    """
    Executes one action in the environment.
    OpenEnv Spec: Returns (observation, reward, done, info).
    """
    try:
        action_payload = payload.get("action", payload)
        action = Action(**action_payload)
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
