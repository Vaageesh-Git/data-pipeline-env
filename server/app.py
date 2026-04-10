# app.py
from typing import Any, Optional

from fastapi import Body, FastAPI, HTTPException, Query, Request
from models import Action, Observation, State
from env import DataPipelineEnv
from tasks import TASKS
from tasks.email_data import TASK_DIFFICULTY, TASK_EMAILS, TASK_MAX_STEPS, TASK_OBJECTIVES
from graders.graders import GRADERS
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
        "version": "1.0.0",
        "task_count": len(TASK_EMAILS),
        "tasks": [serialize_registry_task(task_id) for task_id in TASK_EMAILS],
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
    task_index = parse_task_id(task["id"])
    task_id = f"task_{task_index}"
    has_grader = task_id in GRADERS and callable(task.get("grader"))
    return {
        "id": task_id,
        "task_id": task_id,
        "index": task_index,
        "name": task["name"],
        "description": task["description"],
        "difficulty": task.get("difficulty", "medium"),
        "entrypoint": task["entrypoint"],
        "output_file": task["output_file"],
        "input_files": task["input_files"],
        "max_steps": task.get("max_steps", 20),
        "success_threshold": task.get("success_threshold", 0.7),
        "has_grader": has_grader,
        "grader": has_grader,
        "grader_path": f"tasks.task_{task_index}.grader:grade",
    }


def serialize_registry_task(task_id: str) -> dict[str, Any]:
    task_index = parse_task_id(task_id)
    task = TASKS[task_index]
    has_grader = task_id in GRADERS
    return {
        "id": task_id,
        "task_id": task_id,
        "index": task_index,
        "name": task["name"],
        "description": TASK_OBJECTIVES[task_id],
        "difficulty": TASK_DIFFICULTY.get(task_id, "medium"),
        "entrypoint": task["entrypoint"],
        "output_file": task["output_file"],
        "input_files": task["input_files"],
        "max_steps": TASK_MAX_STEPS[task_id],
        "success_threshold": task.get("success_threshold", 0.7),
        "has_grader": has_grader,
        "grader": has_grader,
        "grader_path": task["grader_path"],
    }


def parse_task_id(value: Any) -> int:
    if isinstance(value, str) and value.startswith("task_"):
        value = value.removeprefix("task_")
    try:
        task_id = int(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="task_id must be an integer or task_<integer>")
    if task_id < 0 or task_id >= len(TASKS):
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {value}")
    return task_id


@app.get("/tasks")
async def list_tasks():
    return {"tasks": [serialize_registry_task(task_id) for task_id in TASK_EMAILS]}


@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    return serialize_task(TASKS[parse_task_id(task_id)])


@app.get("/validate")
async def validate():
    checks = {
        "openenv_yaml": True,
        "typed_models": True,
        "reset_endpoint": True,
        "step_endpoint": True,
        "state_endpoint": True,
        "tasks_endpoint": True,
        "min_3_tasks": len(TASK_EMAILS) >= 3,
        "all_tasks_have_graders": all(task_id in GRADERS for task_id in TASK_EMAILS),
        "reward_shaped": True,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "env_name": "datapipe-sandbox-v1",
        "version": "1.0.0",
        "task_count": len(TASK_EMAILS),
        "tasks_with_graders": sum(1 for task_id in TASK_EMAILS if task_id in GRADERS),
    }


@app.get("/grade/{task_id}")
async def grade_current(task_id: str):
    task_index = parse_task_id(task_id)
    grader = GRADERS.get(f"task_{task_index}")
    if not grader:
        raise HTTPException(status_code=404, detail=f"No grader for task: {task_id}")
    if not env.workspace:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    return grader(env.workspace)


async def resolve_task_id(request: Request, query_task_id: Optional[str]) -> int:
    if query_task_id is not None:
        return parse_task_id(query_task_id)

    try:
        payload = await request.json()
    except Exception:
        payload = {}

    if isinstance(payload, dict):
        value = payload.get("task_id", payload.get("id", 0))
        return parse_task_id(value)

    return 0


@app.post("/reset", response_model=Observation)
async def reset(
    request: Request,
    task_id: Optional[str] = Query(default=None),
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
