from typing import Any

from tasks import TASKS


def grade(workspace_path: str | None = None, **kwargs: Any) -> dict[str, Any]:
    if not workspace_path:
        return {"score": 0.0, "metrics": {}, "feedback": "No workspace_path provided."}
    result = TASKS[0]["grader"](workspace_path, kwargs.get("logs"), kwargs.get("last_exec_time"))
    return {
        "score": max(0.0, min(1.0, float(result.get("score", 0.0)))),
        "metrics": result.get("metrics", {}),
        "feedback": result.get("feedback", ""),
    }
