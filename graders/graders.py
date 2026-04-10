from typing import Any

from tasks import TASKS


def _task_index(task_id: int | str) -> int:
    if isinstance(task_id, str) and task_id.startswith("task_"):
        task_id = task_id.removeprefix("task_")
    return int(task_id)


def _normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    score = float(result.get("score", 0.0))
    metrics = result.get("metrics", {})
    return {
        "score": max(0.0, min(1.0, score)),
        "metrics": metrics,
        "feedback": result.get("feedback", ""),
    }


def grade_task(task_id: int, workspace_path: str | None = None, **kwargs: Any) -> dict[str, Any]:
    task_id = _task_index(task_id)
    if task_id < 0 or task_id >= len(TASKS):
        return {"score": 0.0, "metrics": {}, "feedback": f"Unknown task_id: {task_id}"}

    if not workspace_path:
        return {"score": 0.0, "metrics": {}, "feedback": "No workspace_path provided."}

    result = TASKS[task_id]["grader"](
        workspace_path,
        kwargs.get("logs"),
        kwargs.get("last_exec_time"),
    )
    return _normalize_result(result)


def grade_task_0(workspace_path: str | None = None, **kwargs: Any) -> dict[str, Any]:
    return grade_task(0, workspace_path, **kwargs)


def grade_task_1(workspace_path: str | None = None, **kwargs: Any) -> dict[str, Any]:
    return grade_task(1, workspace_path, **kwargs)


def grade_task_2(workspace_path: str | None = None, **kwargs: Any) -> dict[str, Any]:
    return grade_task(2, workspace_path, **kwargs)


def grade_task_3(workspace_path: str | None = None, **kwargs: Any) -> dict[str, Any]:
    return grade_task(3, workspace_path, **kwargs)


def grade_task_4(workspace_path: str | None = None, **kwargs: Any) -> dict[str, Any]:
    return grade_task(4, workspace_path, **kwargs)


GRADERS = {
    "task_0": grade_task_0,
    "task_1": grade_task_1,
    "task_2": grade_task_2,
    "task_3": grade_task_3,
    "task_4": grade_task_4,
}
