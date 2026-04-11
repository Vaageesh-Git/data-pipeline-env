from typing import Any

from tasks import TASKS


def _normalize(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": max(0.0, min(1.0, float(result.get("score", 0.0)))),
        "metrics": result.get("metrics", {}),
        "feedback": result.get("feedback", ""),
    }


class Task0Grader:
    def __call__(self, workspace_path: str | None = None, **kwargs: Any) -> dict[str, Any]:
        if not workspace_path:
            return {"score": 0.0, "metrics": {}, "feedback": "No workspace_path provided."}
        return _normalize(TASKS[0]["grader"](workspace_path, kwargs.get("logs"), kwargs.get("last_exec_time")))


class Task1Grader:
    def __call__(self, workspace_path: str | None = None, **kwargs: Any) -> dict[str, Any]:
        if not workspace_path:
            return {"score": 0.0, "metrics": {}, "feedback": "No workspace_path provided."}
        return _normalize(TASKS[1]["grader"](workspace_path, kwargs.get("logs"), kwargs.get("last_exec_time")))


class Task2Grader:
    def __call__(self, workspace_path: str | None = None, **kwargs: Any) -> dict[str, Any]:
        if not workspace_path:
            return {"score": 0.0, "metrics": {}, "feedback": "No workspace_path provided."}
        return _normalize(TASKS[2]["grader"](workspace_path, kwargs.get("logs"), kwargs.get("last_exec_time")))


class Task3Grader:
    def __call__(self, workspace_path: str | None = None, **kwargs: Any) -> dict[str, Any]:
        if not workspace_path:
            return {"score": 0.0, "metrics": {}, "feedback": "No workspace_path provided."}
        return _normalize(TASKS[3]["grader"](workspace_path, kwargs.get("logs"), kwargs.get("last_exec_time")))


class Task4Grader:
    def __call__(self, workspace_path: str | None = None, **kwargs: Any) -> dict[str, Any]:
        if not workspace_path:
            return {"score": 0.0, "metrics": {}, "feedback": "No workspace_path provided."}
        return _normalize(TASKS[4]["grader"](workspace_path, kwargs.get("logs"), kwargs.get("last_exec_time")))


class EasyGrader(Task0Grader):
    pass


class MediumGrader(Task1Grader):
    pass


class HardGrader(Task2Grader):
    pass


GRADERS = {
    "task_easy": EasyGrader(),
    "task_medium": MediumGrader(),
    "task_hard": HardGrader(),
    "task_0": Task0Grader(),
    "task_1": Task1Grader(),
    "task_2": Task2Grader(),
    "task_3": Task3Grader(),
    "task_4": Task4Grader(),
}
