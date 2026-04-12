from typing import Any

from tasks import TASKS


def _clamp_score(score: Any) -> float:
    try:
        value = float(score)
    except (TypeError, ValueError):
        value = 0.01
    return max(0.01, min(0.99, value))


class _BaseTaskGrader:
    task_index: int = 0

    def _workspace_path(self, env: Any) -> str | None:
        if isinstance(env, str):
            return env
        if env is None:
            return None
        return getattr(env, "workspace", None)

    def grade(self, env: Any, *args: Any, **kwargs: Any) -> float:
        workspace_path = self._workspace_path(env)
        if not workspace_path:
            return 0.01
        result = TASKS[self.task_index]["grader"](
            workspace_path,
            kwargs.get("logs"),
            kwargs.get("last_exec_time"),
        )
        return _clamp_score(result.get("score", 0.01))

    def __call__(self, env: Any = None, *args: Any, **kwargs: Any) -> dict[str, Any]:
        workspace_path = self._workspace_path(env)
        if not workspace_path:
            return {"score": 0.01, "metrics": {}, "feedback": "No workspace_path provided."}
        result = TASKS[self.task_index]["grader"](
            workspace_path,
            kwargs.get("logs"),
            kwargs.get("last_exec_time"),
        )
        return {
            "score": _clamp_score(result.get("score", 0.01)),
            "metrics": result.get("metrics", {}),
            "feedback": result.get("feedback", ""),
        }


class Task0Grader(_BaseTaskGrader):
    task_index = 0


class Task1Grader(_BaseTaskGrader):
    task_index = 1


class Task2Grader(_BaseTaskGrader):
    task_index = 2


class Task3Grader(_BaseTaskGrader):
    task_index = 3


class Task4Grader(_BaseTaskGrader):
    task_index = 4


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
