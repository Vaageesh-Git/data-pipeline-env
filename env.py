import os
import shutil
import tempfile
from typing import Optional, Tuple

import pandas as pd

from models import Action, Observation, State
from tasks import TASKS, grade_submission, preview_run, run_pipeline


class DataPipelineEnv:
    def __init__(self):
        self.workspace: Optional[str] = None
        self.current_task_id = 0
        self.step_count = 0
        self.logs = "Environment initialized. Call reset() to start a task."
        self.done = False
        self.last_metrics = {"correctness": 0.0, "efficiency": 0.0, "robustness": 0.0}
        self.last_exec_time = 0.0

    def reset(self, task_id: int = 0) -> Observation:
        if task_id < 0 or task_id >= len(TASKS):
            raise ValueError(f"Unknown task_id: {task_id}")

        self.current_task_id = task_id
        self.step_count = 0
        self.done = False
        self.last_exec_time = 0.0
        self.last_metrics = {"correctness": 0.0, "efficiency": 0.0, "robustness": 0.0}

        if self.workspace and os.path.exists(self.workspace):
            shutil.rmtree(self.workspace)
        self.workspace = tempfile.mkdtemp(prefix=f"task_{task_id}_")

        task = TASKS[task_id]
        for filename, content in task["files"].items():
            destination = os.path.join(self.workspace, filename)
            os.makedirs(os.path.dirname(destination) or self.workspace, exist_ok=True)
            with open(destination, "w", encoding="utf-8") as handle:
                handle.write(content)

        self.logs = (
            f"Task Started: {task['name']}\n"
            f"{task['description']}\n"
            f"Entrypoint: {task['entrypoint']}\n"
            f"Deliverable: {task['output_file']}\n"
            "Read task.md for the full instructions."
        )
        return self._get_obs()

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        self.step_count += 1
        reward = 0.0
        info = {}

        if action.command == "read":
            info = self._handle_read(action.path or "")
        elif action.command == "write":
            info = self._handle_write(action.path or "", action.content or "")
        elif action.command == "run":
            info = self._handle_run()
            reward = info["score"]
        elif action.command == "submit":
            info = self._handle_submit()
            reward = info["score"]
            self.done = True
        else:
            self.logs = f"Unknown command: {action.command}"
            info = {"feedback": self.logs}

        observation = self._get_obs()
        if action.command != "submit":
            reward -= 0.01 * self.step_count
        return observation, reward, self.done, info

    def _workspace_path(self, path: str) -> str:
        if not self.workspace:
            raise ValueError("No active workspace. Call reset() first.")
        if not path:
            raise ValueError("A file path is required for this action.")

        normalized = os.path.normpath(os.path.join(self.workspace, path))
        workspace_root = os.path.abspath(self.workspace)
        candidate = os.path.abspath(normalized)
        if not candidate.startswith(workspace_root):
            raise ValueError("Path escapes the task workspace.")
        return candidate

    def _handle_read(self, path: str):
        try:
            full_path = self._workspace_path(path)
            with open(full_path, "r", encoding="utf-8") as handle:
                self.logs = handle.read()
            return {"feedback": f"Read {path}."}
        except Exception as error:
            self.logs = f"Error reading file {path}: {error}"
            return {"feedback": self.logs}

    def _handle_write(self, path: str, content: str):
        try:
            full_path = self._workspace_path(path)
            os.makedirs(os.path.dirname(full_path) or self.workspace, exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as handle:
                handle.write(content or "")
            self.logs = f"Successfully wrote to {path}."
            return {"feedback": self.logs}
        except Exception as error:
            self.logs = f"Error writing to {path}: {error}"
            return {"feedback": self.logs}

    def _handle_run(self):
        task = TASKS[self.current_task_id]
        execution_result = run_pipeline(self.workspace, task)
        self.last_exec_time = execution_result["duration"]

        execution_log = (
            f"STDOUT:\n{execution_result['stdout']}\n"
            f"STDERR:\n{execution_result['stderr']}\n"
            f"Execution Time: {execution_result['duration']:.4f}s"
        )

        evaluation = preview_run(task, self.workspace, execution_result)
        self.last_metrics = evaluation["metrics"]
        self.logs = f"{execution_log}\n\nFeedback:\n{evaluation['feedback']}"
        return evaluation

    def _handle_submit(self):
        task = TASKS[self.current_task_id]
        result = grade_submission(task, self.workspace, self.logs, self.last_exec_time)
        self.last_metrics = result["metrics"]
        bonus = max(0.0, 0.2 - (0.01 * self.step_count))
        result["bonus"] = round(bonus, 4)
        result["score"] = round(result["score"] + bonus, 4)
        self.logs = result["feedback"]
        return result

    def _get_obs(self) -> Observation:
        files = os.listdir(self.workspace) if self.workspace else []
        preview = "No output data found yet."
        output_path = os.path.join(self.workspace, "output.csv") if self.workspace else ""
        if output_path and os.path.exists(output_path):
            try:
                preview = pd.read_csv(output_path).head(5).to_string(index=False)
            except Exception:
                preview = "Output file exists but could not be parsed as CSV."

        return Observation(
            files=files,
            logs=self.logs,
            metrics=self.last_metrics,
            data_preview=preview,
            done=self.done,
        )

    def state(self) -> State:
        return State(
            task_id=self.current_task_id,
            step_count=self.step_count,
            workspace_path=self.workspace or "",
            session_id="local_session",
        )
