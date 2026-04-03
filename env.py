# env.py
import os
import shutil
import subprocess
import time
import tempfile
import pandas as pd
from typing import Tuple
from models import Action, Observation, State
from tasks import TASKS

class DataPipelineEnv:
    def __init__(self):
        self.workspace = None
        self.current_task_id = 0
        self.step_count = 0
        self.start_time = None
        self.logs = "Environment initialized. Call reset() to start a task."
        self.done = False
        # Placeholders for the metrics we will define later
        self.last_metrics = {"correctness": 0.0, "efficiency": 0.0, "robustness": 0.0}

    def reset(self, task_id: int = 0) -> Observation:
        """Initializes a new task session and sets up the file system."""
        self.current_task_id = task_id
        self.step_count = 0
        self.done = False
        self.last_metrics = {"correctness": 0.0, "efficiency": 0.0, "robustness": 0.0}
        
        # 1. Create a clean temporary workspace
        if self.workspace and os.path.exists(self.workspace):
            shutil.rmtree(self.workspace)
        self.workspace = tempfile.mkdtemp(prefix=f"task_{task_id}_")
        
        # 2. Seed the workspace with task files (CSV data, starter code, etc.)
        task = TASKS[task_id]
        for filename, content in task["files"].items():
            with open(os.path.join(self.workspace, filename), "w") as f:
                f.write(content)
        
        self.logs = f"Task Started: {task['name']}\n{task['description']}"
        return self._get_obs()

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """Executes an agent action and returns the result."""
        self.step_count += 1
        reward = 0.0
        
        if action.command == "read":
            self._handle_read(action.path)
        elif action.command == "write":
            self._handle_write(action.path, action.content)
        elif action.command == "run":
            reward = self._handle_run()
        elif action.command == "submit":
            reward = self._handle_submit()
            self.done = True

        obs = self._get_obs()
        return obs, reward, self.done, {}

    def _handle_read(self, path: str):
        try:
            full_path = os.path.join(self.workspace, path)
            with open(full_path, "r") as f:
                self.logs = f.read()
        except Exception as e:
            self.logs = f"Error reading file {path}: {str(e)}"

    def _handle_write(self, path: str, content: str):
        try:
            full_path = os.path.join(self.workspace, path)
            with open(full_path, "w") as f:
                f.write(content)
            self.logs = f"Successfully wrote to {path}."
        except Exception as e:
            self.logs = f"Error writing to {path}: {str(e)}"

    def _handle_run(self) -> float:
        """Executes the pipeline and measures time (efficiency)."""
        start_exec = time.time()
        try:
            # We run as a subprocess to isolate errors
            result = subprocess.run(
                ["python3", "pipeline.py"],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=15 # Prevent infinite loops
            )
            exec_duration = time.time() - start_exec
            self.logs = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\nExecution Time: {exec_duration:.4f}s"
            
            # Here we will later insert logic to update 'self.last_metrics'
            # based on the run result for iterative feedback.
            return 0.1 if result.returncode == 0 else -0.1
            
        except subprocess.TimeoutExpired:
            self.logs = "Execution Timeout! Your pipeline is too slow (Efficiency issue)."
            return -0.5
        except Exception as e:
            self.logs = f"System Error during run: {str(e)}"
            return -1.0

    def _handle_submit(self) -> float:
        """Final grading using the task's grader."""
        task = TASKS[self.current_task_id]
        # We pass self.logs and workspace for the grader to inspect
        result = task["grader"](self.workspace, self.logs)
        self.last_metrics = result["metrics"]
        return result["score"]

    def _get_obs(self) -> Observation:
        """Helper to construct the observation from the current state."""
        files = os.listdir(self.workspace) if self.workspace else []
        
        # Try to provide a data preview if 'output.csv' exists
        preview = "No output data found yet."
        output_path = os.path.join(self.workspace, "output.csv")
        if os.path.exists(output_path):
            try:
                df = pd.read_csv(output_path).head(5)
                preview = df.to_string()
            except:
                preview = "Output file exists but could not be parsed as CSV."

        return Observation(
            files=files,
            logs=self.logs,
            metrics=self.last_metrics,
            data_preview=preview,
            done=self.done
        )

    def state(self) -> State:
        return State(
            task_id=self.current_task_id,
            step_count=self.step_count,
            workspace_path=self.workspace or "",
            session_id="local_session" # In production, this would be a UUID
        )