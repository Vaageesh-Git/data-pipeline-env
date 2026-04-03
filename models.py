# models.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class Action(BaseModel):
    command: str = Field(..., description="The action to take: 'read', 'write', 'run', or 'submit'")
    path: Optional[str] = Field(None, description="The file path to read or write (e.g., 'pipeline.py')")
    content: Optional[str] = Field(None, description="The code content to write to the file")

class Observation(BaseModel):
    files: List[str] = Field(..., description="List of files in the current workspace")
    logs: str = Field(..., description="Stdout/Stderr from the last pipeline execution")
    metrics: Dict[str, float] = Field(
        default_factory=lambda: {"correctness": 0.0, "efficiency": 0.0, "robustness": 0.0},
        description="Iterative feedback signals for the agent"
    )
    data_preview: str = Field(..., description="A snippet (head) of the output data table")
    done: bool = Field(False, description="Whether the task is finished")

class State(BaseModel):
    task_id: int
    step_count: int
    workspace_path: str
    session_id: str