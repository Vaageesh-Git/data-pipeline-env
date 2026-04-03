# tasks.py
import os
import pandas as pd
import duckdb

def placeholder_grader(workspace_path, logs):
    """
    RESEARCH NOTE: We will define the specific logic for 
    Correctness, Efficiency, and Robustness here later.
    """
    return {
        "score": 0.0,
        "metrics": {"correctness": 0.0, "efficiency": 0.0, "robustness": 0.0}
    }

TASKS = [
    {
        "id": 0,
        "name": "The Simple Filter (Easy)",
        "description": "Create a pipeline that filters 'orders.csv' for 'Completed' status only.",
        "files": {
            "orders.csv": "order_id,status,amount\n1,Completed,100\n2,Pending,50\n3,Completed,200",
            "pipeline.py": "# Write your code here\nimport pandas as pd\n"
        },
        "grader": placeholder_grader
    },
    {
        "id": 1,
        "name": "The Dirty Join (Medium)",
        "description": "Join 'users.csv' and 'transactions.csv'. Handle missing user_ids and type mismatches.",
        "files": {
            "users.csv": "id,name\n1,Alice\n2,Bob\nNULL,Charlie",
            "transactions.csv": "user_id,amount\n1,20.0\n'2',40.0\n3,10.0",
            "pipeline.py": "import duckdb\n"
        },
        "grader": placeholder_grader
    },
    {
        "id": 2,
        "name": "High-Volume Aggregation (Hard)",
        "description": "Process 100,000 rows. Optimize for memory and speed. Avoid slow Python loops.",
        "files": {
            "pipeline.py": "import duckdb\n# Generate large data locally or query efficiently\n"
        },
        "grader": placeholder_grader
    }
]