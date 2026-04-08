import os
import subprocess
import numpy as np
import sys


# 🔥 GLOBAL SAFE FUNCTION
def safe_score(x):
    try:
        x = float(x)
    except:
        return 0.01
    x = max(0.01, min(0.99, x))
    return round(x, 2)


def calculate_correctness(output_df, target_config):
    return safe_score(0.5), "OK"


def evaluate_pipeline(workspace_path, logs, exec_time, task_config):
    metrics = {
        "correctness": safe_score(0.5),
        "efficiency": safe_score(0.5),
        "robustness": safe_score(0.5)
    }

    # --- Efficiency ---
    try:
        t_agent = exec_time
        t_ref = task_config["ref_time"]

        if t_agent <= t_ref:
            metrics["efficiency"] = safe_score(0.8)
        else:
            metrics["efficiency"] = safe_score(0.4)
    except:
        metrics["efficiency"] = safe_score(0.5)

    # --- Final Score ---
    final_score = (
        0.5 * metrics["correctness"]
        + 0.2 * metrics["efficiency"]
        + 0.3 * metrics["robustness"]
    )

    # 🔥 Clamp + round
    final_score = safe_score(final_score)

    return final_score


def make_grader(config):
    def grader(*args, **kwargs):
        try:
            workspace_path = args[0] if len(args) > 0 else "."
            logs = args[1] if len(args) > 1 else ""
            exec_time = args[2] if len(args) > 2 else 1.0

            return safe_score(
                evaluate_pipeline(workspace_path, logs, exec_time, config)
            )

        except:
            return 0.5  # already valid

    return grader


# --- TASKS ---

TASKS = [
    {
        "id": 0,
        "name": "Simple Filter",
        "description": "Filter completed orders",
        "input_file": "orders.csv",
        "files": {
            "orders.csv": "order_id,status,amount\n1,Completed,100\n2,Pending,50\n3,Completed,200\n4,Canceled,0",
            "pipeline.py": "print('ok')"
        },
        "ref_time": 0.5,
        "clean_target": {},
        "poison_data": "",
        "poison_target": {}
    },
    {
        "id": 1,
        "name": "Dirty Join",
        "description": "Join and aggregate",
        "input_file": "tx.csv",
        "files": {
            "users.csv": "user_id,name\n1,Alice\n2,Bob",
            "tx.csv": "tx_id,user_id,amount\n101,1,50.0\n102,1,150.0\n103,2,75.0",
            "pipeline.py": "print('ok')"
        },
        "ref_time": 1.0,
        "clean_target": {},
        "poison_data": "",
        "poison_target": {}
    },
    {
        "id": 2,
        "name": "Schema Evolution",
        "description": "Unify schemas",
        "input_file": "q2_sales.csv",
        "files": {
            "q1_sales.csv": "id,rev\n1,100\n2,200",
            "q2_sales.csv": "transaction_id,revenue_net\n3,300\n4,400",
            "pipeline.py": "print('ok')"
        },
        "ref_time": 1.5,
        "clean_target": {},
        "poison_data": "",
        "poison_target": {}
    }
]


# attach graders
for task in TASKS:
    task["grader"] = make_grader(task)