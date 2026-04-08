import os
import subprocess
import pandas as pd
import numpy as np
import sys


def calculate_correctness(output_df, target_config):
    if output_df is None or output_df.empty:
        return 0.0, "Output is missing or empty."

    if len(output_df) < 2:
        return 0.0, "Output too small → possible hardcoding"

    expected_cols = target_config["columns"]
    agent_cols = list(output_df.columns)
    matched_cols = [c for c in expected_cols if c in agent_cols]
    c_schema = len(matched_cols) / len(expected_cols)

    if c_schema < 1.0:
        return (c_schema * 0.2), f"Schema mismatch."

    expected_rows = target_config["row_count"]
    agent_rows = len(output_df)
    c_volume = max(0.0, 1.0 - abs(agent_rows - expected_rows) / expected_rows)

    target_col = target_config["checksum_col"]
    expected_sum = target_config["checksum_val"]

    try:
        agent_series = pd.to_numeric(output_df[target_col], errors='coerce').fillna(0)
        agent_sum = agent_series.sum()
        c_value = max(0.0, 1.0 - abs(agent_sum - expected_sum) / (expected_sum + 1e-9))
    except:
        c_value = 0.0

    c_total = (0.2 * c_schema + 0.25 * c_volume + 0.55 * c_value)

    return c_total, "OK"


def evaluate_pipeline(workspace_path, logs, exec_time, task_config):
    metrics = {"correctness": 0.0, "efficiency": 0.0, "robustness": 0.0}

    # --- Efficiency ---
    try:
        t_agent = exec_time
        t_ref = task_config["ref_time"]

        if t_agent <= t_ref:
            metrics["efficiency"] = 1.0
        else:
            metrics["efficiency"] = max(0.0, 1.0 - (t_agent - t_ref) / (4 * t_ref))
    except:
        metrics["efficiency"] = 0.0

    # --- Correctness ---
    output_path = os.path.join(workspace_path, "output.csv")

    try:
        agent_df = pd.read_csv(output_path)
        c_score, _ = calculate_correctness(agent_df, task_config["clean_target"])
        metrics["correctness"] = c_score
    except:
        metrics["correctness"] = 0.0

    # --- Robustness ---
    if metrics["correctness"] > 0:
        try:
            poison_path = os.path.join(workspace_path, task_config["input_file"])
            r_scores = []

            for poison in [task_config["poison_data"]]:
                try:
                    with open(poison_path, "w") as f:
                        f.write(poison)

                    safe_env = {
                        "PATH": os.environ.get("PATH", ""),
                        "PYTHONPATH": os.environ.get("PYTHONPATH", "")
                    }

                    shadow_run = subprocess.run(
                        [sys.executable, "pipeline.py"],
                        cwd=workspace_path,
                        capture_output=True,
                        text=True,
                        timeout=15,
                        env=safe_env
                    )

                    if shadow_run.returncode == 0:
                        shadow_df = pd.read_csv(output_path)
                        r_score, _ = calculate_correctness(
                            shadow_df,
                            task_config["poison_target"]
                        )
                        r_scores.append(r_score)
                    else:
                        r_scores.append(0.0)

                except:
                    r_scores.append(0.0)

            metrics["robustness"] = sum(r_scores) / len(r_scores)

        except:
            metrics["robustness"] = 0.0

    # --- Final Score ---
    final_score = (
        0.5 * metrics["correctness"]
        + 0.2 * metrics["efficiency"]
        + 0.3 * metrics["robustness"]
    )

    # 🔥 CRITICAL FIX: strict (0,1)
    final_score = float(np.clip(final_score, 0.001, 0.999))

    return final_score   # ✅ RETURN FLOAT ONLY


def make_grader(config):
    def grader(*args, **kwargs):
        try:
            workspace_path = args[0] if len(args) > 0 else "."
            logs = args[1] if len(args) > 1 else ""
            exec_time = args[2] if len(args) > 2 else 1.0

            return evaluate_pipeline(workspace_path, logs, exec_time, config)

        except:
            return 0.5  # safe fallback

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
            "pipeline.py": "import pandas as pd\n# write output.csv\n"
        },
        "ref_time": 0.5,
        "clean_target": {
            "columns": ["order_id", "status", "amount"],
            "row_count": 2,
            "checksum_col": "amount",
            "checksum_val": 300.0
        },
        "poison_data": "order_id,status,amount\n1,Completed,100\n3, Completed ,200\n5,Completed,0",
        "poison_target": {
            "columns": ["order_id", "status", "amount"],
            "row_count": 3,
            "checksum_col": "amount",
            "checksum_val": 300.0
        }
    },
    {
        "id": 1,
        "name": "Dirty Join",
        "description": "Join and aggregate",
        "input_file": "tx.csv",
        "files": {
            "users.csv": "user_id,name\n1,Alice\n2,Bob",
            "tx.csv": "tx_id,user_id,amount\n101,1,50.0\n102,1,150.0\n103,2,75.0",
            "pipeline.py": "# output.csv\n"
        },
        "ref_time": 1.0,
        "clean_target": {
            "columns": ["name", "total_spent"],
            "row_count": 2,
            "checksum_col": "total_spent",
            "checksum_val": 275.0
        },
        "poison_data": "tx_id,user_id,amount\n101,1,50.0\n102,'1',150.0",
        "poison_target": {
            "columns": ["name", "total_spent"],
            "row_count": 1,
            "checksum_col": "total_spent",
            "checksum_val": 200.0
        }
    },
    {
        "id": 2,
        "name": "Schema Evolution",
        "description": "Unify schemas",
        "input_file": "q2_sales.csv",
        "files": {
            "q1_sales.csv": "id,rev\n1,100\n2,200",
            "q2_sales.csv": "transaction_id,revenue_net\n3,300\n4,400",
            "pipeline.py": "# output.csv\n"
        },
        "ref_time": 1.5,
        "clean_target": {
            "columns": ["transaction_id", "revenue"],
            "row_count": 4,
            "checksum_col": "revenue",
            "checksum_val": 1000.0
        },
        "poison_data": "transaction_id,revenue_net\n3,300\n5,five_hundred",
        "poison_target": {
            "columns": ["transaction_id", "revenue"],
            "row_count": 3,
            "checksum_col": "revenue",
            "checksum_val": 600.0
        }
    }
]


# attach graders
for task in TASKS:
    task["grader"] = make_grader(task)