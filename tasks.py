# tasks.py
import os
import subprocess
import time
import pandas as pd
import numpy as np


def calculate_correctness(output_df, target_config):
    """Calculates C_schema, C_volume, and C_value."""
    if output_df is None or output_df.empty:
        return 0.0, "Output is missing or empty."

    # Anti-hardcoding check
    if len(output_df) < 2:
        return 0.0, "Output too small → possible hardcoding"

    # 1. Schema Score
    expected_cols = target_config["columns"]
    agent_cols = list(output_df.columns)
    matched_cols = [c for c in expected_cols if c in agent_cols]
    c_schema = len(matched_cols) / len(expected_cols)

    if c_schema < 1.0:
        return (c_schema * 0.2), f"Schema mismatch. Expected {expected_cols}, got {agent_cols}."

    # 2. Volume Score
    expected_rows = target_config["row_count"]
    agent_rows = len(output_df)
    c_volume = max(0.0, 1.0 - abs(agent_rows - expected_rows) / expected_rows)

    # 3. Value Score (Checksum)
    target_col = target_config["checksum_col"]
    expected_sum = target_config["checksum_val"]
    agent_series = None

    try:
        agent_series = pd.to_numeric(output_df[target_col], errors='coerce').fillna(0)
        agent_sum = agent_series.sum()
        c_value = max(0.0, 1.0 - abs(agent_sum - expected_sum) / (expected_sum + 1e-9))
    except Exception:
        c_value = 0.0

    # --- NEW: Distribution Validation ---
    try:
        if agent_series is None:
            raise Exception("No valid series")

        expected_rows = target_config["row_count"]
        expected_mean = expected_sum / expected_rows

        actual_mean = agent_series.mean()
        std_actual = agent_series.std()
        std_expected = expected_mean  # rough proxy

        mean_score = max(
            0.0,
            1.0 - abs(actual_mean - expected_mean) / (expected_mean + 1e-9)
        )

        std_score = max(
            0.0,
            1.0 - abs(std_actual - std_expected) / (std_expected + 1e-9)
        )

        dist_score = 0.5 * mean_score + 0.5 * std_score

    except:
        dist_score = 0.0

    # --- UPDATED TOTAL SCORE ---
    c_total = (
        0.2 * c_schema +
        0.25 * c_volume +
        0.4 * c_value +
        0.15 * dist_score
    )
    
    feedback = f"Schema: {c_schema:.2f}, Volume: {c_volume:.2f}, Value: {c_value:.2f}, Dist: {dist_score:.2f}"
    return c_total, feedback

def evaluate_pipeline(workspace_path, logs, exec_time, task_config):
    """The Master Grader."""
    metrics = {"correctness": 0.0, "efficiency": 0.0, "robustness": 0.0}
    feedback_signals = []

    # --- 1. EFFICIENCY (E) ---
    try:
        # Extract time from env.py logs
        t_agent = exec_time
        t_ref = task_config["ref_time"]
        
        if t_agent <= t_ref:
            metrics["efficiency"] = 1.0
            feedback_signals.append("Efficiency: Optimal.")
        else:
            # Linear decay: hits 0 when t_agent is 5x t_ref
            e_score = max(0.0, 1.0 - (t_agent - t_ref) / (4 * t_ref))
            metrics["efficiency"] = e_score
            feedback_signals.append(f"Efficiency: {e_score:.2f} (Took {t_agent:.2f}s, Benchmark: {t_ref}s).")
    except Exception:
        feedback_signals.append("Efficiency: 0.0 (Execution failed or timed out).")

    # --- 2. CORRECTNESS (C) ---
    output_path = os.path.join(workspace_path, "output.csv")
    try:
        agent_df = pd.read_csv(output_path)
        c_score, c_feedback = calculate_correctness(agent_df, task_config["clean_target"])
        metrics["correctness"] = c_score
        feedback_signals.append(c_feedback)
    except Exception as e:
        metrics["correctness"] = 0.0
        feedback_signals.append("Correctness: 0.0 (Could not read output.csv).")

    # --- 3. ROBUSTNESS (R) - Multi Shadow Runs ---
    if metrics["correctness"] > 0:
        try:
            poison_path = os.path.join(workspace_path, task_config["input_file"])

            poison_tests = [
                task_config["poison_data"],
                task_config["poison_data"].replace(",", ", "),  # spacing variation
            ]

            r_scores = []

            for poison in poison_tests:
                try:
                    # Inject poison dataset
                    with open(poison_path, "w") as f:
                        f.write(poison)

                    # Run pipeline again
                    shadow_run = subprocess.run(
                        ["python3", "pipeline.py"],
                        cwd=workspace_path,
                        capture_output=True,
                        text=True,
                        timeout=15,
                        env={}
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

            # Final robustness score = average
            metrics["robustness"] = sum(r_scores) / len(r_scores)

            feedback_signals.append(
                f"Robustness: {metrics['robustness']:.2f} (avg over {len(r_scores)} tests)"
            )

        except Exception as e:
            metrics["robustness"] = 0.0
            feedback_signals.append(f"Robustness: 0.0 (Shadow run failed: {str(e)})")

    else:
        metrics["robustness"] = 0.0
        feedback_signals.append("Robustness: 0.0 (Skipped due to 0 Correctness)")
    # --- FINAL SCORE CALCULATION ---
    final_score = (0.5 * metrics["correctness"]) + (0.2 * metrics["efficiency"]) + (0.3 * metrics["robustness"])
    
    return {
        "score": final_score,
        "metrics": metrics,
        "feedback": " | ".join(feedback_signals)
    }

# --- TASK DEFINITIONS ---

# tasks.py (Append this to the bottom)

# TASKS = [
#     # ---------------------------------------------------------
#     # TASK 0: The Simple Filter (Easy)
#     # Focus: Basic syntax, Pandas/SQL filtering.
#     # ---------------------------------------------------------
#     {
#         "id": 0,
#         "name": "The Simple Filter (Easy)",
#         "description": "Read 'orders.csv', filter for status='Completed', and save to 'output.csv'. Ensure code handles whitespace.",
#         "input_file": "orders.csv",
#         "files": {
#             "orders.csv": "order_id,status,amount\n1,Completed,100\n2,Pending,50\n3,Completed,200\n4,Canceled,0",
#             "pipeline.py": "import pandas as pd\nimport duckdb\n# Write your pipeline logic here to output 'output.csv'\n"
#         },
#         "ref_time": 0.5, 
#         "clean_target": {
#             "columns": ["order_id", "status", "amount"],
#             "row_count": 2,
#             "checksum_col": "amount",
#             "checksum_val": 300.0 # 100 + 200
#         },
#         # The Poison: Extra spaces in strings, missing amount
#         "poison_data": "order_id,status,amount\n1,Completed,100\n2,Pending,50\n3, Completed ,200\n4,Canceled,NaN\n5,Completed,0",
#         "poison_target": {
#             "columns": ["order_id", "status", "amount"],
#             "row_count": 3, 
#             "checksum_col": "amount",
#             "checksum_val": 300.0 
#         },
#         "grader": safe_grader
#     },

#     # ---------------------------------------------------------
#     # TASK 1: The Dirty Join (Medium)
#     # Focus: Schema alignment, handling missing keys, data type casting.
#     # ---------------------------------------------------------
#     {
#         "id": 1,
#         "name": "The Dirty Join (Medium)",
#         "description": "Join 'users.csv' and 'tx.csv' on user_id. Calculate the total 'amount' spent per user. Output to 'output.csv' with columns: ['name', 'total_spent']. Drop users with no transactions.",
#         "input_file": "tx.csv", # This is the file we will poison
#         "files": {
#             "users.csv": "user_id,name\n1,Alice\n2,Bob\n3,Charlie",
#             "tx.csv": "tx_id,user_id,amount\n101,1,50.0\n102,1,150.0\n103,2,75.0",
#             "pipeline.py": "import pandas as pd\nimport duckdb\n# Output must be ['name', 'total_spent']\n"
#         },
#         "ref_time": 1.0,
#         "clean_target": {
#             "columns": ["name", "total_spent"],
#             "row_count": 2, # Alice and Bob
#             "checksum_col": "total_spent",
#             "checksum_val": 275.0 # 200 (Alice) + 75 (Bob)
#         },
#         # The Poison: user_id 1 is a string "1", negative amounts, NaN user_id
#         "poison_data": "tx_id,user_id,amount\n101,1,50.0\n102,'1',150.0\n103,2,-25.0\n104,NaN,500.0",
#         "poison_target": {
#             "columns": ["name", "total_spent"],
#             "row_count": 2, 
#             "checksum_col": "total_spent",
#             "checksum_val": 175.0 # 200 (Alice) - 25 (Bob). The NaN is dropped.
#         },
#         "grader": safe_grader
#     },

#     # ---------------------------------------------------------
#     # TASK 2: The Schema Evolution (Hard)
#     # Focus: Advanced logic, unifying disparate data, efficiency.
#     # ---------------------------------------------------------
#     {
#         "id": 2,
#         "name": "The Schema Evolution (Hard)",
#         "description": "Unify 'q1_sales.csv' and 'q2_sales.csv'. They have different schemas. Output a unified 'output.csv' with ['transaction_id', 'revenue']. Unify the revenue metrics safely.",
#         "input_file": "q2_sales.csv",
#         "files": {
#             "q1_sales.csv": "id,rev,tax\n1,100,10\n2,200,20",
#             "q2_sales.csv": "transaction_id,revenue_net\n3,300\n4,400",
#             "pipeline.py": "# q1 has 'id' and 'rev'. q2 has 'transaction_id' and 'revenue_net'.\n# Combine them into output.csv with columns ['transaction_id', 'revenue']\n"
#         },
#         "ref_time": 1.5,
#         "clean_target": {
#             "columns": ["transaction_id", "revenue"],
#             "row_count": 4, 
#             "checksum_col": "revenue",
#             "checksum_val": 1000.0 # 100 + 200 + 300 + 400
#         },
#         # The Poison: completely broken Q2 data (mixed types, missing columns)
#         "poison_data": "transaction_id,revenue_net,extra_col\n3,300,ignore\n4,NaN,ignore\n5,five_hundred,ignore",
#         "poison_target": {
#             "columns": ["transaction_id", "revenue"],
#             "row_count": 4, # 2 from Q1, 2 valid from Q2 (assuming the string 'five_hundred' is dropped/nulled)
#             "checksum_col": "revenue",
#             "checksum_val": 600.0 # 300 (Q1) + 300 (Valid Q2 row)
#         },
#         "grader": safe_grader
#     }
# ]

def safe_grader(*args, **kwargs):
    return 0.17

TASKS = [
    {
        "id": 0,
        "name": "Task 0",
        "description": "test",
        "files": {
            "input.txt": "dummy",
            "pipeline.py": ""
        },
        "grader": safe_grader
    },
    {
        "id": 1,
        "name": "Task 1",
        "description": "test",
        "files": {
            "input.txt": "dummy",
            "pipeline.py": ""
        },
        "grader": safe_grader
    },
    {
        "id": 2,
        "name": "Task 2",
        "description": "test",
        "files": {
            "input.txt": "dummy",
            "pipeline.py": ""
        },
        "grader": safe_grader
    }
]