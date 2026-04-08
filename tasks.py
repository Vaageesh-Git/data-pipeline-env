import io
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time

import pandas as pd


def clamp_metric(value):
    try:
        return round(max(0.0, min(1.0, float(value))), 4)
    except (TypeError, ValueError):
        return 0.0


def average(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def efficiency_score(duration, ref_time):
    if duration <= ref_time:
        return 1.0
    return clamp_metric(1.0 - ((duration - ref_time) / (4.0 * ref_time)))


def starter_pipeline():
    return textwrap.dedent(
        """
        import pandas as pd


        def main():
            # Read the task input files from the current directory.
            # Write your final answer to output.csv.
            raise NotImplementedError("Implement the pipeline for this task")


        if __name__ == "__main__":
            main()
        """
    ).strip() + "\n"


def build_task_file(title, instructions):
    return textwrap.dedent(
        f"""
        # {title}

        {instructions.strip()}

        Rules:
        - Read the CSV inputs from the current working directory.
        - Write the final result to `output.csv`.
        - Keep the requested output columns and ordering.
        - Use only local file operations; no network access is required.
        - Hidden shadow datasets will be used during `submit`.
        """
    ).strip() + "\n"


def read_csv_text(text):
    return pd.read_csv(io.StringIO(text.strip()))


def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_lower(value):
    return normalize_text(value).lower()


def normalize_upper(value):
    return normalize_text(value).upper()


def normalize_key(series):
    return series.map(normalize_lower)


def normalize_label(series):
    return series.map(normalize_upper)


def parse_timestamp_series(series):
    return pd.to_datetime(series, errors="coerce")


def parse_optional_timestamp(series):
    normalized = series.map(normalize_text)
    normalized = normalized.replace({"": pd.NA})
    return pd.to_datetime(normalized, errors="coerce")


def load_case_frames(case_files):
    frames = {}
    for name, content in case_files.items():
        if name.endswith(".csv"):
            frames[name] = read_csv_text(content)
    return frames


def canonicalize_frame(df):
    canonical = df.copy()
    canonical.columns = [normalize_text(column) for column in canonical.columns]
    for column in canonical.columns:
        series = canonical[column]
        if pd.api.types.is_bool_dtype(series):
            canonical[column] = series.map(lambda value: "true" if bool(value) else "false")
        elif pd.api.types.is_numeric_dtype(series):
            numeric_series = pd.to_numeric(series, errors="coerce")
            canonical[column] = numeric_series.map(lambda value: round(float(value), 6) if pd.notna(value) else pd.NA)
        else:
            canonical[column] = series.map(normalize_text)
    return canonical


def cell_equal(left, right):
    if pd.isna(left) and pd.isna(right):
        return True
    try:
        left_num = float(left)
        right_num = float(right)
        if pd.notna(left_num) and pd.notna(right_num):
            return abs(left_num - right_num) <= 1e-6
    except (TypeError, ValueError):
        pass
    return normalize_text(left) == normalize_text(right)


def closeness(actual, expected):
    try:
        actual = float(actual)
        expected = float(expected)
    except (TypeError, ValueError):
        return 0.0
    denominator = max(abs(expected), 1.0)
    return max(0.0, 1.0 - (abs(actual - expected) / denominator))


def compare_frames(expected_df, actual_df, sort_by):
    expected = canonicalize_frame(expected_df)
    actual = canonicalize_frame(actual_df)

    expected_columns = list(expected.columns)
    actual_columns = list(actual.columns)
    shared_columns = [column for column in expected_columns if column in actual_columns]

    if actual_columns == expected_columns:
        schema_score = 1.0
    else:
        extras = len([column for column in actual_columns if column not in expected_columns])
        schema_score = max(0.0, (len(shared_columns) - (0.5 * extras)) / max(len(expected_columns), 1))

    volume_score = max(0.0, 1.0 - (abs(len(actual) - len(expected)) / max(len(expected), 1)))

    if shared_columns:
        sort_columns = [column for column in sort_by if column in shared_columns]
        expected_view = expected[shared_columns].copy()
        actual_view = actual[shared_columns].copy()
        if sort_columns:
            expected_view = expected_view.sort_values(sort_columns, kind="mergesort")
            actual_view = actual_view.sort_values(sort_columns, kind="mergesort")
        expected_view = expected_view.reset_index(drop=True)
        actual_view = actual_view.reset_index(drop=True)

        overlap_rows = min(len(expected_view), len(actual_view))
        matched_cells = 0
        for row_index in range(overlap_rows):
            for column in shared_columns:
                if cell_equal(expected_view.at[row_index, column], actual_view.at[row_index, column]):
                    matched_cells += 1
        total_cells = max(len(expected_view), len(actual_view), 1) * len(shared_columns)
        value_score = matched_cells / total_cells

        numeric_columns = [column for column in shared_columns if pd.api.types.is_numeric_dtype(expected_view[column])]
        if numeric_columns:
            distribution_components = []
            for column in numeric_columns:
                distribution_components.append(
                    closeness(actual_view[column].sum(), expected_view[column].sum())
                )
                distribution_components.append(
                    closeness(actual_view[column].mean(), expected_view[column].mean())
                )
            distribution_score = average(distribution_components)
        else:
            distribution_score = value_score
    else:
        value_score = 0.0
        distribution_score = 0.0

    correctness = (
        (0.25 * schema_score)
        + (0.25 * volume_score)
        + (0.35 * value_score)
        + (0.15 * distribution_score)
    )

    return {
        "correctness": clamp_metric(correctness),
        "schema_score": clamp_metric(schema_score),
        "volume_score": clamp_metric(volume_score),
        "value_score": clamp_metric(value_score),
        "distribution_score": clamp_metric(distribution_score),
        "expected_columns": expected_columns,
        "actual_columns": actual_columns,
        "expected_rows": len(expected),
        "actual_rows": len(actual),
    }


def format_case_feedback(case_name, case_result):
    if case_result.get("timed_out"):
        return f"{case_name}: execution timed out after {case_result['duration']:.3f}s"

    if case_result.get("returncode") not in (None, 0):
        stderr = case_result.get("stderr", "").strip()
        if len(stderr) > 200:
            stderr = stderr[:200] + "..."
        return (
            f"{case_name}: pipeline crashed (exit {case_result['returncode']})"
            + (f" stderr={stderr}" if stderr else "")
        )

    if not case_result.get("output_found"):
        return f"{case_name}: missing output.csv"

    details = case_result["details"]
    return (
        f"{case_name}: correctness={case_result['correctness']:.2f}, "
        f"schema={details['schema_score']:.2f}, rows={details['actual_rows']}/{details['expected_rows']}"
    )


def run_pipeline(workspace_path, task):
    entrypoint = os.path.join(workspace_path, task["entrypoint"])
    if not os.path.exists(entrypoint):
        return {
            "returncode": 1,
            "stdout": "",
            "stderr": f"Missing entrypoint: {task['entrypoint']}",
            "duration": 0.0,
            "timed_out": False,
        }

    output_path = os.path.join(workspace_path, task["output_file"])
    if os.path.exists(output_path):
        os.remove(output_path)

    started_at = time.time()
    try:
        result = subprocess.run(
            [sys.executable, task["entrypoint"]],
            cwd=workspace_path,
            capture_output=True,
            text=True,
            timeout=task.get("timeout", 20),
            env=os.environ.copy(),
        )
        duration = time.time() - started_at
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration": duration,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as error:
        duration = time.time() - started_at
        return {
            "returncode": 124,
            "stdout": error.stdout or "",
            "stderr": error.stderr or "",
            "duration": duration,
            "timed_out": True,
        }


def evaluate_case_output(task, workspace_path, case, execution_result):
    if execution_result["timed_out"] or execution_result["returncode"] != 0:
        return {
            "correctness": 0.0,
            "output_found": False,
            "details": {
                "schema_score": 0.0,
                "volume_score": 0.0,
                "value_score": 0.0,
                "distribution_score": 0.0,
                "expected_columns": [],
                "actual_columns": [],
                "expected_rows": 0,
                "actual_rows": 0,
            },
        }

    output_path = os.path.join(workspace_path, task["output_file"])
    if not os.path.exists(output_path):
        return {
            "correctness": 0.0,
            "output_found": False,
            "details": {
                "schema_score": 0.0,
                "volume_score": 0.0,
                "value_score": 0.0,
                "distribution_score": 0.0,
                "expected_columns": [],
                "actual_columns": [],
                "expected_rows": 0,
                "actual_rows": 0,
            },
        }

    try:
        actual_df = pd.read_csv(output_path)
    except Exception:
        return {
            "correctness": 0.0,
            "output_found": True,
            "details": {
                "schema_score": 0.0,
                "volume_score": 0.0,
                "value_score": 0.0,
                "distribution_score": 0.0,
                "expected_columns": [],
                "actual_columns": [],
                "expected_rows": 0,
                "actual_rows": 0,
            },
        }

    expected_df = task["transform"](load_case_frames(case["files"]))
    details = compare_frames(expected_df, actual_df, task["sort_by"])
    return {
        "correctness": details["correctness"],
        "output_found": True,
        "details": details,
    }


def build_case_result(task, case, workspace_path, execution_result):
    output_result = evaluate_case_output(task, workspace_path, case, execution_result)
    result = {
        "name": case["name"],
        "returncode": execution_result["returncode"],
        "stdout": execution_result["stdout"],
        "stderr": execution_result["stderr"],
        "duration": execution_result["duration"],
        "timed_out": execution_result["timed_out"],
        "efficiency": efficiency_score(execution_result["duration"], task["ref_time"]),
        "correctness": output_result["correctness"],
        "output_found": output_result["output_found"],
        "details": output_result["details"],
    }
    result["feedback"] = format_case_feedback(case["name"], result)
    return result


def preview_run(task, workspace_path, execution_result):
    visible_case = task["public_case"]
    case_result = build_case_result(task, visible_case, workspace_path, execution_result)

    correctness = case_result["correctness"]
    efficiency = case_result["efficiency"]
    robustness = 1.0 if case_result["returncode"] == 0 and case_result["output_found"] else 0.0
    score = (0.5 * correctness) + (0.2 * efficiency) + (0.3 * robustness)
    if case_result["returncode"] != 0:
        score -= 0.3
    if case_result["timed_out"]:
        score = -1.0

    return {
        "score": round(score, 4),
        "metrics": {
            "correctness": clamp_metric(correctness),
            "efficiency": clamp_metric(efficiency),
            "robustness": clamp_metric(robustness),
        },
        "feedback": case_result["feedback"],
        "cases": [case_result],
    }


def clone_workspace_for_case(workspace_path, task, case):
    clone_path = tempfile.mkdtemp(prefix=f"grade_task_{task['id']}_")
    shutil.copytree(workspace_path, clone_path, dirs_exist_ok=True)

    output_path = os.path.join(clone_path, task["output_file"])
    if os.path.exists(output_path):
        os.remove(output_path)

    for input_file in task["input_files"]:
        file_path = os.path.join(clone_path, input_file)
        if os.path.exists(file_path):
            os.remove(file_path)

    for file_name, content in case["files"].items():
        with open(os.path.join(clone_path, file_name), "w", encoding="utf-8") as handle:
            handle.write(content)

    return clone_path


def grade_submission(task, workspace_path, logs=None, last_exec_time=None):
    case_results = []
    all_cases = [task["public_case"]] + task["hidden_cases"]

    for case in all_cases:
        clone_path = clone_workspace_for_case(workspace_path, task, case)
        try:
            execution_result = run_pipeline(clone_path, task)
            case_results.append(build_case_result(task, case, clone_path, execution_result))
        finally:
            shutil.rmtree(clone_path, ignore_errors=True)

    correctness = case_results[0]["correctness"]
    hidden_scores = [result["correctness"] for result in case_results[1:]]
    robustness = average(hidden_scores) if hidden_scores else correctness
    efficiency = average([result["efficiency"] for result in case_results])
    score = (0.5 * correctness) + (0.2 * efficiency) + (0.3 * robustness)

    feedback_lines = [
        f"Final evaluation for {task['name']}",
        f"- correctness={correctness:.2f}",
        f"- efficiency={efficiency:.2f}",
        f"- robustness={robustness:.2f}",
    ]
    for result in case_results:
        feedback_lines.append(f"- {result['feedback']}")

    return {
        "score": round(score, 4),
        "metrics": {
            "correctness": clamp_metric(correctness),
            "efficiency": clamp_metric(efficiency),
            "robustness": clamp_metric(robustness),
        },
        "feedback": "\n".join(feedback_lines),
        "cases": case_results,
    }


def select_interval_row(frame, filters, timestamp):
    scoped = frame.copy()
    for column, value in filters.items():
        scoped = scoped[scoped[column] == value]
    scoped = scoped[
        (scoped["valid_from"] <= timestamp)
        & (scoped["valid_to"].isna() | (timestamp < scoped["valid_to"]))
    ]
    if scoped.empty:
        return None
    scoped = scoped.sort_values(["valid_from", "valid_to"], ascending=[False, False], na_position="last")
    return scoped.iloc[0]


def transform_temporal_discount_attribution(frames):
    history = frames["customer_history.csv"].copy()
    rules = frames["discount_rules.csv"].copy()
    orders = frames["orders.csv"].copy()

    history["customer_id"] = normalize_key(history["customer_id"])
    history["tier"] = normalize_label(history["tier"])
    history["region"] = normalize_label(history["region"])
    history["valid_from"] = parse_timestamp_series(history["valid_from"])
    history["valid_to"] = parse_optional_timestamp(history["valid_to"])

    rules["tier"] = normalize_label(rules["tier"])
    rules["sku"] = normalize_key(rules["sku"])
    rules["discount_pct"] = pd.to_numeric(rules["discount_pct"], errors="coerce").fillna(0.0)
    rules["valid_from"] = parse_timestamp_series(rules["valid_from"])
    rules["valid_to"] = parse_optional_timestamp(rules["valid_to"])

    orders["customer_id"] = normalize_key(orders["customer_id"])
    orders["sku"] = normalize_key(orders["sku"])
    orders["order_ts"] = parse_timestamp_series(orders["order_ts"])
    orders["gross_amount"] = pd.to_numeric(orders["gross_amount"], errors="coerce").fillna(0.0)

    rows = []
    for order in orders.itertuples(index=False):
        customer_row = select_interval_row(
            history,
            {"customer_id": order.customer_id},
            order.order_ts,
        )
        if customer_row is None:
            continue

        rule_row = select_interval_row(
            rules,
            {"tier": customer_row["tier"], "sku": order.sku},
            order.order_ts,
        )
        discount_pct = float(rule_row["discount_pct"]) if rule_row is not None else 0.0
        net_revenue = round(float(order.gross_amount) * (1.0 - (discount_pct / 100.0)), 2)

        rows.append(
            {
                "region": customer_row["region"],
                "tier": customer_row["tier"],
                "order_count": 1,
                "net_revenue": net_revenue,
            }
        )

    output = pd.DataFrame(rows, columns=["region", "tier", "order_count", "net_revenue"])
    if output.empty:
        return output
    output = (
        output.groupby(["region", "tier"], as_index=False)
        .agg(order_count=("order_count", "sum"), net_revenue=("net_revenue", "sum"))
        .sort_values(["region", "tier"])
    )
    output["net_revenue"] = output["net_revenue"].map(lambda value: round(float(value), 2))
    return output.reset_index(drop=True)


def apply_partial_customer_update(existing, row):
    record = dict(existing) if existing else {
        "customer_id": normalize_lower(row["customer_id"]),
        "name": "",
        "email": "",
        "status": "",
        "loyalty_points": 0,
    }

    name = normalize_text(row.get("name", ""))
    email = normalize_lower(row.get("email", ""))
    status = normalize_lower(row.get("status", ""))
    loyalty_points = normalize_text(row.get("loyalty_points", ""))

    if name:
        record["name"] = name
    if email:
        record["email"] = email
    if status:
        record["status"] = status
    if loyalty_points:
        record["loyalty_points"] = int(float(loyalty_points))

    return record


def transform_customer_cdc_merge(frames):
    snapshot = frames["snapshot_customers.csv"].copy()
    cdc = frames["customer_cdc.csv"].copy()

    snapshot["customer_id"] = normalize_key(snapshot["customer_id"])
    snapshot["email"] = normalize_key(snapshot["email"])
    snapshot["status"] = normalize_key(snapshot["status"])
    snapshot["loyalty_points"] = pd.to_numeric(snapshot["loyalty_points"], errors="coerce").fillna(0).astype(int)

    state = {
        row.customer_id: {
            "customer_id": row.customer_id,
            "name": normalize_text(row.name),
            "email": row.email,
            "status": row.status,
            "loyalty_points": int(row.loyalty_points),
        }
        for row in snapshot.itertuples(index=False)
    }

    cdc["customer_id"] = normalize_key(cdc["customer_id"])
    cdc["op"] = normalize_key(cdc["op"])
    cdc["event_ts"] = parse_timestamp_series(cdc["event_ts"])
    cdc["seq"] = pd.to_numeric(cdc["seq"], errors="coerce").fillna(0).astype(int)

    cdc = cdc.sort_values(["event_id", "seq", "event_ts"], kind="mergesort")
    cdc = cdc.drop_duplicates(subset=["event_id"], keep="last")
    cdc = cdc.sort_values(["event_ts", "seq", "event_id"], kind="mergesort")

    for row in cdc.to_dict("records"):
        customer_id = row["customer_id"]
        op = row["op"]
        if op == "delete":
            state.pop(customer_id, None)
            continue
        if op in {"insert", "update", "upsert"}:
            state[customer_id] = apply_partial_customer_update(state.get(customer_id), row)

    output = pd.DataFrame(list(state.values()), columns=[
        "customer_id",
        "name",
        "email",
        "status",
        "loyalty_points",
    ])
    if output.empty:
        return output
    output = output.sort_values("customer_id").reset_index(drop=True)
    output["loyalty_points"] = output["loyalty_points"].astype(int)
    return output


def build_fx_lookup(fx_rates):
    fx_rates = fx_rates.copy()
    fx_rates["rate_date"] = pd.to_datetime(fx_rates["rate_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    fx_rates["from_currency"] = normalize_label(fx_rates["from_currency"])
    fx_rates["to_currency"] = normalize_label(fx_rates["to_currency"])
    fx_rates["rate"] = pd.to_numeric(fx_rates["rate"], errors="coerce").fillna(0.0)

    lookup = {}
    for row in fx_rates.itertuples(index=False):
        if row.to_currency == "USD":
            lookup[(row.rate_date, row.from_currency)] = float(row.rate)
    return lookup


def convert_to_usd(amount, currency, timestamp, lookup):
    amount = float(amount)
    currency = normalize_upper(currency)
    if currency == "USD":
        return amount
    date_key = pd.to_datetime(timestamp, errors="coerce").strftime("%Y-%m-%d")
    rate = lookup.get((date_key, currency), 0.0)
    return round(amount * rate, 6)


def transform_payment_reconciliation(frames):
    payments = frames["payments.csv"].copy()
    refunds = frames["refunds.csv"].copy()
    chargebacks = frames["chargebacks.csv"].copy()
    fx_lookup = build_fx_lookup(frames["fx_rates.csv"])

    payments["payment_id"] = normalize_key(payments["payment_id"])
    payments["merchant_id"] = normalize_label(payments["merchant_id"])
    payments["currency"] = normalize_label(payments["currency"])
    payments["status"] = normalize_key(payments["status"])
    payments["captured_ts"] = parse_timestamp_series(payments["captured_ts"])
    payments["amount"] = pd.to_numeric(payments["amount"], errors="coerce").fillna(0.0)

    captured = payments[payments["status"] == "captured"].copy()
    captured["payment_usd"] = [
        convert_to_usd(row.amount, row.currency, row.captured_ts, fx_lookup)
        for row in captured.itertuples(index=False)
    ]

    payment_lookup = captured.set_index("payment_id")[["merchant_id", "currency"]].to_dict("index")

    refunds["payment_id"] = normalize_key(refunds["payment_id"])
    refunds["status"] = normalize_key(refunds["status"])
    refunds["refund_ts"] = parse_timestamp_series(refunds["refund_ts"])
    refunds["amount"] = pd.to_numeric(refunds["amount"], errors="coerce").fillna(0.0)
    refunds = refunds[refunds["status"] == "succeeded"].copy()
    refunds = refunds[refunds["payment_id"].isin(payment_lookup)].copy()
    refunds["merchant_id"] = refunds["payment_id"].map(lambda payment_id: payment_lookup[payment_id]["merchant_id"])
    refunds["currency"] = refunds["payment_id"].map(lambda payment_id: payment_lookup[payment_id]["currency"])
    refunds["refund_usd"] = [
        convert_to_usd(row.amount, row.currency, row.refund_ts, fx_lookup)
        for row in refunds.itertuples(index=False)
    ]

    chargebacks["payment_id"] = normalize_key(chargebacks["payment_id"])
    chargebacks["status"] = normalize_key(chargebacks["status"])
    chargebacks["chargeback_ts"] = parse_timestamp_series(chargebacks["chargeback_ts"])
    chargebacks["amount"] = pd.to_numeric(chargebacks["amount"], errors="coerce").fillna(0.0)
    chargebacks = chargebacks[chargebacks["status"] == "lost"].copy()
    chargebacks = chargebacks[chargebacks["payment_id"].isin(payment_lookup)].copy()
    chargebacks["merchant_id"] = chargebacks["payment_id"].map(lambda payment_id: payment_lookup[payment_id]["merchant_id"])
    chargebacks["currency"] = chargebacks["payment_id"].map(lambda payment_id: payment_lookup[payment_id]["currency"])
    chargebacks["chargeback_usd"] = [
        convert_to_usd(row.amount, row.currency, row.chargeback_ts, fx_lookup)
        for row in chargebacks.itertuples(index=False)
    ]

    payment_summary = captured.groupby("merchant_id", as_index=False).agg(
        captured_payments=("payment_id", "count"),
        payment_usd=("payment_usd", "sum"),
    )
    refund_summary = refunds.groupby("merchant_id", as_index=False).agg(
        refund_events=("refund_id", "count"),
        refund_usd=("refund_usd", "sum"),
    )
    chargeback_summary = chargebacks.groupby("merchant_id", as_index=False).agg(
        lost_chargebacks=("case_id", "count"),
        chargeback_usd=("chargeback_usd", "sum"),
    )

    output = payment_summary.merge(refund_summary, on="merchant_id", how="left")
    output = output.merge(chargeback_summary, on="merchant_id", how="left")
    output = output.fillna({
        "refund_events": 0,
        "refund_usd": 0.0,
        "lost_chargebacks": 0,
        "chargeback_usd": 0.0,
    })
    output["net_usd"] = output["payment_usd"] - output["refund_usd"] - output["chargeback_usd"]
    output = output[["merchant_id", "captured_payments", "refund_events", "lost_chargebacks", "net_usd"]]
    output["captured_payments"] = output["captured_payments"].astype(int)
    output["refund_events"] = output["refund_events"].astype(int)
    output["lost_chargebacks"] = output["lost_chargebacks"].astype(int)
    output["net_usd"] = output["net_usd"].map(lambda value: round(float(value), 2))
    output = output.sort_values("merchant_id").reset_index(drop=True)
    return output


def transform_session_funnel(frames):
    identity_map = frames["identity_map.csv"].copy()
    events = frames["events.csv"].copy()

    identity_map["identity_type"] = normalize_key(identity_map["identity_type"])
    identity_map["identity_value"] = normalize_key(identity_map["identity_value"])
    identity_map["user_id"] = normalize_key(identity_map["user_id"])
    identity_map["valid_from"] = parse_timestamp_series(identity_map["valid_from"])
    identity_map["valid_to"] = parse_optional_timestamp(identity_map["valid_to"])

    events["identity_type"] = normalize_key(events["identity_type"])
    events["identity_value"] = normalize_key(events["identity_value"])
    events["event_type"] = normalize_key(events["event_type"])
    events["event_ts"] = parse_timestamp_series(events["event_ts"])
    events["order_value"] = pd.to_numeric(events["order_value"], errors="coerce").fillna(0.0)

    resolved_rows = []
    for event in events.itertuples(index=False):
        mapping = select_interval_row(
            identity_map,
            {"identity_type": event.identity_type, "identity_value": event.identity_value},
            event.event_ts,
        )
        if mapping is None:
            continue
        resolved_rows.append(
            {
                "event_id": event.event_id,
                "event_ts": event.event_ts,
                "user_id": mapping["user_id"],
                "event_type": event.event_type,
                "order_value": float(event.order_value),
            }
        )

    resolved = pd.DataFrame(resolved_rows, columns=["event_id", "event_ts", "user_id", "event_type", "order_value"])
    if resolved.empty:
        return pd.DataFrame(columns=["user_id", "sessions", "converting_sessions", "revenue"])

    resolved = resolved.sort_values(["user_id", "event_ts", "event_id"], kind="mergesort").reset_index(drop=True)

    session_records = []
    for user_id, user_events in resolved.groupby("user_id", sort=False):
        session_index = 0
        previous_ts = None
        for row in user_events.itertuples(index=False):
            if previous_ts is None or (row.event_ts - previous_ts).total_seconds() > 1800:
                session_index += 1
            session_records.append(
                {
                    "user_id": user_id,
                    "session_id": session_index,
                    "event_type": row.event_type,
                    "order_value": row.order_value,
                }
            )
            previous_ts = row.event_ts

    sessions = pd.DataFrame(session_records)

    summaries = []
    for user_id, user_sessions in sessions.groupby("user_id"):
        total_sessions = user_sessions["session_id"].nunique()
        converting_sessions = 0
        revenue = 0.0
        for _, session_frame in user_sessions.groupby("session_id"):
            event_types = set(session_frame["event_type"].tolist())
            if {"view_product", "add_to_cart", "purchase"}.issubset(event_types):
                converting_sessions += 1
                revenue += session_frame.loc[session_frame["event_type"] == "purchase", "order_value"].sum()
        summaries.append(
            {
                "user_id": user_id,
                "sessions": int(total_sessions),
                "converting_sessions": int(converting_sessions),
                "revenue": round(revenue, 2),
            }
        )

    output = pd.DataFrame(summaries, columns=["user_id", "sessions", "converting_sessions", "revenue"])
    return output.sort_values("user_id").reset_index(drop=True)


def transform_bom_fulfillment(frames):
    inventory = frames["inventory_snapshots.csv"].copy()
    receipts = frames["receipts.csv"].copy()
    bom = frames["bom.csv"].copy()
    orders = frames["orders.csv"].copy()

    inventory["component_id"] = normalize_key(inventory["component_id"])
    inventory["warehouse_id"] = normalize_label(inventory["warehouse_id"])
    inventory["on_hand"] = pd.to_numeric(inventory["on_hand"], errors="coerce").fillna(0.0)

    receipts["component_id"] = normalize_key(receipts["component_id"])
    receipts["warehouse_id"] = normalize_label(receipts["warehouse_id"])
    receipts["arrival_ts"] = parse_timestamp_series(receipts["arrival_ts"])
    receipts["qty"] = pd.to_numeric(receipts["qty"], errors="coerce").fillna(0.0)
    receipts = receipts.sort_values(["arrival_ts", "warehouse_id", "component_id"], kind="mergesort").reset_index(drop=True)

    bom["sku"] = normalize_key(bom["sku"])
    bom["component_id"] = normalize_key(bom["component_id"])
    bom["units_per_sku"] = pd.to_numeric(bom["units_per_sku"], errors="coerce").fillna(0.0)

    orders["warehouse_id"] = normalize_label(orders["warehouse_id"])
    orders["sku"] = normalize_key(orders["sku"])
    orders["order_ts"] = parse_timestamp_series(orders["order_ts"])
    orders["qty"] = pd.to_numeric(orders["qty"], errors="coerce").fillna(0.0)
    orders["priority"] = pd.to_numeric(orders["priority"], errors="coerce").fillna(0).astype(int)
    orders = orders.sort_values(["order_ts", "priority", "order_id"], kind="mergesort").reset_index(drop=True)

    stock = {}
    for row in inventory.itertuples(index=False):
        stock[(row.warehouse_id, row.component_id)] = float(row.on_hand)

    receipt_pointer = 0
    results = []
    for order in orders.itertuples(index=False):
        while receipt_pointer < len(receipts) and receipts.at[receipt_pointer, "arrival_ts"] <= order.order_ts:
            receipt = receipts.iloc[receipt_pointer]
            key = (receipt["warehouse_id"], receipt["component_id"])
            stock[key] = stock.get(key, 0.0) + float(receipt["qty"])
            receipt_pointer += 1

        requirements = bom[bom["sku"] == order.sku].copy()
        shortages = []
        allocations = []
        for requirement in requirements.itertuples(index=False):
            key = (order.warehouse_id, requirement.component_id)
            available = stock.get(key, 0.0)
            needed = float(requirement.units_per_sku) * float(order.qty)
            allocations.append((key, needed))
            if available + 1e-9 < needed:
                shortages.append(requirement.component_id)

        if shortages:
            results.append(
                {
                    "order_id": int(order.order_id),
                    "warehouse_id": order.warehouse_id,
                    "fulfilled": "false",
                    "missing_components": ";".join(sorted(shortages)),
                }
            )
            continue

        for key, needed in allocations:
            stock[key] = stock.get(key, 0.0) - needed

        results.append(
            {
                "order_id": int(order.order_id),
                "warehouse_id": order.warehouse_id,
                "fulfilled": "true",
                "missing_components": "",
            }
        )

    output = pd.DataFrame(results, columns=["order_id", "warehouse_id", "fulfilled", "missing_components"])
    return output.sort_values("order_id").reset_index(drop=True)


TASKS = [
    {
        "id": 0,
        "name": "Temporal Discount Attribution",
        "description": "Resolve customer tier and SKU discounts with temporal joins, then aggregate net revenue by region and tier.",
        "instructions": (
            "Read `customer_history.csv`, `discount_rules.csv`, and `orders.csv`. Normalize `customer_id` and `sku` by trimming whitespace and lowercasing. "
            "Normalize `tier` and `region` to uppercase. For each order, resolve the effective customer-history row where `valid_from <= order_ts < valid_to`, "
            "treating blank `valid_to` as open-ended; if multiple rows match, keep the row with the latest `valid_from`. Then resolve the effective SKU discount rule "
            "for that order's tier and SKU using the same temporal logic. If no discount rule matches, use `discount_pct = 0`. Compute `net_revenue = gross_amount * (1 - discount_pct / 100)` "
            "rounded to 2 decimals. Write `output.csv` with columns `region`, `tier`, `order_count`, `net_revenue`, sorted by `region`, `tier`."
        ),
        "entrypoint": "pipeline.py",
        "output_file": "output.csv",
        "input_files": ["customer_history.csv", "discount_rules.csv", "orders.csv"],
        "sort_by": ["region", "tier"],
        "ref_time": 1.0,
        "timeout": 20,
        "transform": transform_temporal_discount_attribution,
        "public_case": {
            "name": "public",
            "files": {
                "customer_history.csv": """customer_id,tier,region,valid_from,valid_to\n cust-a ,bronze,us,2024-01-01T00:00:00,2024-02-15T00:00:00\ncust-a,gold,us,2024-02-15T00:00:00,\ncust-b,silver,eu,2024-01-01T00:00:00,\n""",
                "discount_rules.csv": """tier,sku,discount_pct,valid_from,valid_to\nBRONZE,sku-1,0,2024-01-01T00:00:00,\nSILVER,sku-1,10,2024-01-01T00:00:00,\nGOLD,sku-1,15,2024-01-01T00:00:00,\nGOLD,sku-2,20,2024-01-01T00:00:00,\n""",
                "orders.csv": """order_id,customer_id,order_ts,sku,gross_amount\n1,CUST-A,2024-02-10T10:00:00,sku-1,100\n2,cust-a,2024-02-20T10:00:00, sku-1 ,100\n3,cust-b,2024-02-20T12:00:00,sku-1,200\n4, cust-b ,2024-02-20T14:00:00,sku-2,50\n""",
            },
        },
        "hidden_cases": [
            {
                "name": "shadow_case_rule_override",
                "files": {
                    "customer_history.csv": """customer_id,tier,region,valid_from,valid_to\ncust-x,bronze,apac,2024-01-01T00:00:00,2024-01-20T00:00:00\ncust-x,silver,apac,2024-01-20T00:00:00,\ncust-y,gold,eu,2024-01-01T00:00:00,\n""",
                    "discount_rules.csv": """tier,sku,discount_pct,valid_from,valid_to\nBRONZE,sku-9,5,2024-01-01T00:00:00,\nSILVER,sku-9,10,2024-01-01T00:00:00,2024-02-01T00:00:00\nSILVER,sku-9,15,2024-02-01T00:00:00,\nGOLD,sku-8,20,2024-01-01T00:00:00,\n""",
                    "orders.csv": """order_id,customer_id,order_ts,sku,gross_amount\n10,cust-x,2024-01-15T08:00:00,sku-9,100\n11,cust-x,2024-02-02T08:00:00,sku-9,100\n12,cust-y,2024-02-05T08:00:00,sku-8,200\n""",
                },
            },
            {
                "name": "shadow_case_overlapping_history",
                "files": {
                    "customer_history.csv": """customer_id,tier,region,valid_from,valid_to\ncust-z,silver,us,2024-01-01T00:00:00,2024-03-01T00:00:00\ncust-z,gold,us,2024-02-15T00:00:00,\ncust-w,bronze,latam,2024-01-01T00:00:00,\n""",
                    "discount_rules.csv": """tier,sku,discount_pct,valid_from,valid_to\nSILVER,sku-x,5,2024-01-01T00:00:00,\nGOLD,sku-x,25,2024-01-01T00:00:00,\nBRONZE,sku-y,10,2024-01-01T00:00:00,\n""",
                    "orders.csv": """order_id,customer_id,order_ts,sku,gross_amount\n20,cust-z,2024-02-20T09:00:00,sku-x,100\n21,cust-w,2024-02-20T10:00:00,sku-y,60\n""",
                },
            },
        ],
    },
    {
        "id": 1,
        "name": "Customer CDC Merge",
        "description": "Apply retry-deduplicated CDC events with partial updates, deletes, and re-inserts to build the final customer master.",
        "instructions": (
            "Read `snapshot_customers.csv` and `customer_cdc.csv`. Normalize `customer_id` and `email` by trimming whitespace and lowercasing. Normalize `status` "
            "by trimming whitespace and lowercasing. In the CDC stream, rows can be retried: if multiple rows share the same `event_id`, keep only the row with the "
            "highest `seq` (breaking any remaining ties by latest `event_ts`). After deduplication, apply events ordered by `event_ts`, `seq`, `event_id`. `DELETE` removes "
            "the customer entirely. `INSERT`, `UPDATE`, and `UPSERT` all behave as partial upserts: only non-empty fields overwrite the current record; empty strings keep the "
            "previous value. For new customers, missing text fields become empty strings and missing `loyalty_points` becomes `0`. Write `output.csv` with columns `customer_id`, "
            "`name`, `email`, `status`, `loyalty_points`, sorted by `customer_id`."
        ),
        "entrypoint": "pipeline.py",
        "output_file": "output.csv",
        "input_files": ["snapshot_customers.csv", "customer_cdc.csv"],
        "sort_by": ["customer_id"],
        "ref_time": 1.0,
        "timeout": 20,
        "transform": transform_customer_cdc_merge,
        "public_case": {
            "name": "public",
            "files": {
                "snapshot_customers.csv": """customer_id,name,email,status,loyalty_points\nc1,Alice,a@example.com,active,10\nc2,Bob,b@example.com,active,20\n""",
                "customer_cdc.csv": """event_id,event_ts,seq,customer_id,op,name,email,status,loyalty_points\ne1,2024-01-02T10:00:00,1,c1,UPDATE,,,vip,15\ne2,2024-01-02T11:00:00,1,c3,INSERT,Cara,c@example.com,active,5\ne3,2024-01-02T12:00:00,1,c2,DELETE,,,,\ne4,2024-01-02T13:00:00,1,c3,UPDATE,,C+old@example.com,,7\ne4,2024-01-02T13:00:01,2,c3,UPDATE,, C+NEW@example.com ,,8\ne5,2024-01-02T14:00:00,1,c2,INSERT,Bobby,b2@example.com,active,1\n""",
            },
        },
        "hidden_cases": [
            {
                "name": "shadow_case_delete_reinsert",
                "files": {
                    "snapshot_customers.csv": """customer_id,name,email,status,loyalty_points\nu1,Nina,n@example.com,active,3\nu2,Omar,o@example.com,active,9\n""",
                    "customer_cdc.csv": """event_id,event_ts,seq,customer_id,op,name,email,status,loyalty_points\nr1,2024-02-01T09:00:00,1,u1,UPDATE,,new-n@example.com,,\nr1,2024-02-01T09:00:01,2,u1,UPDATE,, newer-n@example.com ,,5\nr2,2024-02-01T10:00:00,1,u2,DELETE,,,,\nr3,2024-02-01T11:00:00,1,u2,INSERT,Omar 2,,reactivated,\nr4,2024-02-01T12:00:00,1,u3,UPSERT,Pia,p@example.com,active,4\n""",
                },
            },
            {
                "name": "shadow_case_partial_fields",
                "files": {
                    "snapshot_customers.csv": """customer_id,name,email,status,loyalty_points\nv1,Quinn,q@example.com,active,11\n""",
                    "customer_cdc.csv": """event_id,event_ts,seq,customer_id,op,name,email,status,loyalty_points\ns1,2024-02-10T08:00:00,1,v1,UPDATE,Quinn Jr.,,,\ns2,2024-02-10T09:00:00,1,v2,INSERT,,v2@example.com,active,\ns3,2024-02-10T10:00:00,1,v2,UPDATE,Rhea,,vip,2\n""",
                },
            },
        ],
    },
    {
        "id": 2,
        "name": "Payment Reconciliation",
        "description": "Reconcile captured payments against refunds and lost chargebacks across currencies using event-date FX conversion.",
        "instructions": (
            "Read `payments.csv`, `refunds.csv`, `chargebacks.csv`, and `fx_rates.csv`. Normalize `merchant_id` and currencies by trimming whitespace and uppercasing; "
            "normalize statuses by trimming whitespace and lowercasing. Consider only payments whose status is `captured`. Convert each captured payment amount to USD using the "
            "FX rate for the payment's capture date (`captured_ts` date part) and `from_currency -> USD`. Refunds count only when status is `succeeded`; chargebacks count only "
            "when status is `lost`. Refunds and chargebacks should be joined to their original captured payment to inherit the merchant and currency, then converted to USD using "
            "their own event dates (`refund_ts` / `chargeback_ts`). Write `output.csv` with columns `merchant_id`, `captured_payments`, `refund_events`, `lost_chargebacks`, "
            "`net_usd`, sorted by `merchant_id`, where `net_usd = payment_usd - refund_usd - chargeback_usd` rounded to 2 decimals."
        ),
        "entrypoint": "pipeline.py",
        "output_file": "output.csv",
        "input_files": ["payments.csv", "refunds.csv", "chargebacks.csv", "fx_rates.csv"],
        "sort_by": ["merchant_id"],
        "ref_time": 1.1,
        "timeout": 20,
        "transform": transform_payment_reconciliation,
        "public_case": {
            "name": "public",
            "files": {
                "payments.csv": """payment_id,merchant_id,currency,captured_ts,status,amount\np1,m1,EUR,2024-02-01T10:00:00,captured,100\np2,m1,USD,2024-02-01T11:00:00,failed,50\np3,m2,GBP,2024-02-03T09:00:00,captured,80\np4,m1,USD,2024-02-02T09:00:00,captured,40\n""",
                "refunds.csv": """refund_id,payment_id,refund_ts,status,amount\nr1,p1,2024-02-02T10:00:00,succeeded,20\nr2,p4,2024-02-03T11:00:00,failed,10\n""",
                "chargebacks.csv": """case_id,payment_id,chargeback_ts,status,amount\ncb1,p3,2024-02-04T10:00:00,lost,30\ncb2,p4,2024-02-04T12:00:00,won,5\n""",
                "fx_rates.csv": """rate_date,from_currency,to_currency,rate\n2024-02-01,EUR,USD,1.10\n2024-02-01,USD,USD,1.00\n2024-02-02,EUR,USD,1.20\n2024-02-02,USD,USD,1.00\n2024-02-03,GBP,USD,1.30\n2024-02-03,USD,USD,1.00\n2024-02-04,GBP,USD,1.25\n2024-02-04,USD,USD,1.00\n""",
            },
        },
        "hidden_cases": [
            {
                "name": "shadow_case_multi_refund",
                "files": {
                    "payments.csv": """payment_id,merchant_id,currency,captured_ts,status,amount\np10,m9,JPY,2024-03-01T09:00:00,captured,10000\np11,m9,USD,2024-03-01T10:00:00,captured,20\np12,m8,EUR,2024-03-02T11:00:00,captured,50\n""",
                    "refunds.csv": """refund_id,payment_id,refund_ts,status,amount\nr10,p10,2024-03-02T09:00:00,succeeded,1000\nr11,p10,2024-03-03T09:00:00,succeeded,500\nr12,p12,2024-03-03T10:00:00,failed,10\n""",
                    "chargebacks.csv": """case_id,payment_id,chargeback_ts,status,amount\ncb10,p12,2024-03-04T12:00:00,lost,15\n""",
                    "fx_rates.csv": """rate_date,from_currency,to_currency,rate\n2024-03-01,JPY,USD,0.0090\n2024-03-01,USD,USD,1.00\n2024-03-02,EUR,USD,1.15\n2024-03-02,JPY,USD,0.0091\n2024-03-03,JPY,USD,0.0092\n2024-03-03,EUR,USD,1.16\n2024-03-04,EUR,USD,1.14\n2024-03-04,USD,USD,1.00\n""",
                },
            },
            {
                "name": "shadow_case_ignored_unmatched",
                "files": {
                    "payments.csv": """payment_id,merchant_id,currency,captured_ts,status,amount\np20,mx,USD,2024-04-01T08:00:00,captured,30\np21,my,GBP,2024-04-01T09:00:00,pending,40\n""",
                    "refunds.csv": """refund_id,payment_id,refund_ts,status,amount\nr20,p20,2024-04-01T12:00:00,succeeded,5\nr21,p21,2024-04-01T13:00:00,succeeded,10\n""",
                    "chargebacks.csv": """case_id,payment_id,chargeback_ts,status,amount\ncb20,p20,2024-04-02T10:00:00,won,10\ncb21,p99,2024-04-02T11:00:00,lost,7\n""",
                    "fx_rates.csv": """rate_date,from_currency,to_currency,rate\n2024-04-01,USD,USD,1.00\n2024-04-01,GBP,USD,1.28\n2024-04-02,USD,USD,1.00\n2024-04-02,GBP,USD,1.27\n""",
                },
            },
        ],
    },
    {
        "id": 3,
        "name": "Session Funnel With Identity Stitching",
        "description": "Resolve temporal identity mappings, sessionize user behavior, and compute conversion-funnel revenue.",
        "instructions": (
            "Read `identity_map.csv` and `events.csv`. Normalize `identity_type`, `identity_value`, `user_id`, and `event_type` by trimming whitespace and lowercasing. "
            "Use `identity_map.csv` to resolve each event to a user: match rows where identity type and value match, and `valid_from <= event_ts < valid_to`, treating blank "
            "`valid_to` as open-ended; if multiple rows match, keep the one with the latest `valid_from`. Drop events that cannot be resolved to a user. For each user, sort "
            "events by `event_ts`, then `event_id`, and create sessions where a new session starts only when the gap from the previous event is strictly greater than 30 minutes. "
            "A session is converting only if it contains at least one `view_product`, one `add_to_cart`, and one `purchase`. Revenue is the sum of `order_value` for purchase events "
            "inside converting sessions only. Write `output.csv` with columns `user_id`, `sessions`, `converting_sessions`, `revenue`, sorted by `user_id`."
        ),
        "entrypoint": "pipeline.py",
        "output_file": "output.csv",
        "input_files": ["identity_map.csv", "events.csv"],
        "sort_by": ["user_id"],
        "ref_time": 1.1,
        "timeout": 20,
        "transform": transform_session_funnel,
        "public_case": {
            "name": "public",
            "files": {
                "identity_map.csv": """identity_type,identity_value,user_id,valid_from,valid_to\ncookie,c-1,u1,2024-03-01T00:00:00,\ndevice,d-9,u1,2024-03-01T00:00:00,\ncookie,c-2,u2,2024-03-01T00:00:00,\n""",
                "events.csv": """event_id,event_ts,identity_type,identity_value,event_type,order_value\n1,2024-03-01T10:00:00,cookie,c-1,view_product,0\n2,2024-03-01T10:10:00,device,d-9,add_to_cart,0\n3,2024-03-01T10:20:00,cookie,c-1,purchase,120\n4,2024-03-01T12:00:00,cookie,c-2,view_product,0\n5,2024-03-01T12:40:00,cookie,c-2,add_to_cart,0\n6,2024-03-01T12:50:00,cookie,c-2,purchase,50\n7,2024-03-01T13:00:00,cookie,missing,view_product,0\n""",
            },
        },
        "hidden_cases": [
            {
                "name": "shadow_case_identity_rebind",
                "files": {
                    "identity_map.csv": """identity_type,identity_value,user_id,valid_from,valid_to\ncookie,c-9,u1,2024-04-01T00:00:00,2024-04-02T00:00:00\ncookie,c-9,u3,2024-04-02T00:00:00,\ndevice,d-1,u3,2024-04-02T00:00:00,\n""",
                    "events.csv": """event_id,event_ts,identity_type,identity_value,event_type,order_value\n10,2024-04-01T10:00:00,cookie,c-9,view_product,0\n11,2024-04-01T10:05:00,cookie,c-9,add_to_cart,0\n12,2024-04-01T10:20:00,cookie,c-9,purchase,40\n13,2024-04-02T09:00:00,cookie,c-9,view_product,0\n14,2024-04-02T09:20:00,device,d-1,add_to_cart,0\n15,2024-04-02T09:25:00,cookie,c-9,purchase,70\n""",
                },
            },
            {
                "name": "shadow_case_gap_boundary",
                "files": {
                    "identity_map.csv": """identity_type,identity_value,user_id,valid_from,valid_to\nemail,e-1,u8,2024-05-01T00:00:00,\n""",
                    "events.csv": """event_id,event_ts,identity_type,identity_value,event_type,order_value\n20,2024-05-01T09:00:00,email,e-1,view_product,0\n21,2024-05-01T09:30:00,email,e-1,add_to_cart,0\n22,2024-05-01T09:59:00,email,e-1,purchase,30\n23,2024-05-01T10:31:00,email,e-1,view_product,0\n24,2024-05-01T10:40:00,email,e-1,purchase,20\n""",
                },
            },
        ],
    },
    {
        "id": 4,
        "name": "BOM Fulfillment Simulation",
        "description": "Simulate warehouse-level component allocation with receipts, priorities, and shared BOM contention.",
        "instructions": (
            "Read `inventory_snapshots.csv`, `receipts.csv`, `bom.csv`, and `orders.csv`. Normalize `component_id` and `sku` by trimming whitespace and lowercasing. "
            "Normalize `warehouse_id` by trimming whitespace and uppercasing. Start from the inventory snapshot. Process receipts in chronological order; a receipt becomes "
            "available only for orders whose `order_ts` is at or after the receipt's `arrival_ts`. Process orders sorted by `order_ts`, then `priority` ascending (smaller value "
            "means higher priority), then `order_id`. For each order, check whether all required BOM components are available in the order's warehouse for the requested quantity. "
            "If the order can be fulfilled, allocate and deduct all required components immediately. If not, do not allocate anything for that order. Write `output.csv` with columns "
            "`order_id`, `warehouse_id`, `fulfilled`, `missing_components`, sorted by `order_id`, where `fulfilled` is the lowercase string `true` or `false` and `missing_components` "
            "is a semicolon-joined sorted list of the missing component ids or an empty string when fulfilled."
        ),
        "entrypoint": "pipeline.py",
        "output_file": "output.csv",
        "input_files": ["inventory_snapshots.csv", "receipts.csv", "bom.csv", "orders.csv"],
        "sort_by": ["order_id"],
        "ref_time": 1.2,
        "timeout": 20,
        "transform": transform_bom_fulfillment,
        "public_case": {
            "name": "public",
            "files": {
                "inventory_snapshots.csv": """component_id,warehouse_id,on_hand\ncomp-a,wh1,10\ncomp-b,wh1,6\ncomp-a,wh2,5\ncomp-c,wh2,8\n""",
                "receipts.csv": """component_id,warehouse_id,arrival_ts,qty\ncomp-b,wh1,2024-04-01T11:00:00,4\ncomp-a,wh2,2024-04-01T10:30:00,2\n""",
                "bom.csv": """sku,component_id,units_per_sku\nsku-1,comp-a,2\nsku-1,comp-b,1\nsku-2,comp-a,1\nsku-2,comp-c,2\n""",
                "orders.csv": """order_id,warehouse_id,order_ts,sku,qty,priority\n1,wh1,2024-04-01T10:00:00,sku-1,3,2\n2,wh1,2024-04-01T10:30:00,sku-1,2,1\n3,wh1,2024-04-01T10:45:00,sku-2,1,1\n4,wh2,2024-04-01T10:40:00,sku-2,3,1\n""",
            },
        },
        "hidden_cases": [
            {
                "name": "shadow_case_priority_contention",
                "files": {
                    "inventory_snapshots.csv": """component_id,warehouse_id,on_hand\ncomp-x,wh3,6\ncomp-y,wh3,3\n""",
                    "receipts.csv": """component_id,warehouse_id,arrival_ts,qty\ncomp-y,wh3,2024-06-01T11:00:00,2\n""",
                    "bom.csv": """sku,component_id,units_per_sku\nsku-a,comp-x,2\nsku-a,comp-y,1\n""",
                    "orders.csv": """order_id,warehouse_id,order_ts,sku,qty,priority\n10,wh3,2024-06-01T10:00:00,sku-a,2,2\n11,wh3,2024-06-01T10:00:00,sku-a,1,1\n12,wh3,2024-06-01T11:30:00,sku-a,1,1\n""",
                },
            },
            {
                "name": "shadow_case_late_receipt",
                "files": {
                    "inventory_snapshots.csv": """component_id,warehouse_id,on_hand\ncomp-m,wh4,2\ncomp-n,wh4,2\n""",
                    "receipts.csv": """component_id,warehouse_id,arrival_ts,qty\ncomp-m,wh4,2024-07-01T09:15:00,2\ncomp-n,wh4,2024-07-01T09:45:00,2\n""",
                    "bom.csv": """sku,component_id,units_per_sku\nsku-b,comp-m,2\nsku-b,comp-n,2\n""",
                    "orders.csv": """order_id,warehouse_id,order_ts,sku,qty,priority\n20,wh4,2024-07-01T09:00:00,sku-b,1,1\n21,wh4,2024-07-01T09:30:00,sku-b,1,1\n22,wh4,2024-07-01T10:00:00,sku-b,1,1\n""",
                },
            },
        ],
    },
]


for task in TASKS:
    task["files"] = dict(task["public_case"]["files"])
    task["files"]["pipeline.py"] = starter_pipeline()
    task["files"]["task.md"] = build_task_file(task["name"], task["instructions"])
    task["grader"] = lambda workspace_path, logs=None, last_exec_time=None, task=task: grade_submission(
        task,
        workspace_path,
        logs,
        last_exec_time,
    )
