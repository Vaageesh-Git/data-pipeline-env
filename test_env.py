import textwrap

from env import DataPipelineEnv
from models import Action


SOLUTIONS = {
    0: textwrap.dedent(
        """
        import pandas as pd


        def ntext(value):
            if pd.isna(value):
                return ""
            return str(value).strip()


        def nlower(value):
            return ntext(value).lower()


        def nupper(value):
            return ntext(value).upper()


        def parse_optional(series):
            values = series.astype(str).str.strip().replace({"": pd.NA})
            return pd.to_datetime(values, errors="coerce")


        def pick_interval(frame, filters, ts):
            scoped = frame.copy()
            for column, value in filters.items():
                scoped = scoped[scoped[column] == value]
            scoped = scoped[(scoped["valid_from"] <= ts) & (scoped["valid_to"].isna() | (ts < scoped["valid_to"]))]
            if scoped.empty:
                return None
            return scoped.sort_values(["valid_from", "valid_to"], ascending=[False, False], na_position="last").iloc[0]


        def main():
            history = pd.read_csv("customer_history.csv")
            rules = pd.read_csv("discount_rules.csv")
            orders = pd.read_csv("orders.csv")

            history["customer_id"] = history["customer_id"].map(nlower)
            history["tier"] = history["tier"].map(nupper)
            history["region"] = history["region"].map(nupper)
            history["valid_from"] = pd.to_datetime(history["valid_from"], errors="coerce")
            history["valid_to"] = parse_optional(history["valid_to"])

            rules["tier"] = rules["tier"].map(nupper)
            rules["sku"] = rules["sku"].map(nlower)
            rules["discount_pct"] = pd.to_numeric(rules["discount_pct"], errors="coerce").fillna(0.0)
            rules["valid_from"] = pd.to_datetime(rules["valid_from"], errors="coerce")
            rules["valid_to"] = parse_optional(rules["valid_to"])

            orders["customer_id"] = orders["customer_id"].map(nlower)
            orders["sku"] = orders["sku"].map(nlower)
            orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
            orders["gross_amount"] = pd.to_numeric(orders["gross_amount"], errors="coerce").fillna(0.0)

            rows = []
            for order in orders.itertuples(index=False):
                customer = pick_interval(history, {"customer_id": order.customer_id}, order.order_ts)
                if customer is None:
                    continue
                rule = pick_interval(rules, {"tier": customer["tier"], "sku": order.sku}, order.order_ts)
                discount_pct = float(rule["discount_pct"]) if rule is not None else 0.0
                rows.append(
                    {
                        "region": customer["region"],
                        "tier": customer["tier"],
                        "order_count": 1,
                        "net_revenue": round(float(order.gross_amount) * (1.0 - (discount_pct / 100.0)), 2),
                    }
                )

            output = pd.DataFrame(rows)
            output = output.groupby(["region", "tier"], as_index=False).agg(order_count=("order_count", "sum"), net_revenue=("net_revenue", "sum"))
            output["net_revenue"] = output["net_revenue"].map(lambda value: round(float(value), 2))
            output = output.sort_values(["region", "tier"])
            output.to_csv("output.csv", index=False)


        if __name__ == "__main__":
            main()
        """
    ).strip()
    + "\n",
    1: textwrap.dedent(
        """
        import pandas as pd


        def ntext(value):
            if pd.isna(value):
                return ""
            return str(value).strip()


        def nlower(value):
            return ntext(value).lower()


        def apply_update(existing, row):
            record = dict(existing) if existing else {
                "customer_id": nlower(row["customer_id"]),
                "name": "",
                "email": "",
                "status": "",
                "loyalty_points": 0,
            }
            name = ntext(row.get("name", ""))
            email = nlower(row.get("email", ""))
            status = nlower(row.get("status", ""))
            points = ntext(row.get("loyalty_points", ""))
            if name:
                record["name"] = name
            if email:
                record["email"] = email
            if status:
                record["status"] = status
            if points:
                record["loyalty_points"] = int(float(points))
            return record


        def main():
            snapshot = pd.read_csv("snapshot_customers.csv")
            cdc = pd.read_csv("customer_cdc.csv")

            snapshot["customer_id"] = snapshot["customer_id"].map(nlower)
            snapshot["email"] = snapshot["email"].map(nlower)
            snapshot["status"] = snapshot["status"].map(nlower)
            snapshot["loyalty_points"] = pd.to_numeric(snapshot["loyalty_points"], errors="coerce").fillna(0).astype(int)

            state = {
                row.customer_id: {
                    "customer_id": row.customer_id,
                    "name": ntext(row.name),
                    "email": row.email,
                    "status": row.status,
                    "loyalty_points": int(row.loyalty_points),
                }
                for row in snapshot.itertuples(index=False)
            }

            cdc["customer_id"] = cdc["customer_id"].map(nlower)
            cdc["op"] = cdc["op"].map(nlower)
            cdc["event_ts"] = pd.to_datetime(cdc["event_ts"], errors="coerce")
            cdc["seq"] = pd.to_numeric(cdc["seq"], errors="coerce").fillna(0).astype(int)
            cdc = cdc.sort_values(["event_id", "seq", "event_ts"], kind="mergesort")
            cdc = cdc.drop_duplicates(subset=["event_id"], keep="last")
            cdc = cdc.sort_values(["event_ts", "seq", "event_id"], kind="mergesort")

            for row in cdc.to_dict("records"):
                customer_id = row["customer_id"]
                if row["op"] == "delete":
                    state.pop(customer_id, None)
                    continue
                state[customer_id] = apply_update(state.get(customer_id), row)

            output = pd.DataFrame(list(state.values()))
            output = output[["customer_id", "name", "email", "status", "loyalty_points"]].sort_values("customer_id")
            output.to_csv("output.csv", index=False)


        if __name__ == "__main__":
            main()
        """
    ).strip()
    + "\n",
    2: textwrap.dedent(
        """
        import pandas as pd


        def ntext(value):
            if pd.isna(value):
                return ""
            return str(value).strip()


        def nlower(value):
            return ntext(value).lower()


        def nupper(value):
            return ntext(value).upper()


        def build_fx_lookup(frame):
            frame = frame.copy()
            frame["rate_date"] = pd.to_datetime(frame["rate_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            frame["from_currency"] = frame["from_currency"].map(nupper)
            frame["to_currency"] = frame["to_currency"].map(nupper)
            frame["rate"] = pd.to_numeric(frame["rate"], errors="coerce").fillna(0.0)
            return {
                (row.rate_date, row.from_currency): float(row.rate)
                for row in frame.itertuples(index=False)
                if row.to_currency == "USD"
            }


        def convert_to_usd(amount, currency, timestamp, lookup):
            currency = nupper(currency)
            if currency == "USD":
                return float(amount)
            date_key = pd.to_datetime(timestamp, errors="coerce").strftime("%Y-%m-%d")
            rate = lookup.get((date_key, currency), 0.0)
            return float(amount) * rate


        def main():
            payments = pd.read_csv("payments.csv")
            refunds = pd.read_csv("refunds.csv")
            chargebacks = pd.read_csv("chargebacks.csv")
            lookup = build_fx_lookup(pd.read_csv("fx_rates.csv"))

            payments["payment_id"] = payments["payment_id"].map(nlower)
            payments["merchant_id"] = payments["merchant_id"].map(nupper)
            payments["currency"] = payments["currency"].map(nupper)
            payments["status"] = payments["status"].map(nlower)
            payments["captured_ts"] = pd.to_datetime(payments["captured_ts"], errors="coerce")
            payments["amount"] = pd.to_numeric(payments["amount"], errors="coerce").fillna(0.0)
            captured = payments[payments["status"] == "captured"].copy()
            captured["payment_usd"] = [
                convert_to_usd(row.amount, row.currency, row.captured_ts, lookup)
                for row in captured.itertuples(index=False)
            ]

            payment_lookup = captured.set_index("payment_id")[["merchant_id", "currency"]].to_dict("index")

            refunds["payment_id"] = refunds["payment_id"].map(nlower)
            refunds["status"] = refunds["status"].map(nlower)
            refunds["refund_ts"] = pd.to_datetime(refunds["refund_ts"], errors="coerce")
            refunds["amount"] = pd.to_numeric(refunds["amount"], errors="coerce").fillna(0.0)
            refunds = refunds[(refunds["status"] == "succeeded") & (refunds["payment_id"].isin(payment_lookup))].copy()
            refunds["merchant_id"] = refunds["payment_id"].map(lambda payment_id: payment_lookup[payment_id]["merchant_id"])
            refunds["currency"] = refunds["payment_id"].map(lambda payment_id: payment_lookup[payment_id]["currency"])
            refunds["refund_usd"] = [
                convert_to_usd(row.amount, row.currency, row.refund_ts, lookup)
                for row in refunds.itertuples(index=False)
            ]

            chargebacks["payment_id"] = chargebacks["payment_id"].map(nlower)
            chargebacks["status"] = chargebacks["status"].map(nlower)
            chargebacks["chargeback_ts"] = pd.to_datetime(chargebacks["chargeback_ts"], errors="coerce")
            chargebacks["amount"] = pd.to_numeric(chargebacks["amount"], errors="coerce").fillna(0.0)
            chargebacks = chargebacks[(chargebacks["status"] == "lost") & (chargebacks["payment_id"].isin(payment_lookup))].copy()
            chargebacks["merchant_id"] = chargebacks["payment_id"].map(lambda payment_id: payment_lookup[payment_id]["merchant_id"])
            chargebacks["currency"] = chargebacks["payment_id"].map(lambda payment_id: payment_lookup[payment_id]["currency"])
            chargebacks["chargeback_usd"] = [
                convert_to_usd(row.amount, row.currency, row.chargeback_ts, lookup)
                for row in chargebacks.itertuples(index=False)
            ]

            payment_summary = captured.groupby("merchant_id", as_index=False).agg(captured_payments=("payment_id", "count"), payment_usd=("payment_usd", "sum"))
            refund_summary = refunds.groupby("merchant_id", as_index=False).agg(refund_events=("refund_id", "count"), refund_usd=("refund_usd", "sum"))
            chargeback_summary = chargebacks.groupby("merchant_id", as_index=False).agg(lost_chargebacks=("case_id", "count"), chargeback_usd=("chargeback_usd", "sum"))

            output = payment_summary.merge(refund_summary, on="merchant_id", how="left")
            output = output.merge(chargeback_summary, on="merchant_id", how="left")
            output = output.fillna({"refund_events": 0, "refund_usd": 0.0, "lost_chargebacks": 0, "chargeback_usd": 0.0})
            output["net_usd"] = output["payment_usd"] - output["refund_usd"] - output["chargeback_usd"]
            output = output[["merchant_id", "captured_payments", "refund_events", "lost_chargebacks", "net_usd"]]
            output["captured_payments"] = output["captured_payments"].astype(int)
            output["refund_events"] = output["refund_events"].astype(int)
            output["lost_chargebacks"] = output["lost_chargebacks"].astype(int)
            output["net_usd"] = output["net_usd"].map(lambda value: round(float(value), 2))
            output = output.sort_values("merchant_id")
            output.to_csv("output.csv", index=False)


        if __name__ == "__main__":
            main()
        """
    ).strip()
    + "\n",
    3: textwrap.dedent(
        """
        import pandas as pd


        def ntext(value):
            if pd.isna(value):
                return ""
            return str(value).strip()


        def nlower(value):
            return ntext(value).lower()


        def parse_optional(series):
            values = series.astype(str).str.strip().replace({"": pd.NA})
            return pd.to_datetime(values, errors="coerce")


        def pick_interval(frame, filters, ts):
            scoped = frame.copy()
            for column, value in filters.items():
                scoped = scoped[scoped[column] == value]
            scoped = scoped[(scoped["valid_from"] <= ts) & (scoped["valid_to"].isna() | (ts < scoped["valid_to"]))]
            if scoped.empty:
                return None
            return scoped.sort_values(["valid_from", "valid_to"], ascending=[False, False], na_position="last").iloc[0]


        def main():
            identity_map = pd.read_csv("identity_map.csv")
            events = pd.read_csv("events.csv")

            identity_map["identity_type"] = identity_map["identity_type"].map(nlower)
            identity_map["identity_value"] = identity_map["identity_value"].map(nlower)
            identity_map["user_id"] = identity_map["user_id"].map(nlower)
            identity_map["valid_from"] = pd.to_datetime(identity_map["valid_from"], errors="coerce")
            identity_map["valid_to"] = parse_optional(identity_map["valid_to"])

            events["identity_type"] = events["identity_type"].map(nlower)
            events["identity_value"] = events["identity_value"].map(nlower)
            events["event_type"] = events["event_type"].map(nlower)
            events["event_ts"] = pd.to_datetime(events["event_ts"], errors="coerce")
            events["order_value"] = pd.to_numeric(events["order_value"], errors="coerce").fillna(0.0)

            resolved_rows = []
            for event in events.itertuples(index=False):
                mapping = pick_interval(identity_map, {"identity_type": event.identity_type, "identity_value": event.identity_value}, event.event_ts)
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

            resolved = pd.DataFrame(resolved_rows)
            resolved = resolved.sort_values(["user_id", "event_ts", "event_id"], kind="mergesort")

            session_rows = []
            for user_id, user_events in resolved.groupby("user_id", sort=False):
                session_id = 0
                previous_ts = None
                for row in user_events.itertuples(index=False):
                    if previous_ts is None or (row.event_ts - previous_ts).total_seconds() > 1800:
                        session_id += 1
                    session_rows.append(
                        {
                            "user_id": user_id,
                            "session_id": session_id,
                            "event_type": row.event_type,
                            "order_value": row.order_value,
                        }
                    )
                    previous_ts = row.event_ts

            sessions = pd.DataFrame(session_rows)
            output_rows = []
            for user_id, user_sessions in sessions.groupby("user_id"):
                session_count = user_sessions["session_id"].nunique()
                converting = 0
                revenue = 0.0
                for _, session_frame in user_sessions.groupby("session_id"):
                    event_types = set(session_frame["event_type"].tolist())
                    if {"view_product", "add_to_cart", "purchase"}.issubset(event_types):
                        converting += 1
                        revenue += session_frame.loc[session_frame["event_type"] == "purchase", "order_value"].sum()
                output_rows.append({
                    "user_id": user_id,
                    "sessions": int(session_count),
                    "converting_sessions": int(converting),
                    "revenue": round(float(revenue), 2),
                })

            output = pd.DataFrame(output_rows).sort_values("user_id")
            output.to_csv("output.csv", index=False)


        if __name__ == "__main__":
            main()
        """
    ).strip()
    + "\n",
    4: textwrap.dedent(
        """
        import pandas as pd


        def ntext(value):
            if pd.isna(value):
                return ""
            return str(value).strip()


        def nlower(value):
            return ntext(value).lower()


        def nupper(value):
            return ntext(value).upper()


        def main():
            inventory = pd.read_csv("inventory_snapshots.csv")
            receipts = pd.read_csv("receipts.csv")
            bom = pd.read_csv("bom.csv")
            orders = pd.read_csv("orders.csv")

            inventory["component_id"] = inventory["component_id"].map(nlower)
            inventory["warehouse_id"] = inventory["warehouse_id"].map(nupper)
            inventory["on_hand"] = pd.to_numeric(inventory["on_hand"], errors="coerce").fillna(0.0)

            receipts["component_id"] = receipts["component_id"].map(nlower)
            receipts["warehouse_id"] = receipts["warehouse_id"].map(nupper)
            receipts["arrival_ts"] = pd.to_datetime(receipts["arrival_ts"], errors="coerce")
            receipts["qty"] = pd.to_numeric(receipts["qty"], errors="coerce").fillna(0.0)
            receipts = receipts.sort_values(["arrival_ts", "warehouse_id", "component_id"], kind="mergesort").reset_index(drop=True)

            bom["sku"] = bom["sku"].map(nlower)
            bom["component_id"] = bom["component_id"].map(nlower)
            bom["units_per_sku"] = pd.to_numeric(bom["units_per_sku"], errors="coerce").fillna(0.0)

            orders["warehouse_id"] = orders["warehouse_id"].map(nupper)
            orders["sku"] = orders["sku"].map(nlower)
            orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
            orders["qty"] = pd.to_numeric(orders["qty"], errors="coerce").fillna(0.0)
            orders["priority"] = pd.to_numeric(orders["priority"], errors="coerce").fillna(0).astype(int)
            orders = orders.sort_values(["order_ts", "priority", "order_id"], kind="mergesort").reset_index(drop=True)

            stock = {(row.warehouse_id, row.component_id): float(row.on_hand) for row in inventory.itertuples(index=False)}

            receipt_pointer = 0
            rows = []
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
                    needed = float(requirement.units_per_sku) * float(order.qty)
                    available = stock.get(key, 0.0)
                    allocations.append((key, needed))
                    if available + 1e-9 < needed:
                        shortages.append(requirement.component_id)

                if shortages:
                    rows.append({
                        "order_id": int(order.order_id),
                        "warehouse_id": order.warehouse_id,
                        "fulfilled": "false",
                        "missing_components": ";".join(sorted(shortages)),
                    })
                    continue

                for key, needed in allocations:
                    stock[key] = stock.get(key, 0.0) - needed

                rows.append({
                    "order_id": int(order.order_id),
                    "warehouse_id": order.warehouse_id,
                    "fulfilled": "true",
                    "missing_components": "",
                })

            output = pd.DataFrame(rows).sort_values("order_id")
            output.to_csv("output.csv", index=False)


        if __name__ == "__main__":
            main()
        """
    ).strip()
    + "\n",
}


def main():
    env = DataPipelineEnv()

    for task_id, code in SOLUTIONS.items():
        env.reset(task_id)
        env.step(Action(command="write", path="pipeline.py", content=code))
        _, run_reward, _, _ = env.step(Action(command="run", path=None, content=None))
        _, submit_reward, done, submit_info = env.step(Action(command="submit", path=None, content=None))

        print(f"task={task_id} run_reward={run_reward:.2f} submit_reward={submit_reward:.2f} done={done}")
        print(submit_info["metrics"])

        if submit_info["metrics"]["correctness"] < 0.99:
            raise RuntimeError(f"task {task_id} correctness too low: {submit_info['metrics']}")
        if not done:
            raise RuntimeError(f"task {task_id} did not finish")


if __name__ == "__main__":
    main()
