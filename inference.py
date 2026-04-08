import os
import asyncio
import json
import re
from typing import Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")


# 🔥 GLOBAL SAFE CLAMP FUNCTION
def safe_score(x):
    try:
        x = float(x)
    except:
        return 0.01
    x = max(0.01, min(0.99, x))
    return round(x, 2)


async def call_env_api(endpoint: str, data: Optional[dict] = None):
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            url = f"{ENV_URL}/{endpoint}"
            if endpoint == "reset":
                response = await client.post(url, params=data or {}, timeout=30)
            elif data:
                response = await client.post(url, json=data, timeout=30)
            else:
                response = await client.get(url, timeout=30)
            return response.json()
    except:
        return {}


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True
    )


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True
    )


async def main():
    client = AsyncOpenAI(
        api_key=os.environ["API_KEY"],
        base_url=os.environ["API_BASE_URL"]
    )

    rewards = []
    total_reward = 0.0
    max_steps = 10
    steps_taken = 0
    success = False
    score = 0.01

    log_start(
        task="data-pipeline-task-0",
        env="datapipe-sandbox-v1",
        model=MODEL_NAME
    )

    try:
        # warmup
        try:
            await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": "ping"}],
                temperature=0
            )
        except:
            pass

        obs = await call_env_api("reset", {"task_id": 0})

        for step in range(1, max_steps + 1):
            steps_taken = step

            # default policy
            if step == 1:
                action = {
                    "command": "write",
                    "path": "pipeline.py",
                    "content": "import pandas as pd\n\n\ndef main():\n    raise NotImplementedError('Implement the task pipeline')\n\n\nif __name__ == '__main__':\n    main()\n"
                }
            elif step <= 6:
                action = {"command": "run"}
            else:
                action = {"command": "submit"}

            # optional LLM override
            try:
                prompt = f"""
State:
Files: {obs.get('files')}
Logs: {obs.get('logs')}
Metrics: {obs.get('metrics')}

Allowed commands: write, run, submit
The runnable file is pipeline.py and the required output is output.csv.

Return ONLY JSON:
{{"command": "..."}}
"""
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0
                    ),
                    timeout=5
                )

                content = response.choices[0].message.content or ""
                match = re.search(r"\{.*\}", content, re.DOTALL)

                if match:
                    llm_action = json.loads(match.group(0))
                    cmd = llm_action.get("command", "run")

                    if cmd == "submit" and step >= 4:
                        action = {"command": "submit"}
                    elif cmd == "run":
                        action = {"command": "run"}

            except:
                pass

            result = await call_env_api("step", action)

            if not result:
                break

            obs = result.get("observation", {})
            raw_reward = result.get("reward", 0.0)
            done = result.get("done", False)
            error = result.get("error", None)

            # 🔥 CLAMP EVERY STEP
            reward = safe_score(raw_reward)

            total_reward += reward
            rewards.append(reward)

            log_step(step, action.get("command", "run"), reward, done, error)

            if done:
                break

        # 🔥 FINAL SCORE CLAMP
        score = safe_score(total_reward)

        success = score > 0.3

    finally:
        log_end(
            success,
            steps_taken,
            score,
            rewards
        )


if __name__ == "__main__":
    asyncio.run(main())
