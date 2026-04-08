import os
import asyncio
import json
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")


async def call_env_api(endpoint: str, data: dict = None):
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            url = f"{ENV_URL}/{endpoint}"
            if data:
                response = await client.post(url, json=data, timeout=30)
            else:
                response = await client.get(url, timeout=30)
            return response.json()
    except Exception as e:
        print("ENV ERROR:", str(e))
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
    use_llm = API_BASE_URL and MODEL_NAME and HF_TOKEN

    if use_llm:
        client = AsyncOpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    rewards = []
    total_reward = 0.0
    max_steps = 10

    # Reset environment
    obs = await call_env_api("reset", {"task_id": 0})
    if not obs:
        print("Failed to reset environment")
        return

    log_start(
        task="data-pipeline-task-0",
        env="datapipe-sandbox-v1",
        model=MODEL_NAME
    )

    for step in range(1, max_steps + 1):

        # Default fallback action
        action = {"command": "run"}

        if use_llm:
            prompt = f"""
You are a Data Engineering Agent.

Goal: Maximize FINAL SCORE.

Evaluation Criteria:
- Correctness (50%)
- Efficiency (20%)
- Robustness (30%)

Current State:
Files: {obs.get('files')}
Logs: {obs.get('logs')}
Metrics: {obs.get('metrics')}
Preview: {obs.get('data_preview')}

Available Actions:
1. write → create/update code
2. run → execute pipeline
3. submit → finalize solution

Rules:
- Always return valid JSON
- Avoid unnecessary writes
- Prefer correctness over speed

Output format:
{{"command": "...", "path": "...", "content": "..."}}
"""

            messages = [{"role": "user", "content": prompt}]

            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=0
                    ),
                    timeout=10
                )

                content = response.choices[0].message.content

                try:
                    action = json.loads(content)
                except Exception:
                    print("JSON parse failed, fallback to run")
                    action = {"command": "run"}

            except Exception as e:
                print("LLM ERROR:", str(e))
                action = {"command": "run"}

        # Execute step
        result = await call_env_api("step", action)

        if not result:
            print("Step failed")
            break

        obs = result.get("observation", {})
        reward = result.get("reward", 0.0)
        done = result.get("done", False)
        error = result.get("error", None)

        total_reward += reward
        rewards.append(reward)

        cmd = action.get("command", "unknown")

        log_step(step, cmd, reward, done, error)

        if done:
            break

    log_end(
        success=(total_reward > 0.7),
        steps=step,
        score=total_reward,
        rewards=rewards
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print("FATAL ERROR:", str(e))