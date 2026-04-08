import os
import asyncio
import json
import re
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")


# -------------------------
# Safe Env API
# -------------------------
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
    except:
        return {}


# -------------------------
# Logging (STRICT FORMAT)
# -------------------------
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


# -------------------------
# Main Agent
# -------------------------
async def main():
    use_llm = API_BASE_URL and MODEL_NAME and HF_TOKEN

    if use_llm:
        client = AsyncOpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    rewards = []
    total_reward = 0.0
    max_steps = 10
    steps_taken = 0
    success = False

    log_start(
        task="data-pipeline-task-0",
        env="datapipe-sandbox-v1",
        model=MODEL_NAME
    )

    try:
        obs = await call_env_api("reset", {"task_id": 0})

        for step in range(1, max_steps + 1):
            steps_taken = step

            # -------------------------
            # STRICT SAFE STRATEGY
            # -------------------------
            if step == 1:
                action = {
                    "command": "write",
                    "path": "solution.py",
                    "content": "print('pipeline executed')"
                }
            elif step in [2, 3]:
                action = {"command": "run"}
            else:
                action = {"command": "submit"}

            # -------------------------
            # OPTIONAL LLM (STRICTLY CONTROLLED)
            # -------------------------
            if use_llm:
                try:
                    prompt = f"""
State:
Files: {obs.get('files')}
Logs: {obs.get('logs')}
Metrics: {obs.get('metrics')}

Allowed commands: write, run, submit

Return ONLY JSON:
{{"command": "..."}}
"""
                    messages = [{"role": "user", "content": prompt}]

                    response = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=messages,
                            temperature=0
                        ),
                        timeout=5
                    )

                    content = response.choices[0].message.content

                    match = re.search(r"\{.*\}", content, re.DOTALL)
                    if match:
                        llm_action = json.loads(match.group(0))
                        cmd = llm_action.get("command", "run")

                        # ✅ STRICT CONTROL
                        if cmd == "submit" and step >= 3:
                            action = {"command": "submit"}
                        elif cmd == "run":
                            action = {"command": "run"}
                        # ❌ write NOT allowed after step 1

                except:
                    pass

            # -------------------------
            # Execute Step
            # -------------------------
            result = await call_env_api("step", action)

            if not result:
                break

            obs = result.get("observation", {})
            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            error = result.get("error", None)

            total_reward += reward
            rewards.append(reward)

            log_step(step, action.get("command", "run"), reward, done, error)

            if done:
                break

        # Normalize score safely
        score = max(0.0, min(1.0, total_reward))
        success = score > 0.7

    finally:
        log_end(success, steps_taken, score if 'score' in locals() else 0.0, rewards)


# -------------------------
# Entry Point
# -------------------------
if __name__ == "__main__":
    asyncio.run(main())