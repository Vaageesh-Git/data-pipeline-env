# inference.py
import os
import asyncio
import json
from openai import AsyncOpenAI
from dotenv import load_dotenv  # pip install python-dotenv

# Load local .env file (ignored by git for security)
load_dotenv()

# Mandatory Environment Variables from Problem Statement
API_BASE_URL = os.getenv("API_BASE_URL") 
MODEL_NAME = os.getenv("MODEL_NAME")     # used first groq llama for testing
HF_TOKEN = os.getenv("HF_TOKEN")         # api key

# OpenEnv execution variables
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

async def call_env_api(endpoint: str, data: dict = None):
    import httpx
    async with httpx.AsyncClient() as client:
        url = f"{ENV_URL}/{endpoint}"
        if data:
            response = await client.post(url, json=data, timeout=30)
        else:
            response = await client.get(url, timeout=30)
        return response.json()

def log_start(task):
    print(f"[START] Task: {task}", flush=True)

def log_step(step, action, reward, done):
    print(f"[STEP] Step: {step} | Action: {action} | Reward: {reward} | Done: {done}", flush=True)

def log_end(success, score):
    print(f"[END] Success: {success} | Final Score: {score}", flush=True)

async def main():
    # The client uses HF_TOKEN as the API key as per competition rules
    client = AsyncOpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    
    # 1. Start Environment
    init_data = await call_env_api("reset", {"task_id": 0})
    obs = init_data # In OpenEnv, reset returns the observation
    
    log_start("Data Pipeline Task 0")
    
    total_reward = 0.0
    max_steps = 10

    for step in range(1, max_steps + 1):
        # 2. Agent Reasoning
        prompt = f"""
        You are a Data Engineering Agent.

        Goal: Maximize FINAL SCORE.

        Evaluation Criteria:
        - Correctness (50%) → schema, row count, values
        - Efficiency (20%) → execution time
        - Robustness (30%) → handles dirty data

        Current State:
        Files: {obs.get('files')}
        Logs: {obs.get('logs')}
        Metrics: {obs.get('metrics')}
        Preview: {obs.get('data_preview')}

        Strategy:
        - If correctness < 1 → fix logic
        - If efficiency low → optimize computation
        - If robustness low → handle nulls, types, edge cases

        Available Actions:
        1. write → modify pipeline.py
        2. run → test pipeline
        3. submit → final answer

        Return JSON only:
        {{"command": "...", "path": "...", "content": "..."}}
        """

        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        
        action = json.loads(response.choices[0].message.content)
        
        # 3. Environment Step
        result = await call_env_api("step", action)
        obs = result["observation"]
        reward = result["reward"]
        done = result["done"]
        
        total_reward += reward
        log_step(step, action['command'], reward, done)
        
        if done: break

    log_end(success=(total_reward > 0.7), score=total_reward)

if __name__ == "__main__":
    asyncio.run(main())