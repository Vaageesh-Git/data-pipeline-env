# inference.py
import os
import asyncio
import json
from openai import AsyncOpenAI
from models import Action, Observation

# Mandatory Environment Variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Note: In a real OpenEnv deployment, the environment is accessed via 
# its URL. For this baseline, we assume it's running locally on 7860.
ENV_URL = "http://localhost:7860"

async def call_env_api(endpoint: str, data: dict = None):
    import httpx
    async with httpx.AsyncClient() as client:
        url = f"{ENV_URL}/{endpoint}"
        if data:
            response = await client.post(url, json=data)
        else:
            response = await client.get(url)
        return response.json()

def log_start(task):
    print(f"[START] Task: {task}")

def log_step(step, action, reward, done):
    print(f"[STEP] Step: {step} | Action: {action} | Reward: {reward} | Done: {done}")

def log_end(success, score):
    print(f"[END] Success: {success} | Final Score: {score}")

async def main():
    client = AsyncOpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    
    # We will test Task 0 (The Easy Task)
    task_id = 0
    init_obs = await call_env_api("reset", {"task_id": task_id})
    
    log_start(init_obs['current_task'] if 'current_task' in init_obs else "Task 0")
    
    history = []
    total_reward = 0.0
    done = False
    max_steps = 5

    for step in range(1, max_steps + 1):
        # 1. Ask the Model what to do
        prompt = f"""
        You are a Data Engineer. Your task is: {init_obs.get('logs', 'Fix the pipeline')}.
        Files available: {init_obs.get('files')}
        Previous Logs: {init_obs.get('logs')}
        Current Data Preview: {init_obs.get('data_preview')}
        
        Respond ONLY with a JSON object containing:
        {{"command": "write", "path": "pipeline.py", "content": "YOUR_CODE_HERE"}} 
        OR {{"command": "run"}} 
        OR {{"command": "submit"}}
        """
        
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        
        action_data = json.loads(response.choices[0].message.content)
        
        # 2. Execute Action in Env
        step_result = await call_env_api("step", action_data)
        obs = step_result["observation"]
        reward = step_result["reward"]
        done = step_result["done"]
        
        total_reward += reward
        log_step(step, action_data['command'], reward, done)
        
        if done:
            break

    success = total_reward > 0.5
    log_end(success, total_reward)

if __name__ == "__main__":
    asyncio.run(main())