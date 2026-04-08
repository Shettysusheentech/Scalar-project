import os
import json
import sys
from typing import Dict, Any, List
from openai import OpenAI
from src.env import make
from src.models import Action, ActionType, CategoryType

# Environment Variables
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Initialize OpenAI Client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or os.environ.get("OPENAI_API_KEY", "sk-dummy")
)

def log_start(task_id: str):
    print(f"[START] {json.dumps({'task_id': task_id})}")
    sys.stdout.flush()

def log_step(step: int, action: Dict[str, Any], reward: float, done: bool):
    print(f"[STEP] {json.dumps({'step': step, 'action': action, 'reward': reward, 'done': done})}")
    sys.stdout.flush()

def log_end(task_id: str, total_reward: float):
    print(f"[END] {json.dumps({'task_id': task_id, 'total_reward': total_reward})}")
    sys.stdout.flush()

def run_task(task_id: str) -> float:
    env = make(task_id)
    observation = env.reset()
    log_start(task_id)
    
    total_reward = 0.0
    done = False
    step_count = 0
    max_steps = 2 # As per openenv.yaml
    
    while not done and step_count < max_steps:
        step_count += 1
        
        # Construct prompt for the agent
        prompt = f"""
        You are a content moderation agent for NexusSocial.
        Your task is to review the following post and decide on an action.
        
        Ticket ID: {observation.ticket_id}
        Content: {observation.content}
        Metadata: {observation.metadata.dict()}
        Policy Context: {observation.policy_context}
        
        Respond in JSON format with the following fields:
        - action: "APPROVE", "REJECT", "FLAG", or "REQUEST_CONTEXT"
        - reason: A brief justification for your action.
        - category: "SPAM", "HATE_SPEECH", "VIOLENCE", "SAFE", "MISINFORMATION", or "OTHER"
        """
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            # Parse response
            action_data = json.loads(response.choices[0].message.content)
            action = Action(**action_data)
            
            # Step the environment
            observation, reward_obj, done, info = env.step(action)
            
            reward = reward_obj.score
            total_reward += reward
            
            log_step(step_count, action.dict(), reward, done)
            
        except Exception as e:
            print(f"Error in step {step_count}: {str(e)}", file=sys.stderr)
            break
            
    # Normalize score to (0, 1) range
    final_score = total_reward / max(1, step_count)
    final_score = max(0.05, min(0.95, final_score))
    
    log_end(task_id, final_score)
    return final_score

if __name__ == "__main__":
    tasks = [
        "easy_spam_detection", 
        "medium_policy_nuance", 
        "hard_context_request",
        "medium_misinformation",
        "hard_coordinated_behavior"
    ]
    
    overall_score = 0.0
    for task in tasks:
        try:
            score = run_task(task)
            overall_score += score
        except Exception as e:
            print(f"Failed to run task {task}: {str(e)}", file=sys.stderr)
            
    print(f"\nOverall Score: {overall_score / len(tasks):.2f}")
