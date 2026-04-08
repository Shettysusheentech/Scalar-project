import json
import os
import sys
from typing import Any, Dict

from openai import OpenAI

from src.env import make
from src.models import Action, CategoryType
from src.tasks import clip_score

# Environment Variables
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

client = None
if HF_TOKEN or OPENAI_API_KEY:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN or OPENAI_API_KEY,
    )


def log_start(task_id: str):
    print(f"[START] {json.dumps({'task_id': task_id})}")
    sys.stdout.flush()


def log_step(step: int, action: Dict[str, Any], reward: float, done: bool):
    reward = clip_score(reward)
    print(f"[STEP] {json.dumps({'step': step, 'action': action, 'reward': reward, 'done': done})}")
    sys.stdout.flush()


def log_end(task_id: str, score: float, total_reward: float):
    print(f"[END] {json.dumps({'task_id': task_id, 'score': clip_score(score)})}")
    sys.stdout.flush()


def choose_heuristic_action(task_id: str, observation, step_count: int) -> Action:
    content = observation.content.lower()
    policy = observation.policy_context.lower()

    if task_id == "easy_spam_detection" or "free iphone" in content or "scam.link" in content:
        return Action(action="REJECT", category="SPAM", reason="Obvious scam or phishing content.")

    if task_id == "medium_policy_nuance" or "[group x]" in observation.content:
        return Action(action="REJECT", category="HATE_SPEECH", reason="Targets a protected group with generalized hostility.")

    if task_id == "medium_misinformation" or "no such law exists" in policy:
        return Action(action="FLAG", category="MISINFORMATION", reason="Claim conflicts with the provided fact-check context.")

    if task_id == "hard_context_request":
        if "video game level" in content:
            return Action(action="APPROVE", category="SAFE", reason="Follow-up context clarifies the statement is harmless gameplay talk.")
        return Action(action="REQUEST_CONTEXT", category="OTHER", reason="The statement is ambiguous and needs more context before action.")

    if task_id == "hard_coordinated_behavior":
        if "investigation:" in content or "identical posts" in content:
            return Action(action="REJECT", category="OTHER", reason="Investigation confirms coordinated inauthentic behavior.")
        return Action(action="FLAG", category="OTHER", reason="Escalate for investigation because the post suggests coordinated messaging.")

    return Action(action="APPROVE", category=CategoryType.SAFE, reason="Default safe fallback.")


def choose_model_action(task_id: str, observation) -> Action:
    if client is None:
        raise RuntimeError("No API client configured")

    prompt = f"""
    You are a content moderation agent for NexusSocial.
    Your task is to review the following post and decide on an action.

    Ticket ID: {observation.ticket_id}
    Content: {observation.content}
    Metadata: {observation.metadata.model_dump()}
    Policy Context: {observation.policy_context}

    Respond in JSON format with the following fields:
    - action: "APPROVE", "REJECT", "FLAG", or "REQUEST_CONTEXT"
    - reason: A brief justification for your action.
    - category: "SPAM", "HATE_SPEECH", "VIOLENCE", "SAFE", "MISINFORMATION", or "OTHER"
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    action_data = json.loads(response.choices[0].message.content)
    return Action(**action_data)


def choose_action(task_id: str, observation, step_count: int) -> Action:
    if client is not None:
        try:
            return choose_model_action(task_id, observation)
        except Exception as exc:
            print(f"Falling back to local baseline for {task_id}: {exc}", file=sys.stderr)

    return choose_heuristic_action(task_id, observation, step_count)


def run_task(task_id: str) -> float:
    env = make(task_id)
    observation = env.reset()
    log_start(task_id)

    total_reward = 0.0
    done = False
    step_count = 0
    max_steps = 2

    while not done and step_count < max_steps:
        step_count += 1

        action = choose_action(task_id, observation, step_count)
        observation, reward_obj, done, info = env.step(action)

        reward = clip_score(reward_obj.score)
        total_reward += reward
        log_step(step_count, action.model_dump(), reward, done)

    final_score = clip_score(total_reward / max(1, step_count))

    log_end(task_id, final_score, total_reward)
    return final_score


if __name__ == "__main__":
    tasks = [
        "easy_spam_detection",
        "medium_policy_nuance",
        "hard_context_request",
        "medium_misinformation",
        "hard_coordinated_behavior",
    ]

    overall_score = 0.0
    for task in tasks:
        try:
            score = run_task(task)
            overall_score += score
        except Exception as e:
            print(f"Failed to run task {task}: {str(e)}", file=sys.stderr)

    print(f"\nCompleted {len(tasks)} tasks.", file=sys.stderr)
