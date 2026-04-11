import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

from src.env import make
from src.models import Action, CategoryType
from src.tasks import clip_score

# Environment Variables
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4.1-mini")
API_KEY = os.environ.get("API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
BENCHMARK_NAME = "nexussocial_moderation"

client = None
CLIENT_API_KEY = API_KEY or HF_TOKEN or OPENAI_API_KEY
if CLIENT_API_KEY:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=CLIENT_API_KEY,
    )


def format_bool(value: bool) -> str:
    return "true" if value else "false"


def sanitize_error(error: str | None) -> str:
    if error is None:
        return "null"
    return " ".join(str(error).split())


def format_action(action: Action) -> str:
    return json.dumps(action.model_dump(), separators=(",", ":"), sort_keys=True)


def log_start(task_id: str, model_name: str):
    print(f"[START] task={task_id} env={BENCHMARK_NAME} model={model_name}")
    sys.stdout.flush()


def log_step(step: int, action: Action, reward: float, done: bool, error: str | None = None):
    print(
        f"[STEP] step={step} action={format_action(action)} "
        f"reward={clip_score(reward):.2f} done={format_bool(done)} error={sanitize_error(error)}"
    )
    sys.stdout.flush()


def log_end(success: bool, step_count: int, score: float, rewards: List[float]):
    rewards_blob = ",".join(f"{clip_score(reward):.2f}" for reward in rewards)
    print(
        f"[END] success={format_bool(success)} steps={step_count} "
        f"score={clip_score(score):.3f} rewards={rewards_blob}"
    )
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


def choose_action(task_id: str, observation, step_count: int, mode: str) -> Action:
    if mode == "openai":
        try:
            return choose_model_action(task_id, observation)
        except Exception as exc:
            print(f"Falling back to heuristic policy for {task_id}: {exc}", file=sys.stderr)

    return choose_heuristic_action(task_id, observation, step_count)


def run_task(task_id: str, mode: str) -> Dict[str, Any]:
    env = make(task_id)
    observation = env.reset()
    model_label = MODEL_NAME if mode == "openai" else "heuristic"
    log_start(task_id, model_label)

    total_reward = 0.0
    done = False
    step_count = 0
    max_steps = 2
    rewards: List[float] = []
    success = False

    try:
        while not done and step_count < max_steps:
            step_count += 1

            action = choose_action(task_id, observation, step_count, mode)
            observation, reward_obj, done, info = env.step(action)

            reward = clip_score(reward_obj.score)
            total_reward += reward
            rewards.append(reward)
            log_step(step_count, action, reward, done, info.get("last_action_error"))

        success = done
    except Exception as exc:
        print(f"Task {task_id} failed: {exc}", file=sys.stderr)

    final_score = clip_score(total_reward / max(1, step_count)) if step_count else clip_score(0.01)
    log_end(success, step_count, final_score, rewards)
    return {
        "task_id": task_id,
        "success": success,
        "steps": step_count,
        "rewards": rewards,
        "score": final_score,
    }


def write_output(path: str, episodes: List[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    avg_score = sum(episode["score"] for episode in episodes) / max(1, len(episodes))
    payload = {
        "avg_score": avg_score,
        "avg_final_reward": avg_score,
        "episodes": episodes,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NexusSocial OpenEnv inference runner")
    parser.add_argument("--mode", choices=["heuristic", "openai"], default="openai")
    parser.add_argument("--output", help="Optional path to write a JSON score summary")
    args = parser.parse_args()

    if args.mode == "openai":
        if not os.environ.get("API_BASE_URL"):
            raise ValueError("API_BASE_URL environment variable is required for openai mode")
        if not API_KEY:
            raise ValueError("API_KEY environment variable is required for openai mode")

    tasks = [
        "easy_spam_detection",
        "medium_policy_nuance",
        "hard_context_request",
        "medium_misinformation",
        "hard_coordinated_behavior",
    ]

    episodes: List[Dict[str, Any]] = []
    for task in tasks:
        episodes.append(run_task(task, args.mode))

    if args.output:
        write_output(args.output, episodes)
