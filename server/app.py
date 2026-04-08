import json
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.env import make
from src.models import Action, Observation, Reward, State
from src.tasks import TASKS

app = FastAPI(title="OpenEnv: NexusSocial Moderation")

envs: Dict[str, Any] = {}


class StepRequest(BaseModel):
    task_id: str
    action: Action


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]


def build_env(task_id: str):
    try:
        return make(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


async def resolve_task_id(raw_request: Optional[Request], explicit_task_id: Optional[str] = None) -> str:
    if explicit_task_id:
        return explicit_task_id

    if raw_request is not None:
        query_task_id = raw_request.query_params.get("task_id")
        if query_task_id:
            return query_task_id

        content_type = raw_request.headers.get("content-type", "").lower()

        if "application/json" in content_type:
            try:
                payload = await raw_request.json()
            except Exception:
                payload = None
            if isinstance(payload, dict) and isinstance(payload.get("task_id"), str):
                return payload["task_id"]

        if "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
            try:
                form = await raw_request.form()
            except Exception:
                form = None
            if form is not None and isinstance(form.get("task_id"), str):
                return form["task_id"]

    return "easy_spam_detection"


def get_or_create_env(task_id: str):
    env = envs.get(task_id)
    if env is None:
        env = build_env(task_id)
        envs[task_id] = env
    return env


def serialize_model(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_env(ws: WebSocket):
    await ws.accept()
    current_task_id = "easy_spam_detection"
    env = build_env(current_task_id)

    try:
        while True:
            raw = await ws.receive_text()
            message = json.loads(raw)
            msg_type = message.get("type")
            data = message.get("data", {}) or {}

            if msg_type == "close":
                await ws.send_json({"type": "result", "data": {"closed": True}})
                await ws.close()
                return

            if msg_type == "reset":
                requested_task_id = (
                    data.get("task_id")
                    or data.get("task")
                    or data.get("taskId")
                    or current_task_id
                )
                current_task_id = requested_task_id
                env = build_env(current_task_id)
                envs[current_task_id] = env
                observation = env.reset()
                await ws.send_json(
                    {
                        "type": "result",
                        "data": {
                            "observation": serialize_model(observation),
                            "reward": None,
                            "done": False,
                        },
                    }
                )
                continue

            if msg_type == "step":
                requested_task_id = (
                    data.get("task_id")
                    or data.get("task")
                    or data.get("taskId")
                    or current_task_id
                )
                if requested_task_id != current_task_id:
                    current_task_id = requested_task_id
                    env = get_or_create_env(current_task_id)

                if isinstance(data.get("action"), dict):
                    action_payload = data["action"]
                else:
                    action_payload = data
                action = Action(**action_payload)
                observation, reward, done, info = env.step(action)
                await ws.send_json(
                    {
                        "type": "result",
                        "data": {
                            "observation": serialize_model(observation),
                            "reward": float(reward.score),
                            "done": done,
                            "info": info,
                        },
                    }
                )
                continue

            if msg_type == "state":
                state = env.state()
                await ws.send_json(
                    {
                        "type": "result",
                        "data": {
                            "episode_id": current_task_id,
                            "step_count": len(state.history),
                            "current_ticket_id": state.current_ticket_id,
                            "history": state.history,
                            "done": state.done,
                        },
                    }
                )
                continue

            await ws.send_json(
                {
                    "type": "error",
                    "data": {
                        "code": "UNKNOWN_MESSAGE_TYPE",
                        "message": f"Unsupported message type: {msg_type}",
                    },
                }
            )
    except WebSocketDisconnect:
        return
    except Exception as exc:
        try:
            await ws.send_json(
                {
                    "type": "error",
                    "data": {
                        "code": "SERVER_ERROR",
                        "message": str(exc),
                    },
                }
            )
        finally:
            await ws.close()


@app.get("/", response_class=HTMLResponse)
async def get_index():
    task_buttons = "\n".join(
        f"<button onclick=\"loadTask('{task_id}')\" class=\"task-btn\">{task['name']}</button>"
        for task_id, task in TASKS.items()
    )

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NexusSocial | OpenEnv Moderator Dashboard</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: #0f172a;
                color: #e2e8f0;
                margin: 0;
                padding: 24px;
            }}
            .wrap {{
                max-width: 960px;
                margin: 0 auto;
            }}
            .card {{
                background: #111827;
                border: 1px solid #334155;
                border-radius: 16px;
                padding: 20px;
                margin-bottom: 20px;
            }}
            .grid {{
                display: grid;
                gap: 12px;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            }}
            .task-btn, button {{
                background: #1d4ed8;
                color: white;
                border: 0;
                border-radius: 10px;
                padding: 12px 14px;
                cursor: pointer;
            }}
            .task-btn:hover, button:hover {{
                background: #2563eb;
            }}
            select, input {{
                width: 100%;
                padding: 10px;
                border-radius: 10px;
                border: 1px solid #475569;
                background: #020617;
                color: white;
                box-sizing: border-box;
                margin-top: 6px;
            }}
            pre {{
                white-space: pre-wrap;
                word-break: break-word;
                background: #020617;
                padding: 12px;
                border-radius: 10px;
                border: 1px solid #334155;
            }}
            .hidden {{
                display: none;
            }}
            .muted {{
                color: #94a3b8;
            }}
        </style>
    </head>
    <body>
        <div class="wrap">
            <div class="card">
                <h1>NexusSocial Moderation OpenEnv</h1>
                <p class="muted">Simulation dashboard for the OpenEnv moderation environment.</p>
            </div>

            <div class="card">
                <h2>Tasks</h2>
                <div class="grid">
                    {task_buttons}
                </div>
            </div>

            <div id="task-view" class="card hidden">
                <h2 id="task-title">Task</h2>
                <p><strong>Content</strong></p>
                <pre id="post-content"></pre>
                <p><strong>Metadata</strong></p>
                <pre id="post-metadata"></pre>
                <p><strong>Policy Context</strong></p>
                <pre id="policy-context"></pre>

                <label>Action</label>
                <select id="action-select">
                    <option value="APPROVE">APPROVE</option>
                    <option value="REJECT">REJECT</option>
                    <option value="FLAG">FLAG</option>
                    <option value="REQUEST_CONTEXT">REQUEST_CONTEXT</option>
                </select>

                <label>Category</label>
                <select id="category-select">
                    <option value="SAFE">SAFE</option>
                    <option value="SPAM">SPAM</option>
                    <option value="HATE_SPEECH">HATE_SPEECH</option>
                    <option value="VIOLENCE">VIOLENCE</option>
                    <option value="MISINFORMATION">MISINFORMATION</option>
                    <option value="OTHER">OTHER</option>
                </select>

                <label>Reason</label>
                <input id="reason-input" value="No reason provided" />

                <div style="margin-top: 16px; display: flex; gap: 12px;">
                    <button onclick="submitAction()">Submit Step</button>
                    <button onclick="resetTask()" style="background:#475569;">Reset</button>
                </div>
            </div>

            <div id="result-view" class="card hidden">
                <h2>Latest Result</h2>
                <p><strong>Reward</strong></p>
                <pre id="reward-score"></pre>
                <p><strong>Explanation</strong></p>
                <pre id="reward-explanation"></pre>
                <p><strong>Done</strong></p>
                <pre id="done-flag"></pre>
            </div>
        </div>

        <script>
            let currentTaskId = null;

            function updateObservation(obs) {{
                document.getElementById('post-content').textContent = obs.content;
                document.getElementById('post-metadata').textContent = JSON.stringify(obs.metadata, null, 2);
                document.getElementById('policy-context').textContent = obs.policy_context;
            }}

            async function loadTask(taskId) {{
                currentTaskId = taskId;
                const response = await fetch(`/reset/${{taskId}}`, {{ method: 'POST' }});
                const obs = await response.json();
                document.getElementById('task-view').classList.remove('hidden');
                updateObservation(obs);
                document.getElementById('task-title').textContent = taskId;
            }}

            async function submitAction() {{
                const action = {{
                    action: document.getElementById('action-select').value,
                    category: document.getElementById('category-select').value,
                    reason: document.getElementById('reason-input').value || 'No reason provided'
                }};

                const response = await fetch('/step', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ task_id: currentTaskId, action }})
                }});
                const result = await response.json();
                document.getElementById('result-view').classList.remove('hidden');
                document.getElementById('reward-score').textContent = String(result.reward.score);
                document.getElementById('reward-explanation').textContent = result.reward.explanation;
                document.getElementById('done-flag').textContent = String(result.done);
                updateObservation(result.observation);
            }}

            async function resetTask() {{
                if (currentTaskId) {{
                    await loadTask(currentTaskId);
                }}
            }}
        </script>
    </body>
    </html>
    """


@app.get("/tasks")
async def list_tasks():
    return {
        "tasks": [
            {
                "id": task_id,
                "name": task["name"],
                "description": task["description"],
                "difficulty": task["difficulty"],
            }
            for task_id, task in TASKS.items()
        ]
    }


@app.post("/reset", response_model=Observation)
async def reset_env_standard(raw_request: Request):
    task_id = await resolve_task_id(raw_request)
    envs[task_id] = build_env(task_id)
    return envs[task_id].reset()


@app.post("/reset/{task_id}", response_model=Observation)
async def reset_env(task_id: str):
    envs[task_id] = build_env(task_id)
    return envs[task_id].reset()


@app.post("/step", response_model=StepResponse)
async def step_env(request: StepRequest):
    env = get_or_create_env(request.task_id)
    try:
        observation, reward, done, info = env.step(request.action)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state", response_model=State)
async def get_state(task_id: Optional[str] = None):
    resolved_task_id = task_id or next(iter(envs), "easy_spam_detection")
    env = get_or_create_env(resolved_task_id)
    return env.state()


@app.get("/state/{task_id}", response_model=State)
async def get_task_state(task_id: str):
    env = get_or_create_env(task_id)
    return env.state()


def main():
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
