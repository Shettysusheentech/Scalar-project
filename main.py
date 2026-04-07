import os
import json
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from src.env import make
from src.models import Action, Observation, Reward, State, ActionType, CategoryType
from src.tasks import TASKS

app = FastAPI(title="OpenEnv: NexusSocial Moderation")

# Global environment instances (for demo purposes)
envs: Dict[str, Any] = {}

class StepRequest(BaseModel):
    task_id: str
    action: Action

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
async def get_index():
    # Simple dashboard to visualize the environment
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>OpenEnv Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body {{ background-color: #f3f4f6; }}
            .card {{ background: white; padding: 1.5rem; border-radius: 0.75rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }}
        </style>
    </head>
    <body class="p-8">
        <div class="max-w-4xl mx-auto space-y-8">
            <header class="flex justify-between items-center">
                <h1 class="text-3xl font-bold text-gray-900">OpenEnv: NexusSocial Moderation</h1>
                <span class="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium">v1.0.0</span>
            </header>
            
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div class="card border-t-4 border-blue-500">
                    <h2 class="text-lg font-semibold mb-2">Easy Task</h2>
                    <p class="text-sm text-gray-600 mb-4">Spam Detection</p>
                    <button onclick="loadTask('easy_spam_detection')" class="w-full bg-blue-600 text-white py-2 rounded-md hover:bg-blue-700 transition">Load</button>
                </div>
                <div class="card border-t-4 border-yellow-500">
                    <h2 class="text-lg font-semibold mb-2">Medium Task</h2>
                    <p class="text-sm text-gray-600 mb-4">Policy Nuance</p>
                    <button onclick="loadTask('medium_policy_nuance')" class="w-full bg-yellow-600 text-white py-2 rounded-md hover:bg-yellow-700 transition">Load</button>
                </div>
                <div class="card border-t-4 border-red-500">
                    <h2 class="text-lg font-semibold mb-2">Hard Task</h2>
                    <p class="text-sm text-gray-600 mb-4">Contextual Moderation</p>
                    <button onclick="loadTask('hard_context_request')" class="w-full bg-red-600 text-white py-2 rounded-md hover:bg-red-700 transition">Load</button>
                </div>
            </div>
            
            <div id="task-view" class="card hidden">
                <h2 id="task-title" class="text-xl font-bold mb-4">Task Details</h2>
                <div class="space-y-4">
                    <div class="bg-gray-50 p-4 rounded-md border">
                        <h3 class="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2">Post Content</h3>
                        <p id="post-content" class="text-gray-800 italic"></p>
                    </div>
                    <div class="grid grid-cols-2 gap-4">
                        <div class="bg-gray-50 p-4 rounded-md border">
                            <h3 class="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2">Metadata</h3>
                            <pre id="post-metadata" class="text-xs text-blue-600"></pre>
                        </div>
                        <div class="bg-gray-50 p-4 rounded-md border">
                            <h3 class="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2">Policy Context</h3>
                            <p id="policy-context" class="text-xs text-gray-600"></p>
                        </div>
                    </div>
                </div>
                
                <div class="mt-8 pt-8 border-t">
                    <h3 class="text-lg font-semibold mb-4">Take Action (Agent Simulation)</h3>
                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-1">Action</label>
                            <select id="action-select" class="w-full border rounded-md p-2">
                                <option value="APPROVE">APPROVE</option>
                                <option value="REJECT">REJECT</option>
                                <option value="FLAG">FLAG</option>
                                <option value="REQUEST_CONTEXT">REQUEST_CONTEXT</option>
                            </select>
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-1">Category</label>
                            <select id="category-select" class="w-full border rounded-md p-2">
                                <option value="SAFE">SAFE</option>
                                <option value="SPAM">SPAM</option>
                                <option value="HATE_SPEECH">HATE_SPEECH</option>
                                <option value="VIOLENCE">VIOLENCE</option>
                                <option value="MISINFORMATION">MISINFORMATION</option>
                                <option value="OTHER">OTHER</option>
                            </select>
                        </div>
                    </div>
                    <div class="mt-4">
                        <label class="block text-sm font-medium text-gray-700 mb-1">Reason</label>
                        <input id="reason-input" type="text" class="w-full border rounded-md p-2" placeholder="Explain your decision...">
                    </div>
                    <button onclick="submitAction()" class="mt-6 w-full bg-black text-white py-3 rounded-md font-bold hover:bg-gray-800 transition">Submit Step</button>
                </div>
            </div>
            
            <div id="result-view" class="card hidden border-2 border-green-500">
                <h2 class="text-xl font-bold mb-4 text-green-700">Step Result</h2>
                <div class="space-y-4">
                    <div class="flex items-center justify-between">
                        <span class="text-gray-600">Reward Score:</span>
                        <span id="reward-score" class="text-2xl font-black">0.0</span>
                    </div>
                    <div class="bg-green-50 p-4 rounded-md">
                        <p id="reward-explanation" class="text-green-800"></p>
                    </div>
                    <button onclick="resetTask()" class="w-full bg-gray-200 text-gray-800 py-2 rounded-md hover:bg-gray-300 transition">Reset Environment</button>
                </div>
            </div>
        </div>
        
        <script>
            let currentTaskId = null;
            
            async function loadTask(taskId) {{
                currentTaskId = taskId;
                const response = await fetch(`/reset/${{taskId}}`, {{ method: 'POST' }});
                const obs = await response.json();
                
                document.getElementById('task-view').classList.remove('hidden');
                document.getElementById('result-view').classList.add('hidden');
                document.getElementById('task-title').innerText = "Task: " + taskId;
                document.getElementById('post-content').innerText = obs.content;
                document.getElementById('post-metadata').innerText = JSON.stringify(obs.metadata, null, 2);
                document.getElementById('policy-context').innerText = obs.policy_context;
                
                window.scrollTo({{ top: document.getElementById('task-view').offsetTop - 20, behavior: 'smooth' }});
            }}
            
            async function submitAction() {{
                const action = {{
                    action: document.getElementById('action-select').value,
                    category: document.getElementById('category-select').value,
                    reason: document.getElementById('reason-input').value || "No reason provided"
                }};
                
                const response = await fetch('/step', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ task_id: currentTaskId, action: action }})
                }});
                
                const result = await response.json();
                
                document.getElementById('result-view').classList.remove('hidden');
                document.getElementById('reward-score').innerText = result.reward.score.toFixed(2);
                document.getElementById('reward-explanation').innerText = result.reward.explanation;
                
                if (result.reward.score >= 0.8) {{
                    document.getElementById('reward-score').className = "text-2xl font-black text-green-600";
                }} else if (result.reward.score >= 0.4) {{
                    document.getElementById('reward-score').className = "text-2xl font-black text-yellow-600";
                }} else {{
                    document.getElementById('reward-score').className = "text-2xl font-black text-red-600";
                }}
                
                window.scrollTo({{ top: document.getElementById('result-view').offsetTop - 20, behavior: 'smooth' }});
            }}
            
            function resetTask() {{
                loadTask(currentTaskId);
            }}
        </script>
    </body>
    </html>
    """

@app.post("/reset/{task_id}", response_model=Observation)
async def reset_env(task_id: str):
    envs[task_id] = make(task_id)
    return envs[task_id].reset()

@app.post("/step")
async def step_env(request: StepRequest):
    if request.task_id not in envs:
        envs[request.task_id] = make(request.task_id)
    
    obs, reward, done, info = envs[request.task_id].step(request.action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state/{task_id}", response_model=State)
async def get_state(task_id: str):
    if task_id not in envs:
        raise HTTPException(status_code=404, detail="Task not initialized")
    return envs[task_id].state()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
