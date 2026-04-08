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

class ResetRequest(BaseModel):
    task_id: str = "easy_spam_detection"

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
async def get_index():
    # Refined dashboard to visualize the environment
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NexusSocial | OpenEnv Moderator Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
        <style>
            :root {{
                --font-sans: 'Inter', sans-serif;
                --font-mono: 'JetBrains Mono', monospace;
            }}
            body {{ 
                font-family: var(--font-sans);
                background-color: #0f172a; 
                color: #f8fafc;
            }}
            .glass {{ 
                background: rgba(30, 41, 59, 0.7); 
                backdrop-filter: blur(12px); 
                border: 1px solid rgba(255, 255, 255, 0.1); 
            }}
            .card-gradient {{
                background: linear-gradient(135deg, rgba(51, 65, 85, 0.5) 0%, rgba(15, 23, 42, 0.5) 100%);
            }}
            .mono {{ font-family: var(--font-mono); }}
            .animate-in {{ animation: fadeIn 0.5s ease-out forwards; }}
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(10px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            select, input {{
                background-color: #1e293b !important;
                border-color: #334155 !important;
                color: white !important;
            }}
            .btn-primary {{
                background: linear-gradient(to right, #6366f1, #a855f7);
                transition: all 0.3s ease;
            }}
            .btn-primary:hover {{
                transform: translateY(-1px);
                box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.4);
            }}
        </style>
    </head>
    <body class="min-h-screen p-4 md:p-8 selection:bg-indigo-500/30">
        <div class="max-w-5xl mx-auto space-y-10">
            <!-- Header -->
            <header class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-slate-800 pb-8">
                <div>
                    <h1 class="text-4xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 to-purple-400">
                        NexusSocial
                    </h1>
                    <p class="text-slate-400 mt-1 font-medium">Safety & Policy Enforcement Environment</p>
                </div>
                <div class="flex items-center gap-3">
                    <div class="flex flex-col items-end">
                        <span class="text-[10px] uppercase tracking-widest text-slate-500 font-bold">Environment Status</span>
                        <span class="text-xs font-semibold text-emerald-400 flex items-center gap-1.5">
                            <span class="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></span>
                            Live & Operational
                        </span>
                    </div>
                    <div class="h-8 w-[1px] bg-slate-800 mx-2"></div>
                    <span class="px-3 py-1 bg-slate-800 text-slate-300 rounded-lg text-xs font-mono border border-slate-700">v1.2.0-openenv</span>
                </div>
            </header>
            
            <!-- Task Selection -->
            <section>
                <h2 class="text-sm font-bold text-slate-500 uppercase tracking-[0.2em] mb-6">Select Evaluation Scenario</h2>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <!-- Easy -->
                    <div class="glass p-6 rounded-2xl group hover:border-indigo-500/50 transition-all duration-300">
                        <div class="w-10 h-10 rounded-xl bg-emerald-500/10 flex items-center justify-center mb-4 border border-emerald-500/20">
                            <svg class="w-5 h-5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                        </div>
                        <h3 class="text-lg font-bold text-white group-hover:text-indigo-300 transition-colors">Spam Detection</h3>
                        <p class="text-sm text-slate-400 mt-2 mb-6 leading-relaxed">Identify obvious phishing and scam links with high precision.</p>
                        <button onclick="loadTask('easy_spam_detection')" class="w-full bg-slate-800 text-white py-2.5 rounded-xl font-semibold hover:bg-slate-700 border border-slate-700 transition-all active:scale-95">Initialize Task</button>
                    </div>
                    
                    <!-- Medium -->
                    <div class="glass p-6 rounded-2xl group hover:border-indigo-500/50 transition-all duration-300">
                        <div class="w-10 h-10 rounded-xl bg-amber-500/10 flex items-center justify-center mb-4 border border-amber-500/20">
                            <svg class="w-5 h-5 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>
                        </div>
                        <h3 class="text-lg font-bold text-white group-hover:text-indigo-300 transition-colors">Policy Nuance</h3>
                        <p class="text-sm text-slate-400 mt-2 mb-6 leading-relaxed">Distinguish between general complaints and targeted hate speech.</p>
                        <button onclick="loadTask('medium_policy_nuance')" class="w-full bg-slate-800 text-white py-2.5 rounded-xl font-semibold hover:bg-slate-700 border border-slate-700 transition-all active:scale-95">Initialize Task</button>
                    </div>
                    
                    <!-- Hard -->
                    <div class="glass p-6 rounded-2xl group hover:border-indigo-500/50 transition-all duration-300">
                        <div class="w-10 h-10 rounded-xl bg-rose-500/10 flex items-center justify-center mb-4 border border-rose-500/20">
                            <svg class="w-5 h-5 text-rose-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
                        </div>
                        <h3 class="text-lg font-bold text-white group-hover:text-indigo-300 transition-colors">Contextual Triage</h3>
                        <p class="text-sm text-slate-400 mt-2 mb-6 leading-relaxed">Recognize ambiguous threats that require multi-step investigation.</p>
                        <button onclick="loadTask('hard_context_request')" class="w-full bg-slate-800 text-white py-2.5 rounded-xl font-semibold hover:bg-slate-700 border border-slate-700 transition-all active:scale-95">Initialize Task</button>
                    </div>
                </div>
            </section>
            
            <!-- Main Interaction Area -->
            <div id="interaction-container" class="grid grid-cols-1 lg:grid-cols-5 gap-8 items-start">
                
                <!-- Task View (Left 3/5) -->
                <div id="task-view" class="lg:col-span-3 glass rounded-3xl overflow-hidden hidden animate-in">
                    <div class="p-8 space-y-8">
                        <div class="flex justify-between items-center">
                            <h2 id="task-title" class="text-2xl font-bold text-white">Scenario Details</h2>
                            <span class="px-3 py-1 bg-indigo-500/20 text-indigo-300 rounded-full text-[10px] font-bold uppercase tracking-widest border border-indigo-500/30">Active Session</span>
                        </div>
                        
                        <div class="space-y-6">
                            <div class="space-y-2">
                                <label class="text-[10px] font-bold text-slate-500 uppercase tracking-widest">User Post Content</label>
                                <div class="bg-slate-900/50 p-6 rounded-2xl border border-slate-800 italic text-lg text-slate-200 leading-relaxed">
                                    "<span id="post-content"></span>"
                                </div>
                            </div>
                            
                            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <div class="space-y-2">
                                    <label class="text-[10px] font-bold text-slate-500 uppercase tracking-widest">User Metadata</label>
                                    <div class="bg-slate-900/50 p-4 rounded-2xl border border-slate-800">
                                        <pre id="post-metadata" class="text-xs text-indigo-400 mono"></pre>
                                    </div>
                                </div>
                                <div class="space-y-2">
                                    <label class="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Applicable Policy</label>
                                    <div class="bg-slate-900/50 p-4 rounded-2xl border border-slate-800">
                                        <p id="policy-context" class="text-xs text-slate-400 leading-relaxed"></p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="pt-8 border-t border-slate-800">
                            <h3 class="text-lg font-bold text-white mb-6 flex items-center gap-2">
                                <svg class="w-5 h-5 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"></path></svg>
                                Moderator Action
                            </h3>
                            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <div class="space-y-2">
                                    <label class="text-xs font-semibold text-slate-400">Primary Decision</label>
                                    <select id="action-select" class="w-full rounded-xl p-3 text-sm focus:ring-2 focus:ring-indigo-500 outline-none transition-all">
                                        <option value="APPROVE">APPROVE</option>
                                        <option value="REJECT">REJECT</option>
                                        <option value="FLAG">FLAG</option>
                                        <option value="REQUEST_CONTEXT">REQUEST_CONTEXT</option>
                                    </select>
                                </div>
                                <div class="space-y-2">
                                    <label class="text-xs font-semibold text-slate-400">Violation Category</label>
                                    <select id="category-select" class="w-full rounded-xl p-3 text-sm focus:ring-2 focus:ring-indigo-500 outline-none transition-all">
                                        <option value="SAFE">SAFE</option>
                                        <option value="SPAM">SPAM</option>
                                        <option value="HATE_SPEECH">HATE_SPEECH</option>
                                        <option value="VIOLENCE">VIOLENCE</option>
                                        <option value="MISINFORMATION">MISINFORMATION</option>
                                        <option value="OTHER">OTHER</option>
                                    </select>
                                </div>
                            </div>
                            <div class="mt-6 space-y-2">
                                <label class="text-xs font-semibold text-slate-400">Justification / Reason</label>
                                <input id="reason-input" type="text" class="w-full rounded-xl p-3 text-sm focus:ring-2 focus:ring-indigo-500 outline-none transition-all" placeholder="Enter reasoning for audit trail...">
                            </div>
                            <button onclick="submitAction()" class="btn-primary mt-8 w-full text-white py-4 rounded-2xl font-bold text-lg shadow-xl active:scale-[0.98]">
                                Execute Moderation Step
                            </button>
                        </div>
                    </div>
                </div>
                
                <!-- Result View (Right 2/5) -->
                <div id="result-view" class="lg:col-span-2 space-y-6 hidden animate-in">
                    <div class="glass rounded-3xl p-8 border-2 border-indigo-500/30">
                        <h2 class="text-xl font-bold text-white mb-6 flex items-center gap-2">
                            <svg class="w-6 h-6 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                            Evaluation Result
                        </h2>
                        <div class="space-y-8">
                            <div class="text-center p-8 rounded-2xl bg-slate-900/50 border border-slate-800">
                                <span class="text-[10px] font-bold text-slate-500 uppercase tracking-[0.3em] block mb-2">Reward Score</span>
                                <span id="reward-score" class="text-6xl font-black tracking-tighter">0.0</span>
                                <span class="text-slate-500 text-sm block mt-2">/ 1.00</span>
                            </div>
                            
                            <div class="space-y-2">
                                <label class="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Grader Feedback</label>
                                <div class="bg-indigo-500/10 p-4 rounded-2xl border border-indigo-500/20">
                                    <p id="reward-explanation" class="text-indigo-200 text-sm leading-relaxed"></p>
                                </div>
                            </div>
                            
                            <button onclick="resetTask()" class="w-full bg-slate-800 text-slate-300 py-3 rounded-xl font-bold hover:bg-slate-700 border border-slate-700 transition-all active:scale-95">
                                Reset Environment
                            </button>
                        </div>
                    </div>
                    
                    <!-- Stats Card -->
                    <div class="glass rounded-3xl p-6 bg-indigo-500/5 border-indigo-500/10">
                        <div class="flex items-center gap-4">
                            <div class="w-12 h-12 rounded-full bg-indigo-500/20 flex items-center justify-center">
                                <svg class="w-6 h-6 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                            </div>
                            <div>
                                <h4 class="text-sm font-bold text-white">OpenEnv Compliance</h4>
                                <p class="text-xs text-slate-400">This session is being logged for baseline evaluation.</p>
                            </div>
                        </div>
                    </div>
                </div>
                
            </div>
            
            <!-- Empty State -->
            <div id="empty-state" class="text-center py-20 glass rounded-[3rem] border-dashed border-2 border-slate-800">
                <div class="w-20 h-20 bg-slate-800/50 rounded-full flex items-center justify-center mx-auto mb-6">
                    <svg class="w-10 h-10 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path></svg>
                </div>
                <h3 class="text-xl font-bold text-slate-300">No Active Session</h3>
                <p class="text-slate-500 mt-2">Select a scenario above to begin the evaluation environment.</p>
            </div>
        </div>
        
        <script>
            let currentTaskId = null;
            
            async function loadTask(taskId) {{
                currentTaskId = taskId;
                
                // Show loading state if needed
                
                const response = await fetch(`/reset/${{taskId}}`, {{ method: 'POST' }});
                const obs = await response.json();
                
                document.getElementById('empty-state').classList.add('hidden');
                document.getElementById('task-view').classList.remove('hidden');
                document.getElementById('result-view').classList.add('hidden');
                
                document.getElementById('task-title').innerText = "Scenario: " + taskId.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
                document.getElementById('post-content').innerText = obs.content;
                document.getElementById('post-metadata').innerText = JSON.stringify(obs.metadata, null, 2);
                document.getElementById('policy-context').innerText = obs.policy_context;
                
                // Reset inputs
                document.getElementById('reason-input').value = "";
                
                window.scrollTo({{ top: document.getElementById('task-view').offsetTop - 40, behavior: 'smooth' }});
            }}
            
            async function submitAction() {{
                const btn = event.target;
                const originalText = btn.innerText;
                btn.innerText = "Processing...";
                btn.disabled = true;
                btn.classList.add('opacity-50');

                const action = {{
                    action: document.getElementById('action-select').value,
                    category: document.getElementById('category-select').value,
                    reason: document.getElementById('reason-input').value || "No reason provided"
                }};
                
                try {{
                    const response = await fetch('/step', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ task_id: currentTaskId, action: action }})
                    }});
                    
                    const result = await response.json();
                    
                    document.getElementById('result-view').classList.remove('hidden');
                    document.getElementById('reward-score').innerText = result.reward.score.toFixed(2);
                    document.getElementById('reward-explanation').innerText = result.reward.explanation;
                    
                    const scoreEl = document.getElementById('reward-score');
                    if (result.reward.score >= 0.8) {{
                        scoreEl.className = "text-6xl font-black tracking-tighter text-emerald-400";
                    }} else if (result.reward.score >= 0.4) {{
                        scoreEl.className = "text-6xl font-black tracking-tighter text-amber-400";
                    }} else {{
                        scoreEl.className = "text-6xl font-black tracking-tighter text-rose-400";
                    }}
                    
                    window.scrollTo({{ top: document.getElementById('result-view').offsetTop - 40, behavior: 'smooth' }});
                }} catch (e) {{
                    console.error(e);
                    alert("Error submitting action. Check console.");
                }} finally {{
                    btn.innerText = originalText;
                    btn.disabled = false;
                    btn.classList.remove('opacity-50');
                }}
            }}
            
            function resetTask() {{
                loadTask(currentTaskId);
            }}
        </script>
    </body>
    </html>
    """

@app.get("/tasks")
async def list_tasks():
    return {
        "tasks": [
            {"id": t_id, "name": t["name"], "difficulty": t["difficulty"]}
            for t_id, t in TASKS.items()
        ]
    }

@app.post("/reset", response_model=Observation)
async def reset_env_standard(request: ResetRequest = None, task_id: str = None):
    t_id = task_id or (request.task_id if request else "easy_spam_detection")
    envs[t_id] = make(t_id)
    return envs[t_id].reset()

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

def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main()
