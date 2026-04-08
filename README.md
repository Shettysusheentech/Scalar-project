---
title: NexusSocial Moderation OpenEnv
emoji: "🛡️"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# NexusSocial Moderation OpenEnv

An OpenEnv-compatible content moderation environment for evaluating agent behavior on spam, hate speech, misinformation, ambiguous threats, and coordinated inauthentic behavior.

## What is in this repo

- `server/app.py`: FastAPI server and built-in dashboard used by the Docker deployment.
- `src/env.py`: Environment lifecycle and multi-step task flow.
- `src/tasks.py`: Task definitions and grading logic.
- `src/App.tsx`: Standalone React client for local frontend development.
- `inference.py`: Baseline script for running an LLM against all tasks.

## Running locally

```bash
pip install -r requirements.txt
py -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Baseline inference

`inference.py` can run with an OpenAI-compatible API key, or fall back to a deterministic local baseline when no key is configured.
