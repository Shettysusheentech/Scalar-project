---
title: NexusSocial Moderation OpenEnv
emoji: "shield"
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
- `openenv.yaml`: OpenEnv task metadata and action/observation schema.

## Tasks

- `easy_spam_detection`
- `medium_policy_nuance`
- `hard_context_request`
- `medium_misinformation`
- `hard_coordinated_behavior`

## Local setup

### Python server

```bash
py -m pip install -r requirements.txt
py -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Then open `http://localhost:7860`.

### React client

```bash
npm install
npm run dev
```

Then open `http://localhost:3000`.

## Environment variables

The baseline script supports OpenAI-compatible endpoints:

```env
OPENAI_API_KEY=your_key_here
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4
HF_TOKEN=
```

`HF_TOKEN` is optional and can be used as the API key when targeting a Hugging Face-hosted compatible endpoint.

## Run the baseline

```bash
py inference.py
```

## API endpoints

- `GET /health`
- `GET /tasks`
- `POST /reset`
- `POST /reset/{task_id}`
- `POST /step`
- `GET /state/{task_id}`

## Hugging Face Spaces

This repo is configured for Docker Spaces. The container starts the FastAPI app on port `7860`, which matches Hugging Face Spaces expectations.
