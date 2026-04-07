# OpenEnv: NexusSocial Content Moderation

A real-world OpenEnv environment designed for training and evaluating AI agents in the domain of content moderation and policy enforcement.

## Environment Description

NexusSocial is a simulated social media platform where an AI agent acts as a "Safety Moderator". The agent receives "tickets" containing user-generated content, metadata about the user, and relevant policy snippets. The agent must decide whether to approve, reject, flag, or request more context for each post.

### Motivation
Content moderation is a high-stakes, real-world task that requires nuanced understanding of language, context, and policy. This environment provides a safe sandbox to test how well agents can follow complex instructions and handle ambiguity.

## OpenEnv Specification

This environment follows the full OpenEnv spec, including typed Pydantic models for actions, observations, and rewards.

### Action Space
The agent must provide a dictionary with:
- `action`: `APPROVE`, `REJECT`, `FLAG`, or `REQUEST_CONTEXT`
- `category`: `SPAM`, `HATE_SPEECH`, `VIOLENCE`, `SAFE`, `MISINFORMATION`, or `OTHER`
- `reason`: A string explaining the decision.

### Observation Space
The agent receives:
- `ticket_id`: Unique identifier.
- `content`: The text of the post.
- `metadata`: `{"user_reputation": float, "report_count": int}`
- `policy_context`: Relevant text from the platform's rulebook.

## Tasks & Difficulty

1. **Easy: Spam Detection** (`easy_spam_detection`)
   - Objective: Identify obvious phishing and scam links.
   - Reward: 1.0 for REJECT + SPAM.

2. **Medium: Policy Nuance** (`medium_policy_nuance`)
   - Objective: Distinguish between general behavior complaints and targeted hate speech.
   - Reward: Weighted score based on correct action (0.6) and correct category (0.4).

3. **Hard: Contextual Moderation** (`hard_context_request`)
   - Objective: Recognize ambiguous threats that require more context before a decision.
   - Reward: 1.0 for REQUEST_CONTEXT. Partial credit for safe flagging.

4. **Medium: Misinformation** (`medium_misinformation`)
   - Objective: Identify and flag potential misinformation based on provided context.
   - Reward: Weighted score based on FLAG + MISINFORMATION.

5. **Hard: Coordinated Behavior** (`hard_coordinated_behavior`)
   - Objective: Detect patterns of coordinated inauthentic behavior (CIB).
   - Reward: 1.0 for REJECT.

## Setup & Usage

### Local Installation
```bash
pip install -r requirements.txt
```

## Configuration & API Keys

### Providing the API Key
To run the baseline inference script or use the environment with an AI agent, you need a **Gemini API Key**.

#### In Google AI Studio:
1.  Go to the **Settings** menu (gear icon).
2.  Select **Secrets**.
3.  Add a new secret with the key `GEMINI_API_KEY` and your actual API key as the value.
4.  The platform will automatically inject this into the environment at runtime.

#### Locally:
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_api_key_here
```

### Running the Environment
You can run the FastAPI server to interact with the environment via a web dashboard:
```bash
python main.py
```
Then visit `http://localhost:3000`.

### Running Baseline Inference
To run the baseline script:
```bash
python -m src.inference
```

## OpenEnv Specification Compliance
This environment is fully compliant with the OpenEnv specification.
- **Validation**: You can validate the spec using `openenv validate .`
- **Interface**: Implements `step(action)`, `reset()`, and `state()` with typed Pydantic models.
- **Metadata**: Defined in `openenv.yaml`.

## 🚀 Deployment to Hugging Face Spaces

This environment is pre-configured for deployment to Hugging Face Spaces.

1.  **Export the Code**: Click on the **Settings** menu in AI Studio and select **Export to GitHub** or **Download ZIP**.
2.  **Create a Space**: Go to [Hugging Face Spaces](https://huggingface.co/new) and create a new Space (e.g., `susheen/patient-triage`).
3.  **Select SDK**: Choose **Docker** as the Space SDK.
4.  **Upload Files**: Upload all the files from this project (including `Dockerfile`, `requirements.txt`, `openenv.yaml`, `main.py`, and the `src/` directory).
5.  **Set Secrets**: In your Space's **Settings > Variables and secrets**, add:
    *   `GEMINI_API_KEY`: Your Google AI Studio API key (if using Gemini).
    *   `HF_TOKEN`: Your Hugging Face token (for the inference script).
    *   `API_BASE_URL`: The LLM API endpoint.
    *   `MODEL_NAME`: The model identifier.

The Space will automatically build the Docker image and start the FastAPI server on port 7860.
