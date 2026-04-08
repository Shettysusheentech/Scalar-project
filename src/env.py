import random
from typing import Tuple, Dict, Any
from .models import Action, Observation, Reward, State, Metadata, ActionType, CategoryType
from .tasks import TASKS

class NexusSocialEnv:
    def __init__(self, task_id: str = "easy_spam_detection"):
        self.task_id = task_id
        self.task_data = TASKS.get(task_id)
        if not self.task_data:
            raise ValueError(f"Task {task_id} not found.")
        
        self.reset()

    def _make_observation(self, content: str, metadata: Dict[str, Any], policy_context: str, ticket_id: str = "") -> Observation:
        return Observation(
            ticket_id=ticket_id or f"TKT-{random.randint(1000, 9999)}",
            content=content,
            metadata=Metadata(**metadata),
            policy_context=policy_context,
        )

    def reset(self) -> Observation:
        self.current_step = 0
        self.done = False
        self.history = []

        self.current_observation = self._make_observation(
            content=self.task_data["content"],
            metadata=self.task_data["metadata"],
            policy_context=self.task_data["policy_context"],
        )
        return self.current_observation

    def _finish(self, reward: Reward, info: Dict[str, Any] | None = None) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self.done = True
        return self.current_observation, reward, True, info or {}

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Environment is already done. Call reset().")
        
        self.current_step += 1
        self.history.append(action.model_dump())

        # Special handling for multi-step tasks
        if self.task_id == "hard_context_request" and self.current_step == 1:
            if action.action == ActionType.REQUEST_CONTEXT:
                reward = Reward(
                    score=0.49,
                    explanation="Correct first move: more context is needed before taking action.",
                )
                self.current_observation = Observation(
                    ticket_id="TKT-CONTEXT",
                    content="Context: User is referring to a video game level they were playing.",
                    metadata=Metadata(user_reputation=0.9, report_count=0),
                    policy_context="Policy 3.4: Threats of self-harm or violence are prohibited. Ambiguous statements should be escalated or context requested."
                )
                return self.current_observation, reward, False, {"message": "Additional context provided."}
            if action.action == ActionType.FLAG:
                reward = Reward(
                    score=0.2,
                    explanation="Escalation is cautious, but the preferred first move was to request context.",
                )
                return self._finish(reward)

        if self.task_id == "hard_context_request" and self.current_step == 2:
            if action.action == ActionType.APPROVE and action.category == CategoryType.SAFE:
                reward = Reward(
                    score=0.49,
                    explanation="Correctly cleared the post after the benign gaming context was revealed.",
                )
            elif action.action == ActionType.FLAG:
                reward = Reward(
                    score=0.2,
                    explanation="Still too cautious after the follow-up context clarified the post.",
                )
            else:
                reward = Reward(
                    score=0.01,
                    explanation="The follow-up action did not match the clarified benign context.",
                )
            return self._finish(reward)

        if self.task_id == "hard_coordinated_behavior" and self.current_step == 1:
            if action.action == ActionType.FLAG:
                reward = Reward(
                    score=0.39,
                    explanation="Good first move: escalate the post while gathering evidence of coordination.",
                )
                self.current_observation = Observation(
                    ticket_id="TKT-CIB-INVESTIGATION",
                    content="Investigation: 50 identical posts from accounts created in the last 24 hours.",
                    metadata=Metadata(user_reputation=0.1, report_count=100),
                    policy_context="Policy 5.2: Coordinated inauthentic behavior (CIB) is confirmed when multiple accounts post identical content."
                )
                return self.current_observation, reward, False, {"message": "Investigation results provided."}
            if action.action == ActionType.REJECT and action.category == CategoryType.OTHER:
                reward = Reward(
                    score=0.7,
                    explanation="The final action is right, but the task expected an investigation step first.",
                )
                return self._finish(reward)
            reward = self.task_data["grader"].grade(action, self.task_data["ground_truth"])
            return self._finish(reward)

        if self.task_id == "hard_coordinated_behavior" and self.current_step == 2:
            if action.action == ActionType.REJECT and action.category == CategoryType.OTHER:
                reward = Reward(
                    score=0.59,
                    explanation="Correctly rejected the content after the investigation confirmed coordinated behavior.",
                )
            else:
                reward = Reward(
                    score=0.01,
                    explanation="The follow-up action did not address the confirmed coordinated behavior.",
                )
            return self._finish(reward)

        reward = self.task_data["grader"].grade(action, self.task_data["ground_truth"])
        return self._finish(reward)

    def state(self) -> State:
        return State(
            current_ticket_id=self.current_observation.ticket_id,
            history=self.history,
            done=self.done
        )

# OpenEnv standard API wrapper
def make(task_id: str):
    return NexusSocialEnv(task_id)
