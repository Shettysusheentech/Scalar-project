import random
from typing import Tuple, Dict, Any, Optional
from .models import Action, Observation, Reward, State, Metadata, ActionType, CategoryType
from .tasks import TASKS

class NexusSocialEnv:
    def __init__(self, task_id: str = "easy_spam_detection"):
        self.task_id = task_id
        self.task_data = TASKS.get(task_id)
        if not self.task_data:
            raise ValueError(f"Task {task_id} not found.")
        
        self.reset()

    def reset(self) -> Observation:
        self.current_step = 0
        self.done = False
        self.history = []
        
        # Initial observation
        return Observation(
            ticket_id=f"TKT-{random.randint(1000, 9999)}",
            content=self.task_data["content"],
            metadata=Metadata(**self.task_data["metadata"]),
            policy_context=self.task_data["policy_context"]
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Environment is already done. Call reset().")
        
        self.current_step += 1
        self.history.append(action.dict())
        
        # Grade the action
        reward = self.task_data["grader"].grade(action, self.task_data["ground_truth"])
        
        # Check if done
        max_steps = self.task_data.get("max_steps", 1)
        
        if self.current_step >= max_steps:
            self.done = True
            return self.reset(), reward, self.done, {}

        # Special handling for multi-step tasks
        if self.task_id == "hard_context_request":
            if action.action == ActionType.REQUEST_CONTEXT:
                # Simulate providing more context
                new_observation = Observation(
                    ticket_id="TKT-CONTEXT",
                    content="Context: User is referring to a video game level they were playing.",
                    metadata=Metadata(user_reputation=0.9, report_count=0),
                    policy_context="Policy 3.4: Threats of self-harm or violence are prohibited. Ambiguous statements should be escalated or context requested."
                )
                return new_observation, reward, self.done, {"info": "Context provided."}
            else:
                self.done = True
        elif self.task_id == "hard_coordinated_behavior":
            # For CIB, maybe the first step is flagging, second is rejection after investigation
            if action.action == ActionType.FLAG:
                new_observation = Observation(
                    ticket_id="TKT-CIB-INVESTIGATION",
                    content="Investigation: 50 identical posts from accounts created in the last 24 hours.",
                    metadata=Metadata(user_reputation=0.1, report_count=100),
                    policy_context="Policy 5.2: Coordinated inauthentic behavior (CIB) is confirmed when multiple accounts post identical content."
                )
                return new_observation, reward, self.done, {"info": "Investigation results provided."}
            else:
                self.done = True
        else:
            self.done = True
            
        # Return same observation if done or no new info
        return self.reset(), reward, self.done, {}

    def state(self) -> State:
        return State(
            current_ticket_id="TKT-CURRENT",
            history=self.history,
            done=self.done
        )

# OpenEnv standard API wrapper
def make(task_id: str):
    return NexusSocialEnv(task_id)
