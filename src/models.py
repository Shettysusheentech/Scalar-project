from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class ActionType(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    FLAG = "FLAG"
    REQUEST_CONTEXT = "REQUEST_CONTEXT"

class CategoryType(str, Enum):
    SPAM = "SPAM"
    HATE_SPEECH = "HATE_SPEECH"
    VIOLENCE = "VIOLENCE"
    SAFE = "SAFE"
    MISINFORMATION = "MISINFORMATION"
    OTHER = "OTHER"

class Action(BaseModel):
    action: ActionType
    reason: str = Field(..., description="A brief justification for the action.")
    category: CategoryType

class Metadata(BaseModel):
    user_reputation: float
    report_count: int

class Observation(BaseModel):
    ticket_id: str
    content: str
    metadata: Metadata
    policy_context: str

class Reward(BaseModel):
    score: float = Field(..., gt=0.0, lt=1.0)
    explanation: str

    @property
    def value(self) -> float:
        return self.score

class State(BaseModel):
    current_ticket_id: str
    history: List[Dict]
    done: bool
