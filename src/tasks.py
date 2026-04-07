from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from .models import Action, ActionType, CategoryType, Reward

class Grader(ABC):
    @abstractmethod
    def grade(self, action: Action, ground_truth: Dict[str, Any]) -> Reward:
        pass

class SpamGrader(Grader):
    def grade(self, action: Action, ground_truth: Dict[str, Any]) -> Reward:
        if action.action == ActionType.REJECT and action.category == CategoryType.SPAM:
            return Reward(score=1.0, explanation="Correctly identified and rejected spam.")
        elif action.action == ActionType.REJECT:
            return Reward(score=0.5, explanation="Correctly rejected, but category was wrong.")
        elif action.category == CategoryType.SPAM:
            return Reward(score=0.3, explanation="Correctly identified spam, but took the wrong action.")
        return Reward(score=0.0, explanation="Failed to identify spam.")

class PolicyGrader(Grader):
    def grade(self, action: Action, ground_truth: Dict[str, Any]) -> Reward:
        correct_action = ground_truth.get("correct_action")
        correct_category = ground_truth.get("correct_category")
        
        score = 0.0
        explanation = []
        
        if action.action == correct_action:
            score += 0.6
            explanation.append("Correct action.")
        if action.category == correct_category:
            score += 0.4
            explanation.append("Correct category.")
            
        if not explanation:
            explanation.append("Incorrect action and category.")
            
        return Reward(score=score, explanation=" ".join(explanation))

class ContextGrader(Grader):
    def grade(self, action: Action, ground_truth: Dict[str, Any]) -> Reward:
        if action.action == ActionType.REQUEST_CONTEXT:
            return Reward(score=1.0, explanation="Correctly identified that more context was needed.")
        elif action.action == ActionType.FLAG:
            return Reward(score=0.4, explanation="Flagging is a safe fallback, but requesting context is better.")
        return Reward(score=0.0, explanation="Premature decision without sufficient context.")

# Task definitions
TASKS = {
    "easy_spam_detection": {
        "content": "Click here to win a FREE iPhone! No strings attached! http://scam.link",
        "metadata": {"user_reputation": 0.1, "report_count": 15},
        "policy_context": "Policy 1.1: Spam and phishing links are strictly prohibited.",
        "ground_truth": {"correct_action": ActionType.REJECT, "correct_category": CategoryType.SPAM},
        "grader": SpamGrader(),
        "max_steps": 1
    },
    "medium_policy_nuance": {
        "content": "I hate how people from [Group X] are always so loud in public.",
        "metadata": {"user_reputation": 0.5, "report_count": 2},
        "policy_context": "Policy 2.3: Hate speech targeting protected groups is prohibited. General complaints about behavior are allowed unless they generalize to an entire group.",
        "ground_truth": {"correct_action": ActionType.REJECT, "correct_category": CategoryType.HATE_SPEECH},
        "grader": PolicyGrader(),
        "max_steps": 1
    },
    "hard_context_request": {
        "content": "I'm going to finish what I started tonight.",
        "metadata": {"user_reputation": 0.8, "report_count": 0},
        "policy_context": "Policy 3.4: Threats of self-harm or violence are prohibited. Ambiguous statements should be escalated or context requested.",
        "ground_truth": {"correct_action": ActionType.REQUEST_CONTEXT, "correct_category": CategoryType.OTHER},
        "grader": ContextGrader(),
        "max_steps": 2
    },
    "medium_misinformation": {
        "content": "The new law says that all cats must wear hats on Tuesdays or owners will be fined $500.",
        "metadata": {"user_reputation": 0.3, "report_count": 8},
        "policy_context": "Policy 4.1: Misinformation that causes public confusion or financial harm must be flagged. Fact-check context: No such law exists.",
        "ground_truth": {"correct_action": ActionType.FLAG, "correct_category": CategoryType.MISINFORMATION},
        "grader": PolicyGrader(),
        "max_steps": 1
    },
    "hard_coordinated_behavior": {
        "content": "Everyone should post the same message about the protest at 5 PM today.",
        "metadata": {"user_reputation": 0.2, "report_count": 50},
        "policy_context": "Policy 5.2: Coordinated inauthentic behavior (CIB) involving multiple accounts posting identical content to manipulate discourse is prohibited.",
        "ground_truth": {"correct_action": ActionType.REJECT, "correct_category": CategoryType.OTHER},
        "grader": PolicyGrader(),
        "max_steps": 2
    }
}
