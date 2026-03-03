import os
from typing import Optional
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class EmotionalState:
    def __init__(self):
        self.situation_summary: Optional[str] = None
        self.reaction_type: Optional[str] = None
        self.intensity: int = 0
        self.candidate_reaction: Optional[str] = None
        self.previous_reaction: Optional[str] = None

        # ---- To-Do System (NEW) ----
        self.awaiting_todo_offer: bool = False
        self.awaiting_todo_dump: bool = False
        self.awaiting_todo_confirmation: bool = False
        self.current_todo: list[str] = []
        self.auto_todo_used: bool = False


# ---------------- Hyperarousal Detection ---------------- #

def detect_hyperarousal(text: str) -> bool:
    keywords = [
        "can't breathe", "cant breathe",
        "heart racing", "racing heart",
        "panic attack", "shaking",
        "trembling", "dizzy",
        "faint", "chest tight",
        "losing control", "suffocating"
    ]
    text = text.lower()
    return any(k in text for k in keywords)


# ---------------- Reaction Classification ---------------- #

def classify_reaction_llm(message: str) -> str:

    prompt = f"""
Classify the user's emotional reaction into ONE of these labels:

freeze
rumination
anticipatory
none

Return ONLY the label.

Message:
{message}
"""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You are a precise emotional classifier."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        label = response.choices[0].message.content.strip().lower()
        allowed = {"freeze", "rumination", "anticipatory", "none"}
        return label if label in allowed else "none"

    except:
        return "none"


def update_reaction(state: EmotionalState, new_label: str):

    if new_label == "none":
        return

    if state.reaction_type is None:
        state.reaction_type = new_label
        state.intensity = 1
        state.auto_todo_used = False
        return

    if new_label == state.reaction_type:
        state.candidate_reaction = None
        return

    if state.candidate_reaction == new_label:
        state.reaction_type = new_label
        state.candidate_reaction = None
        state.auto_todo_used = False
    else:
        state.candidate_reaction = new_label


def update_situation(state: EmotionalState, text: str):
    text = text.lower()

    if "exam" in text:
        state.situation_summary = "Exam upcoming"
    elif "interview" in text:
        state.situation_summary = "Interview upcoming"
    elif "job" in text:
        state.situation_summary = "Career transition"


# ---------------- Cognitive Overload Gate ---------------- #

def is_cognitive_overload(message: str) -> bool:
    message = message.lower()

    task_signals = {
        "too much", "lot left", "everything left",
        "so much to do", "don't know where to start",
        "portion", "syllabus", "deadline",
        "prepare", "study", "competition",
        "exam", "interview", "assignment"
    }

    emotional_distress = {
        "crying", "heartbroken", "devastated",
        "can't stop crying", "feel broken",
        "panic", "terrified", "uncontrollably"
    }

    if any(term in message for term in emotional_distress):
        return False

    if any(term in message for term in task_signals):
        return True

    return False


# ---------------- Intervention Decision ---------------- #

def decide_intervention(state: EmotionalState, user_message: str) -> str:
    message = user_message.lower().strip()

    confirm_terms = {"yes", "yeah", "ok", "okay", "sure", "that works"}
    decline_terms = {"no", "nah", "not now"}
    planning_terms = {"to do", "todo", "plan", "schedule", "make a list"}

    if state.awaiting_todo_offer:
        if any(term in message for term in confirm_terms):
            state.awaiting_todo_offer = False
            state.awaiting_todo_dump = True
            return "request_dump"
        if any(term in message for term in decline_terms):
            state.awaiting_todo_offer = False
            return "decline_offer"
        return "none"

    if state.awaiting_todo_dump:
        state.awaiting_todo_dump = False
        return "process_dump"

    if any(term in message for term in planning_terms):
        state.awaiting_todo_dump = True
        return "request_dump"

    if (
        state.reaction_type in {"anticipatory", "freeze"}
        and state.intensity <= 1
        and not state.auto_todo_used
        and is_cognitive_overload(message)
    ):
        state.auto_todo_used = True
        state.awaiting_todo_offer = True
        return "offer"

    return "none"


# ---------------- Main Update ---------------- #

def update_state(state: EmotionalState, user_message: str):

    if detect_hyperarousal(user_message):
        state.previous_reaction = state.reaction_type
        state.reaction_type = "hyperarousal"
        state.intensity = 2
        state.candidate_reaction = None
        return state

    update_situation(state, user_message)

    new_label = classify_reaction_llm(user_message)
    update_reaction(state, new_label)

    if state.reaction_type == "hyperarousal" and state.intensity < 2:
        state.reaction_type = state.previous_reaction

    return state