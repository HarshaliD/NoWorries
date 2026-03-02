# state_engine.py

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

    def __repr__(self):
        return (
            f"EmotionalState("
            f"situation_summary={self.situation_summary}, "
            f"reaction_type={self.reaction_type}, "
            f"intensity={self.intensity}, "
            f"candidate_reaction={self.candidate_reaction})"
        )


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


# ---------------- LLM Reaction Classification ---------------- #

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

    except Exception as e:
        print("Classifier error:", e)
        return "none"


# ---------------- Sticky Reaction Logic ---------------- #

def update_reaction(state: EmotionalState, new_label: str):

    if new_label == "none":
        return

    # First activation
    if state.reaction_type is None:
        state.reaction_type = new_label
        state.intensity = max(state.intensity, 1)
        return

    # Stable
    if new_label == state.reaction_type:
        state.candidate_reaction = None
        return

    # Switch requires confirmation
    if state.candidate_reaction == new_label:
        state.reaction_type = new_label
        state.candidate_reaction = None
    else:
        state.candidate_reaction = new_label


# ---------------- Situation Summary ---------------- #

def update_situation(state: EmotionalState, text: str):
    text = text.lower()

    if "exam" in text:
        state.situation_summary = "Exam upcoming"

    elif "interview" in text:
        state.situation_summary = "Interview upcoming"

    elif "job" in text:
        state.situation_summary = "Career transition"


# ---------------- Main Update ---------------- #

def update_state(state: EmotionalState, user_message: str):

    # Hyperarousal override
    if detect_hyperarousal(user_message):
        state.previous_reaction = state.reaction_type
        state.reaction_type = "hyperarousal"
        state.intensity = 2
        state.candidate_reaction = None
        return state

    update_situation(state, user_message)

    new_label = classify_reaction_llm(user_message)
    update_reaction(state, new_label)

    # Return from hyperarousal
    if state.reaction_type == "hyperarousal" and state.intensity < 2:
        state.reaction_type = state.previous_reaction

    return state