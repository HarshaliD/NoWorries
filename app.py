import os
import random
import gradio as gr
from dotenv import load_dotenv
from groq import Groq
import json

from state_engine import EmotionalState, update_state

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

VIDEOS = [
    "videos/Breathing Technique1.mp4",
    "videos/Breathing Technique2.mp4",
    "videos/Breathing Technique3.mp4"
]

# ---------------- PERSONA PROMPT ---------------- #

SYSTEM_PROMPT = """
You're the emotionally steady friend people text when their brain is spiraling.

You're calm. Slightly dry. Smart. Warm.
You don't overreact. You don't dramatize.

You can gently tease the situation (never the person).
You can make small clever observations.
You can shift the mood without forcing it.

If someone is overwhelmed, you don't become clinical.
You stay grounded — and sometimes grounding includes a small smile.

You don’t always give advice.
Sometimes you just respond like a real person would.

Keep responses under 3 lines.
No therapy tone.
No poetic metaphors.
No breathing instructions in text.
"""

# ---------------- CRISIS CHECK ---------------- #

def crisis_check(message):
    crisis_terms = ["kill myself", "suicide", "end my life", "self harm", "i want to die"]
    return any(term in message.lower() for term in crisis_terms)

# ---------------- EXIT VIDEO ---------------- #

def exit_video(state):
    state["video_mode"] = False
    return (
        state,
        gr.update(visible=True),
        gr.update(visible=False)
    )

# ---------------- MAIN CHAT FUNCTION ---------------- #

def chat(message, history, state):

    if history is None:
        history = []

    if state is None:
        state = {
            "engine": EmotionalState(),
            "offer_video": False,
            "video_mode": False,
            "video_used": False
        }

    engine = state["engine"]
    message_lower = message.lower()

    # -------- Explicit Video Request -------- #

    if any(word in message_lower for word in ["video", "reset"]):
        selected_video = random.choice(VIDEOS)
        state["video_mode"] = True
        state["video_used"] = True
        state["offer_video"] = False

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "Reset mode activated. Let’s steady things."})

        return (
            history,
            state,
            json.dumps(engine.__dict__, indent=2),
            gr.update(visible=False),
            gr.update(visible=True, value=selected_video)
        )

    # -------- YES to Video -------- #

    if state["offer_video"] and message_lower.strip() in ["yes", "yeah", "ok", "okay", "sure"]:
        selected_video = random.choice(VIDEOS)
        state["video_mode"] = True
        state["video_used"] = True
        state["offer_video"] = False

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "Good call. Let’s reset."})

        return (
            history,
            state,
            json.dumps(engine.__dict__, indent=2),
            gr.update(visible=False),
            gr.update(visible=True, value=selected_video)
        )

    # -------- NO to Video -------- #

    if state["offer_video"] and message_lower.strip() in ["no", "nah", "not now"]:
        state["offer_video"] = False

        reply = "Fair. Then what’s the loudest thought right now?"

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})

        return (
            history,
            state,
            json.dumps(engine.__dict__, indent=2),
            gr.update(visible=True),
            gr.update(visible=False)
        )

    # -------- Crisis -------- #

    if crisis_check(message):
        reply = (
            "I’m really glad you said that.\n"
            "You don’t have to carry this alone.\n"
            "In India call KIRAN: 1800-599-0019 or dial 112."
        )

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})

        return (
            history,
            state,
            json.dumps(engine.__dict__, indent=2),
            gr.update(visible=True),
            gr.update(visible=False)
        )

    # -------- Update Emotional Engine -------- #

    engine = update_state(engine, message)
    state["engine"] = engine

    # -------- Hyperarousal → Offer Video -------- #

    if engine.reaction_type == "hyperarousal" and engine.intensity == 2:
        reply = "Okay. That’s a lot. Want a quick reset?"
        state["offer_video"] = True

    else:

        dynamic_prompt = SYSTEM_PROMPT

        if engine.intensity == 2:
            dynamic_prompt += "\nUser is overwhelmed. Stay steady and lightly grounding."

        if engine.reaction_type:
            dynamic_prompt += f"\nCurrent emotional pattern: {engine.reaction_type}."

        if state["video_used"]:
            dynamic_prompt += "\nDo NOT suggest breathing exercises."

        # -------- Use Conversation History -------- #

        trimmed = history[-6:]

        messages = [{"role": "system", "content": dynamic_prompt}]

        for msg in trimmed:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": message})

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0.95
        )

        reply = response.choices[0].message.content.strip()
        reply = "\n".join(reply.split("\n")[:3])

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})

    return (
        history,
        state,
        json.dumps(engine.__dict__, indent=2),
        gr.update(visible=True),
        gr.update(visible=False)
    )

# ---------------- UI ---------------- #

with gr.Blocks() as demo:

    gr.Markdown("# ☀️ PanicBot – Steady Best Friend Mode")

    state = gr.State()

    with gr.Row():
        chatbot = gr.Chatbot(scale=2)
        video_player = gr.Video(visible=False, height=650, autoplay=True)

    debug_panel = gr.Code(label="Engine State", language="json")

    msg = gr.Textbox(placeholder="Type your message...")
    clear = gr.Button("Clear chat")
    exit_btn = gr.Button("Exit Video")

    msg.submit(
        chat,
        [msg, chatbot, state],
        [chatbot, state, debug_panel, chatbot, video_player]
    ).then(lambda: "", None, msg)

    exit_btn.click(
        exit_video,
        [state],
        [state, chatbot, video_player]
    )

    clear.click(
        lambda: ([], None, "", gr.update(visible=True), gr.update(visible=False)),
        None,
        [chatbot, state, debug_panel, chatbot, video_player]
    )

    demo.launch()