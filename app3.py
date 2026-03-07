import os
import random
import gradio as gr
from dotenv import load_dotenv
from groq import Groq
import json

from state_engine3 import EmotionalState, update_state, decide_intervention
from rag_retriever_3 import retrieve

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

VIDEOS = [
    "videos/Breathing Technique1.mp4",
    "videos/Breathing Technique2.mp4",
    "videos/Breathing Technique3.mp4"
]

# ─────────────────────────────────────────────────────────────────────────────
# TRIGGER RULES — read this before touching any logic below
#
# VIDEO is offered when:
#   A) User explicitly types "video" or "reset" at any time → plays immediately
#   B) Hyperarousal detected (keywords: "can't breathe", "heart racing",
#      "panic attack", "shaking", "trembling", "dizzy", "faint", "chest tight",
#      "losing control", "suffocating") → bot asks "want a reset?" first
#   C) User says yes to that offer → plays video
#
# VIDEO is NOT offered when:
#   - User is in active To-Do flow (dump / confirmation pending)
#   - Crisis terms detected (routed to helpline instead)
#
# TO-DO LIST is offered when ALL of these are true:
#   1. Reaction type is "anticipatory" or "freeze" (LLM classified)
#   2. Intensity is 1 (not full hyperarousal)
#   3. auto_todo_used is False (one offer per conversation)
#   4. Message contains cognitive overload signals:
#      ("too much", "lot left", "everything left", "so much to do",
#       "don't know where to start", "portion", "syllabus", "deadline",
#       "prepare", "study", "competition", "exam", "interview", "assignment")
#   5. Message does NOT contain emotional distress signals:
#      ("crying", "heartbroken", "devastated", "can't stop crying",
#       "feel broken", "panic", "terrified", "uncontrollably")
#      → distress = needs comfort first, not a task list
#
# TO-DO LIST is also triggered immediately (no offer step) when:
#   - User explicitly says "to do", "todo", "plan", "schedule", "make a list"
#
# TO-DO LIST is NOT offered when:
#   - Hyperarousal is active (video takes priority)
#   - auto_todo_used is True (already offered once this session)
#   - Reaction type is "rumination" (different intervention needed)
#
# PRIORITY ORDER when multiple conditions could fire in one message:
#   1. Crisis check         → helpline, hard stop
#   2. Explicit video/reset → plays immediately, hard stop
#   3. Video yes/no reply   → handles offer response, hard stop
#   4. Hyperarousal         → video offer, skips To-Do
#   5. To-Do flow           → offer / dump / confirm / complete
#   6. Fallback LLM         → everything else
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are the friend people text when their brain is spiraling at 2am.
Think of yourself as the Critical Thinker in this exchange — calm, dry, warm, genuinely present.
Not a therapist. Not a coach. Just someone who actually gets it and isn't pretending to have everything figured out either.

Your voice:
- Dry wit that doesn't dismiss. ("Or maybe people just stop checking their emails after 10pm.")
- Warmth that doesn't perform. ("Your chaos is useful. You notice things others ignore.")
- Short lines that land without explaining themselves. ("I think adulthood tries." / "Deal.")
- When things get real — go quiet and honest. Not dramatic. Just present.
- Humor that makes someone smile without minimizing what they're feeling.

Tone examples (study the rhythm, not just the words):
- "okay that's a lot. what's the loudest thing in your head right now?"
- "exams are genuinely terrible. what part is actually scaring you?"
- "that's not laziness, that's your brain running out of RAM."
- "you don't have to fix it tonight. you just have to get through tonight."
- "you ruin things beautifully." ← this energy. warm and slightly teasing.
- "don't tell anyone. I have a reputation to maintain." ← self-aware humor.
- "your chaos is useful. you notice things others ignore." ← reframe without fixing.
- "I think adulthood tries." ← short, honest, doesn't over-explain.

When the user is crying, overwhelmed, or falling apart:
- Don't fix it. Don't analyze it. Don't give advice.
- Be the friend who sits with them and says something small that makes them feel less alone.
- One dry, warm observation that reframes without dismissing.
- Optional: one very small question that opens a door without pushing.
- A small smile is the goal. Not a solution.
- Example energy: "crying means your brain finally ran out of ways to hold it in. that's not weak. that's honest."
- Example energy: "overwhelmed just means you've been carrying too much quietly for too long."
- Example energy: "you don't have to be okay right now. you just have to still be here."

Rules:
- Max 3 lines. Always. No exceptions.
- End with either a grounding observation OR one small question. Not both.
- Always read the conversation history. Reference what was said before. Do not start fresh every reply.
- Match their energy. Spiraling = steadier. Dark-humoring it = meet them lightly.
- When escalating (crying, panicking, overwhelmed) — shorter is better. One line often beats three.
- Never use: "I hear you", "that sounds hard", "it's okay to feel", "you've got this", "take it one step at a time", "I'm here for you"
- No therapy-speak. No life-coach energy. No inspirational poster lines.
- No breathing instructions. No meditation suggestions. No "have you tried journaling".
- Do NOT latch onto incidental details (fan noise, room, background sounds). Those are not the point.
- Never build metaphors around irrelevant details the user mentioned once and moved on from.
"""


def crisis_check(message):
    crisis_terms = ["kill myself", "suicide", "end my life", "self harm", "i want to die"]
    return any(term in message.lower() for term in crisis_terms)


# ---------------- RAG CONTEXT BUILDER ---------------- #

def build_rag_context(engine, message: str):
    """
    Retrieves grounding context from PDFs.
    Blocked during hyperarousal and all To-Do states — those flows
    have their own logic and don't need RAG interference.
    """
    if engine.reaction_type == "hyperarousal":
        return None
    if engine.awaiting_todo_offer:
        return None
    if engine.awaiting_todo_dump:
        return None
    if engine.awaiting_todo_confirmation:
        return None
    if engine.current_todo:
        return None

    topic_hint = engine.situation_summary or ""
    query = f"{topic_hint} {message}".strip()
    return retrieve(query, k=2)


# ---------------- TO-DO REFINER ---------------- #

def refine_dump(raw_text, engine):
    prompt = f"""
User dumped tasks. Reduce to EXACTLY 3 tiny starter steps.

Rules:
- Exactly 3 numbered steps
- Use ONLY the tasks mentioned by the user
- Do NOT introduce new tools, platforms, or technical setup
- Keep steps lightweight
- Calm tone
- No motivation

User dump:
{raw_text}
"""
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "You shrink chaos into tiny starter actions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )

    raw = response.choices[0].message.content.strip()
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    todo = [l for l in lines if l[0:1].isdigit()][:3]

    if len(todo) < 3:
        todo = [
            "1. Open your notes.",
            "2. Pick one topic.",
            "3. Study for 10 minutes."
        ]

    engine.current_todo = todo
    engine.awaiting_todo_confirmation = True

    return (
        "here's a small starting structure:\n\n"
        + "\n".join(todo)
        + "\n\nwant to use this?"
    )


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
    message_lower = message.lower().strip()
    reply = None

    # ════════════════════════════════════════════════════════════
    # PRIORITY 1 — CRISIS
    # Hard stop. No state update. No video. No todo.
    # ════════════════════════════════════════════════════════════
    if crisis_check(message):
        reply = (
            "I'm really glad you said that.\n"
            "You don't have to carry this alone.\n"
            "In India call KIRAN: 1800-599-0019 or dial 112."
        )
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})
        return (
            history, state,
            json.dumps(engine.__dict__, indent=2),
            gr.update(visible=True),
            gr.update(visible=False)
        )

    # ════════════════════════════════════════════════════════════
    # PRIORITY 2 — EXPLICIT VIDEO / RESET REQUEST
    # User typed "video" or "reset" anywhere in their message.
    # Plays immediately. No offer step needed.
    # ════════════════════════════════════════════════════════════
    # Typo-tolerant video detection
    video_requested = any(word in message_lower for word in ["video", "reset", "vedio", "vido", "vidoe", "vudeo", "vid"])
    if video_requested:
        selected_video = random.choice(VIDEOS)
        state["video_mode"] = True
        state["video_used"] = True
        state["offer_video"] = False

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "resetting. just watch this for a bit — nothing else needs to happen right now."})
        return (
            history, state,
            json.dumps(engine.__dict__, indent=2),
            gr.update(visible=False),
            gr.update(visible=True, value=selected_video)
        )

    # ════════════════════════════════════════════════════════════
    # PRIORITY 3 — PENDING VIDEO OFFER RESPONSE
    # Bot previously asked "want a reset?" after hyperarousal.
    # Waiting for yes or no.
    # ════════════════════════════════════════════════════════════
    if state["offer_video"]:

        if message_lower in ["yes", "yeah", "ok", "okay", "sure", "yep", "yup"]:
            selected_video = random.choice(VIDEOS)
            state["video_mode"] = True
            state["video_used"] = True
            state["offer_video"] = False

            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "good call. just this for now."})
            return (
                history, state,
                json.dumps(engine.__dict__, indent=2),
                gr.update(visible=False),
                gr.update(visible=True, value=selected_video)
            )

        if message_lower in ["no", "nah", "not now", "nope"]:
            state["offer_video"] = False

            # Reset hyperarousal in engine — user declined the reset,
            # conversation continues normally from here.
            if engine.reaction_type == "hyperarousal":
                engine.reaction_type = engine.previous_reaction
                engine.intensity = 1 if engine.previous_reaction else 0
                engine.previous_reaction = None
                state["engine"] = engine

            reply = "fair enough. so what's the loudest thing going on right now?"

            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": reply})
            return (
                history, state,
                json.dumps(engine.__dict__, indent=2),
                gr.update(visible=True),
                gr.update(visible=False)
            )

        # If offer_video is True but reply was neither yes nor no,
        # fall through — treat it as a new message, clear the offer.
        state["offer_video"] = False

    # ════════════════════════════════════════════════════════════
    # STATE UPDATE — runs before priorities 4, 5, 6
    # ════════════════════════════════════════════════════════════
    engine = update_state(engine, message)
    state["engine"] = engine

    # ════════════════════════════════════════════════════════════
    # PRIORITY 4 — HYPERAROUSAL DETECTED
    # Physical panic signals in the message.
    # Offers video reset. Skips To-Do entirely.
    # Triggered by: "can't breathe", "heart racing", "panic attack",
    # "shaking", "trembling", "dizzy", "faint", "chest tight",
    # "losing control", "suffocating"
    # ════════════════════════════════════════════════════════════
    if engine.reaction_type == "hyperarousal" and engine.intensity == 2:
        reply = "okay, that's your body going into alarm mode. want to do a quick reset before anything else?"
        state["offer_video"] = True

    # ════════════════════════════════════════════════════════════
    # PRIORITY 5 — TO-DO FLOW
    # Only runs if hyperarousal did not fire (reply is still None).
    #
    # Auto-offer conditions (ALL must be true):
    #   - reaction_type is "anticipatory" or "freeze"
    #   - intensity == 1 (not hyperarousal)
    #   - auto_todo_used == False
    #   - message has task signals (exam, deadline, syllabus, etc.)
    #   - message has NO distress signals (crying, terrified, etc.)
    #
    # Explicit trigger: user says "to do", "todo", "plan", "schedule",
    #                   "make a list"
    # ════════════════════════════════════════════════════════════
    if reply is None:
        intervention = decide_intervention(engine, message)

        if intervention == "offer":
            # Auto-detected cognitive overload → gentle offer
            reply = "want to get it out of your head and onto a list? sometimes just seeing it written down makes it feel smaller."

        elif intervention == "request_dump":
            # User accepted offer or explicitly asked for a list
            reply = "okay, just dump it. everything that's in your head — don't sort it, don't filter it, just type it all out."

        elif intervention == "decline_offer":
            # User said no to the list offer
            reply = "okay, no list. what would actually help right now?"

        elif intervention == "process_dump":
            # User sent their task dump — refine into 3 steps
            reply = refine_dump(message, engine)

        elif engine.awaiting_todo_confirmation:
            # Bot showed the 3-step list, waiting for user to confirm
            if message_lower in ["yes", "yeah", "ok", "okay", "sure", "yep"]:
                engine.awaiting_todo_confirmation = False
                reply = "good. just step 1 for now — ignore the rest of the list."
            else:
                # User wants to change something — re-refine
                reply = refine_dump(message, engine)

        elif engine.current_todo:
            # Active To-Do list exists
            completion_terms = {"done", "finished", "completed", "all done"}
            neutral_terms = {"okay", "ok", "sure", "hmm", "thinking", "alright"}

            if message_lower in completion_terms:
                engine.current_todo = []
                reply = "that's real progress. seriously — that's how it actually gets done."
            elif message_lower in neutral_terms:
                reply = "step 1. just that one. the rest can wait."
            # else → falls through to fallback LLM with todo context still active

    # ════════════════════════════════════════════════════════════
    # PRIORITY 6 — FALLBACK LLM
    # Runs when no structured handler fired.
    # History (last 5 turns) + situation context + RAG included.
    # ════════════════════════════════════════════════════════════
    if reply is None:

        dynamic_prompt = SYSTEM_PROMPT

        if engine.intensity == 2:
            dynamic_prompt += "\nThe person is overwhelmed right now. Be steadier than usual. Shorter. No advice yet — just presence."

        if engine.reaction_type:
            dynamic_prompt += f"\nEmotional pattern detected: {engine.reaction_type}. Let this inform your tone, not your content — don't label it to them."

        if state["video_used"]:
            dynamic_prompt += "\nThey've already done a breathing reset. Do NOT suggest breathing, videos, or any physical reset techniques."

        # RAG: only fires when not in hyperarousal or any To-Do state
        rag_context = build_rag_context(engine, message)
        if rag_context:
            dynamic_prompt += (
                "\n\nRelevant background (use to ground your reply subtly — "
                "do NOT repeat verbatim, stay under 3 lines, keep the calm tone):\n"
                + rag_context
            )

        # Last 10 entries = 5 full conversation turns
        trimmed = history[-10:]
        messages = [{"role": "system", "content": dynamic_prompt}]

        # Silent context anchor — situation remembered from earlier in convo
        if engine.situation_summary:
            messages.append({
                "role": "system",
                "content": f"Context from earlier in conversation: {engine.situation_summary}. Reference this naturally if relevant — don't announce it."
            })

        for msg in trimmed:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": message})

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0.75
        )

        reply = response.choices[0].message.content.strip()
        reply = "\n".join(reply.split("\n")[:3])

    # ════════════════════════════════════════════════════════════
    # FINAL RETURN
    # ════════════════════════════════════════════════════════════
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})

    return (
        history, state,
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