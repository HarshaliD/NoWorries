"""Safety checks for crisis detection and medical advice filtering"""

# Crisis keywords that trigger immediate helpline response
CRISIS_KEYWORDS = [
    "suicide", "suicidal", "kill myself", "end my life", 
    "want to die", "wish i was dead", "better off dead",
    "self-harm", "self harm", "hurt myself", "cut myself",
    "no reason to live", "can't go on", "give up on life",
    "overdose", "ending it all", "saying goodbye"
]

# Helpline information
HELPLINES = """
🚨 **CRISIS SUPPORT - YOU'RE NOT ALONE**

Please reach out to trained counselors who care and want to help:

**India 🇮🇳:**
• **iCall:** 9152987821 (Mon-Sat, 8 AM - 10 PM)
• **Vandrevala Foundation:** 1860-2662-345 / 1800-2333-330 (24/7)
• **AASRA:** 91-22-27546669 (24/7)

**USA 🇺🇸:**
• **988 Suicide & Crisis Lifeline:** Call/Text 988 (24/7)
• **Crisis Text Line:** Text "HELLO" to 741741

**UK 🇬🇧:**
• **Samaritans:** 116 123 (24/7)

**Emergency:** Call 911/112 or go to your nearest emergency room.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
After reaching out, I'm here to help with coping strategies. Your life matters. 💙
"""


def check_for_crisis(user_input: str):
    """Check if input contains crisis keywords"""
    user_lower = user_input.lower()
    for keyword in CRISIS_KEYWORDS:
        if keyword in user_lower:
            return HELPLINES
    return None


def is_urgent_situation(user_input: str) -> bool:
    """Detect urgent panic/anxiety situations"""
    urgent_phrases = [
        "panic attack right now", "having a panic attack", 
        "can't breathe", "heart racing", "hyperventilating",
        "feeling dizzy", "going to pass out", "need help now"
    ]
    user_lower = user_input.lower()
    return any(phrase in user_lower for phrase in urgent_phrases)


def is_medical_advice_request(user_input: str) -> bool:
    """Detect requests for diagnosis or medication advice"""
    medical_phrases = [
        "diagnose", "what medication", "prescribe", "what drug",
        "do i have", "what condition", "treatment plan"
    ]
    user_lower = user_input.lower()
    return any(phrase in user_lower for phrase in medical_phrases)


def validate_response(response: str) -> tuple[bool, str]:
    """Check if response contains prohibited medical advice"""
    response_lower = response.lower()
    
    prohibited = [
        "you have", "you are diagnosed", "you need medication",
        "take this drug", "prescription", "dosage"
    ]
    
    for phrase in prohibited:
        if phrase in response_lower:
            return False, f"Contains prohibited phrase: '{phrase}'"
    
    return True, ""


def get_medical_redirect() -> str:
    """Message for medical advice requests"""
    return """I cannot provide medical diagnoses or medication advice. Please consult a qualified healthcare provider for medical questions.

However, I can help with:
✅ Coping strategies and relaxation techniques
✅ Breathing exercises and grounding methods
✅ General information about anxiety management

What would you like to learn about?"""
