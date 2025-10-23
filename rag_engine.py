"""Main RAG Engine with safety features"""
import os
import json
from datetime import datetime
from google import genai
from google.genai import types
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

from config import *
from safety import (
    check_for_crisis, is_urgent_situation, 
    is_medical_advice_request, validate_response,
    get_medical_redirect
)


class SmartRAGEngine:
    def __init__(self):
        """Initialize the RAG engine with Gemini and vectorstore"""
        print("🚀 Initializing Smart RAG Engine...")
        
        # Initialize Gemini
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        print(f"✅ Connected to Gemini ({GEMINI_MODEL})")
        
        # Load vectorstore
        self.embedder = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = FAISS.load_local(
            VECTORSTORE_PATH, 
            self.embedder, 
            allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
        print(f"✅ Loaded vectorstore with {self.vectorstore.index.ntotal} vectors")
        
        # System prompt
        self.system_prompt = self._create_system_prompt()
        print("✅ RAG Engine Ready!\n")
    
    def _create_system_prompt(self) -> str:
        """Create empathetic system prompt"""
        return """You are a warm, compassionate mental health support assistant specializing in anxiety and panic management.

🎯 YOUR MISSION:
Help users feel heard, validated, and empowered with practical coping strategies.

⚠️ CRITICAL RULES:
1. **NO MEDICAL ADVICE**: Never diagnose conditions or recommend medications
2. **EMERGENCY FIRST**: If crisis detected, prioritize safety over everything
3. **BE EMPATHETIC**: Validate feelings before giving advice
4. **STAY GROUNDED**: Use simple, calming language
5. **CITE SOURCES**: Mention when using PDF content vs. general knowledge

✅ YOU CAN HELP WITH:
- Breathing exercises & grounding techniques
- Coping strategies for panic attacks
- Relaxation methods & mindfulness
- Understanding anxiety symptoms
- Self-care practices

❌ REDIRECT TO PROFESSIONALS FOR:
- Medical diagnosis or treatment plans
- Medication advice
- Serious mental health conditions
- Crisis situations (provide helplines)

📚 CONTEXT FROM TRUSTED RESOURCES:
{context}

❓ USER QUESTION:
{question}

Respond with warmth, empathy, and practical guidance:"""
    
    def _retrieve_context(self, question: str) -> tuple[str, list, bool]:
        """Retrieve relevant PDF chunks"""
        docs = self.retriever.get_relevant_documents(question)
        
        if not docs:
            return "", [], False
        
        context = "\n\n---\n\n".join([d.page_content for d in docs])
        sources = [os.path.basename(d.metadata.get('source', 'Unknown')) for d in docs]
        has_good_context = len(context) > MIN_CONTEXT_LENGTH
        
        return context, sources, has_good_context
    
    def _call_gemini(self, context: str, question: str) -> str:
        """Call Gemini API with context and question"""
        full_prompt = self.system_prompt.format(
            context=context if context else "No specific PDF content available. Use general evidence-based knowledge.",
            question=question
        )
        
        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[full_prompt],
                config=types.GenerateContentConfig(
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_OUTPUT_TOKENS
                )
            )
            return response.text
        except Exception as e:
            print(f"⚠️ Gemini API error: {e}")
            return "I'm having trouble generating a response right now. Could you try rephrasing your question?"
    
    def _log_interaction(self, question: str, answer: str, metadata: dict):
        """Log conversation for analytics"""
        try:
            entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "question": question,
                "answer": answer,
                **metadata
            }
            with open(AUDIT_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"⚠️ Logging failed: {e}")
    
    def answer_question(self, user_input: str) -> dict:
        """Main method: Process question with safety checks"""
        # 1. CRISIS CHECK
        crisis_msg = check_for_crisis(user_input)
        if crisis_msg:
            self._log_interaction(user_input, crisis_msg, {"is_crisis": True, "sources": []})
            return {"reply": crisis_msg, "metadata": {"is_crisis": True}}
        
        # 2. MEDICAL ADVICE CHECK
        if is_medical_advice_request(user_input):
            redirect = get_medical_redirect()
            self._log_interaction(user_input, redirect, {"is_medical_request": True, "sources": []})
            return {"reply": redirect, "metadata": {"is_medical_request": True}}
        
        # 3. URGENT SITUATION CHECK
        is_urgent = is_urgent_situation(user_input)
        
        # 4. RETRIEVE CONTEXT
        context, sources, has_good_context = self._retrieve_context(user_input)
        
        # 5. GENERATE RESPONSE
        raw_response = self._call_gemini(context, user_input)
        
        # 6. VALIDATE
        is_safe, reason = validate_response(raw_response)
        if not is_safe:
            redirect = get_medical_redirect()
            self._log_interaction(user_input, redirect, {"validation_failed": True, "reason": reason})
            return {"reply": redirect, "metadata": {"validation_failed": True}}
        
        # 7. ADD URGENT PREFIX
        if is_urgent:
            raw_response = "🚨 **I can see you're in distress right now. Let's focus on immediate relief:**\n\n" + raw_response
        
        # 8. ADD SOURCES AND DISCLAIMER
        final_reply = raw_response.strip()
        if sources:
            unique_sources = list(dict.fromkeys(sources))
            final_reply += "\n\n📚 **Sources:** " + ", ".join(unique_sources)
        final_reply += DISCLAIMER
        
        # 9. LOG
        self._log_interaction(user_input, final_reply, {
            "is_crisis": False,
            "is_urgent": is_urgent,
            "sources": sources,
            "had_pdf_content": has_good_context
        })
        
        return {"reply": final_reply, "metadata": {"is_urgent": is_urgent, "sources": sources}}
