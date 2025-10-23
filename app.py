"""Gradio chatbot interface"""
import gradio as gr
from rag_engine import SmartRAGEngine

# Initialize engine
print("Starting Panic & Anxiety Support Chatbot...\n")
engine = SmartRAGEngine()

def chat_response(message, history):
    """Process user message and return response"""
    if not message.strip():
        return history
    
    result = engine.answer_question(message)
    reply = result["reply"]
    
    history.append((message, reply))
    return history

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧘 Panic & Anxiety Support Assistant
    
    A compassionate AI assistant to help you manage anxiety and panic with evidence-based techniques.
    
    **This is NOT a replacement for professional care.** In crisis, contact emergency services or helplines.
    """)
    
    chatbot = gr.Chatbot(value=[], height=500, show_label=False)
    
    with gr.Row():
        msg_input = gr.Textbox(
            placeholder="Type your message here... (e.g., 'I'm having a panic attack right now')",
            show_label=False,
            scale=9
        )
        submit_btn = gr.Button("Send", scale=1, variant="primary")
    
    clear_btn = gr.Button("Clear Chat")
    
    gr.Markdown("""
    ### 💡 Try asking:
    - "What are grounding techniques?"
    - "I'm having a panic attack right now"
    - "How can I calm down quickly?"
    """)
    
    msg_input.submit(chat_response, [msg_input, chatbot], [chatbot]).then(lambda: "", None, msg_input)
    submit_btn.click(chat_response, [msg_input, chatbot], [chatbot]).then(lambda: "", None, msg_input)
    clear_btn.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
