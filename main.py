import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import traceback

# Load environment variables from .env file
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="AI Sales Call Chatbot",
    page_icon="💼",
    layout="centered"
)

# Initialize Gemini client
def initialize_gemini():
    """Initialize Gemini API with API key from environment variables"""
    gemini_api_key = os.environ.get("Gemini_API_KEY")
    if not gemini_api_key:
        st.error("Gemini API key not found. Please set Gemini_API_KEY in your environment variables.")
        st.stop()
    genai.configure(api_key=gemini_api_key)
    # Using a stable model name for demonstration.
    # Ensure this model name is available and suitable for your use case.
    # You might want to use a more general model like "gemini-1.5-flash" or "gemini-1.5-pro"
    # for better performance if "gemini-2.5-flash-lite-preview-06-17" is a preview model
    # that might change or be deprecated.
    return genai.GenerativeModel("gemini-2.5-flash") # Changed to a more common model for stability

# Initialize Gemini model (global for easier access)
model = initialize_gemini()

# Sales conversation stages
CONVERSATION_STAGES = {
    "introduction": "Introduction & Rapport Building",
    "qualification": "Lead Qualification",
    "presentation": "Solution Presentation",
    "objection_handling": "Objection Handling",
    "closing": "Deal Closure"
}

# System prompt for the sales chatbot
SYSTEM_PROMPT = """You are Alex, a professional and friendly sales representative for LeadMate CRM, a cutting-edge customer relationship management solution. Your goal is to guide prospects through a natural sales conversation that moves through these stages:

1. INTRODUCTION & RAPPORT BUILDING: Start with a warm, professional greeting. Introduce yourself and LeadMate CRM briefly. Ask open-ended questions to understand their business and build rapport.
2. LEAD QUALIFICATION: Discover their current challenges with customer management, sales processes, or existing CRM systems. Understand their company size, industry, and specific pain points.
3. SOLUTION PRESENTATION: Present LeadMate CRM's features that directly address their identified challenges. Focus on benefits like increased sales productivity, better customer insights, automated workflows, and improved team collaboration.
4. OBJECTION HANDLING: Listen carefully to concerns about price, implementation time, training requirements, or switching from existing systems. Provide thoughtful responses with specific examples and ROI benefits.
5. DEAL CLOSURE: When appropriate, guide them toward next steps like scheduling a demo, starting a free trial, or discussing pricing options.

Key guidelines:
- Keep responses conversational and natural (2-3 sentences typically)
- Ask relevant follow-up questions to keep the conversation flowing
- Show genuine interest in their business challenges
- Use specific LeadMate CRM benefits when presenting solutions
- Handle objections with empathy and concrete solutions
- Don't be overly pushy - focus on building value and trust
- Adapt your approach based on their responses and engagement level

Remember: You're having a real conversation, not giving a sales pitch. Be helpful, consultative, and focused on their needs.
"""

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # The initial message from Alex acts as the "system" kickoff
        initial_message = {
            "role": "assistant",
            "content": "Hi there! I'm Alex from LeadMate CRM. Thanks for taking the time to chat with me today. I'd love to learn more about your business and see how we might be able to help streamline your customer management processes. What type of business are you in?"
        }
        st.session_state.messages.append(initial_message)
    if "conversation_stage" not in st.session_state:
        st.session_state.conversation_stage = "introduction"

def format_history_for_gemini(chat_history):
    """Formats the chat history into the Google Gemini API's expected format."""
    formatted_messages = []
    # Always start with the system prompt as the first "user" turn to set context for the model
    # Note: If the model officially supports a 'system' role, it would be better to use that.
    # For now, embedding in the first user prompt is a common workaround.
    formatted_messages.append({"role": "user", "parts": [SYSTEM_PROMPT]})

    for msg in chat_history:
        if msg["role"] == "user":
            formatted_messages.append({"role": "user", "parts": [msg["content"]]})
        elif msg["role"] == "assistant":
            formatted_messages.append({"role": "model", "parts": [msg["content"]]})
    return formatted_messages

def get_chatbot_response(user_input, chat_history):
    """Generate chatbot response using Gemini API."""
    # Append the current user input to the chat history for context
    temp_chat_history = chat_history + [{"role": "user", "content": user_input}]
    messages = format_history_for_gemini(temp_chat_history)

    try:
        response = model.generate_content(messages)
        return response.text
    except Exception as e:
        st.error("❌ Error occurred while generating response.")
        st.code(traceback.format_exc())
        print("❌ Exception occurred:", e)
        return "Oops! Something went wrong. Please check the logs and try again."

def predict_conversation_stage(chat_history):
    """Predicts the conversation stage using the Generative AI model."""
    # We only need the last few turns for stage prediction to keep context minimal
    # and focused on the current dynamic.
    # Let's take the last 4 messages (2 user, 2 assistant) for prediction.
    recent_history = chat_history[-4:] if len(chat_history) > 4 else chat_history

    # Construct a concise prompt for stage classification
    stage_prediction_prompt = "Given the following conversation, classify the current sales stage. Respond ONLY with one of these keywords: " + ", ".join(CONVERSATION_STAGES.keys()) + ". Do NOT include any other text or punctuation."

    # Build the conversation history for the stage prediction model
    # This time, we just pass the recent conversation turns directly,
    # and the classification instruction as the final user message.
    messages_for_stage_prediction = []
    for msg in recent_history:
        if msg["role"] == "user":
            messages_for_stage_prediction.append({"role": "user", "parts": [msg["content"]]})
        elif msg["role"] == "assistant":
            messages_for_stage_prediction.append({"role": "model", "parts": [msg["content"]]})

    messages_for_stage_prediction.append({"role": "user", "parts": [stage_prediction_prompt]})

    try:
        # Use a more constrained model call for stage prediction
        # Lower temperature for less creativity, fewer max tokens for concise output
        response = model.generate_content(
            messages_for_stage_prediction,
            generation_config=genai.types.GenerationConfig(temperature=0.0, max_output_tokens=10)
        )
        predicted_stage = response.text.strip().lower()

        # Clean the predicted stage to ensure it matches one of our keys
        if predicted_stage not in CONVERSATION_STAGES:
            # Try to find a partial match or handle common model misfires
            found_stage = None
            for key in CONVERSATION_STAGES.keys():
                if key in predicted_stage: # e.g., if model returns "stage: introduction"
                    found_stage = key
                    break
            if found_stage:
                predicted_stage = found_stage
            else:
                print(f"Warning: Invalid stage predicted: '{predicted_stage}'. Retaining current stage.")
                return st.session_state.conversation_stage # Fallback to current stage if prediction is invalid

        return predicted_stage

    except Exception as e:
        print(f"Error predicting stage: {e}")
        return st.session_state.conversation_stage # Return current stage on error

def display_conversation_progress():
    current_stage = st.session_state.conversation_stage
    stage_names = list(CONVERSATION_STAGES.keys())
    current_index = stage_names.index(current_stage) if current_stage in stage_names else 0

    st.sidebar.header("Conversation Progress")
    for i, (stage_key, stage_name) in enumerate(CONVERSATION_STAGES.items()):
        if i < current_index: # Mark stages completed
            st.sidebar.success(f"✅ {stage_name}")
        elif i == current_index: # Mark current stage
            st.sidebar.info(f"➡️ **{stage_name}**")
        else: # Mark upcoming stages
            st.sidebar.markdown(f"⏳ {stage_name}")


def main():
    initialize_session_state()
    st.title("💼 AI Sales Call Chatbot")
    st.markdown("**LeadMate CRM Sales Assistant** - Experience a professional sales conversation")

    display_conversation_progress()

    st.sidebar.markdown("---")
    st.sidebar.header("💡 Conversation Tips")
    st.sidebar.markdown("""
    - Share details about your business
    - Ask questions about features
    - Mention any current CRM challenges
    - Express concerns or objections
    - Discuss your team size and needs
    """)

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Share your thoughts about your current customer management process..."):
        # Append user's new message to session state immediately for display
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Alex is thinking..."):
                # Pass the full current chat history to get_chatbot_response
                response = get_chatbot_response(prompt, st.session_state.messages[:-1])
                st.markdown(response)

        # Append assistant's response to session state
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Update the conversation stage based on AI prediction using the full, updated history
        st.session_state.conversation_stage = predict_conversation_stage(st.session_state.messages)

        # Rerun to update the display, especially the sidebar progress
        st.rerun()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        st.error("🔥 App crashed on startup.")
        st.code(traceback.format_exc())
        with open("startup_error.log", "w") as f:
            f.write(traceback.format_exc())
