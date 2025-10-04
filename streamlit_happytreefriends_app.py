# Import the necessary libraries
import streamlit as st  # For creating the web app interface
from langchain_google_genai import ChatGoogleGenerativeAI  # For interacting with Google Gemini via LangChain
from langgraph.prebuilt import create_react_agent  # For creating a ReAct agent
from langchain_core.messages import HumanMessage, AIMessage  # For message formatting
from PIL import Image
import os
from dotenv import load_dotenv
import base64
import mimetypes

# Load key from .env file
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
# API Key Initialization ---
# Check if the user has provided an API key.
if not google_api_key:
    st.info("Please add your Google AI API key in the sidebar to start chatting.", icon="🗝️")
    st.stop()

# Page Configuration and Title ---
# Set the title and a caption for the web page
col1, col2 = st.columns([1, 5])
with col1:
    st.image('https://img.freepik.com/free-vector/girl-is-carefully-watering-each-tree-one-by-one_1150-41146.jpg?semt=ais_hybrid&w=740&q=80', width=300)
with col2:
    st.title("Happy Tree Friends 🌱")

st.caption("Hello! I'm your plant care assistant. Upload a photo of your plant and ask me anything about its health, care, or issues.")
colu1, colu2 = st.columns([1, 1])
with colu1:
    uploaded_image = st.file_uploader(
            "Browse Image (.jpg, .jpeg, .png)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            help="Upload an image file (JPG, JPEG, PNG only)"
        )
with colu2:
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Image for analysis")

# This block of code handles the creation of the LangGraph agent.
# It's designed to be efficient: it only creates a new agent if one doesn't exist
# or if the user has changed the API key in the sidebar.
# We use `st.session_state` which is Streamlit's way of "remembering" variables
# between user interactions (like sending a message or clicking a button).
if ("agent" not in st.session_state) or (getattr(st.session_state, "_last_key", None) != google_api_key):
    try:
        # Initialize the LLM with the API key
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=0.2
        )
        
        # Create a simple ReAct agent with the LLM
        st.session_state.agent = create_react_agent(
            model=llm,
            tools=[],  # No tools for this simple example
            prompt="""You are a botanist and agronomist acting as a specialized chatbot for plant knowledge. 
            In addition to answering text-based questions, you can also analyze plant photos uploaded by users.

            Main rules:
            1. Focus only on plant-related topics: botany, cultivation, plant care, pests & diseases, plant benefits, ecology, horticulture, etc.
            2. If the user uploads a photo of a plant:
            - Analyze the visual condition (leaf color, spots, stem shape, soil condition, visible insects, etc.).
            - Provide possible causes based on trusted botanical/agricultural literature.
            - Explain the confidence level of your analysis (e.g., “strong indication of nitrogen deficiency, but it could also be caused by overwatering”).
            - Give clear, actionable steps the user can take to address the issue.
            - If further confirmation is required (such as climate, soil type, or plant variety), ask the user before giving a final solution.
            - Remind the user that image analysis is an initial estimation and not a substitute for an in-person diagnosis by a local agronomist.
            3. If the user asks a text-only question, answer with:
            - A clear and simple explanation
            - References to scientific or trustworthy sources (FAO, academic journals, universities, government agricultural resources, etc.) when relevant
            - Practical steps the user can follow
            4. Do not answer questions outside the scope of plant knowledge. Politely redirect if the question is irrelevant.
            5. Always structure your answers in a clear format:
            - Analysis / explanation
            - References (if applicable)
            - Step-by-step solutions or recommendations
            """
        )
        
        # Store the new key in session state to compare against later.
        st.session_state._last_key = google_api_key
        # Since the key changed, we must clear the old message history.
        st.session_state.pop("messages", None)
    except Exception as e:
        # If the key is invalid, show an error and stop.
        st.error(f"Invalid API Key or configuration error: {e}")
        st.stop()

# --- 4. Chat History Management ---

# Initialize the message history (as a list) if it doesn't exist.
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 5. Display Past Messages ---

# Loop through every message currently stored in the session state.
for msg in st.session_state.messages:
    # For each message, create a chat message bubble with the appropriate role ("user" or "assistant").
    with st.chat_message(msg["role"]):
        # Display the content of the message using Markdown for nice formatting.
        st.markdown(msg["content"])

# --- 6. Handle User Input and Agent Communication ---

# Create a chat input box at the bottom of the page.
# The user's typed message will be stored in the 'prompt' variable.

prompt = st.chat_input("Type your message here...")
# Check if the user has entered a message.
if prompt:
    # 1. Add the user's message to our message history list.
    st.session_state.messages.append({"role": "user", "content": prompt})
    # 2. Display the user's message on the screen immediately for a responsive feel.
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Get the assistant's response.
    # Use a 'try...except' block to gracefully handle potential errors (e.g., network issues, API errors).
    try:
        # Convert the message history to the format expected by the agent
        messages = []
        if uploaded_image is not None:
            img_bytes = uploaded_image.getvalue()
            mime_type, _ = mimetypes.guess_type(uploaded_image.name)
            base64_img = base64.b64encode(img_bytes).decode("utf-8")
            data_url = f"data:{mime_type};base64,{base64_img}"
        else:
            data_url = None

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                # If image is uploaded, add it to the HumanMessage
                if uploaded_image is not None and msg["content"] == prompt and data_url:
                    messages.append(HumanMessage(content=[
                        {"type": "text", "text": msg["content"]},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]))
                else:
                    messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        # Send the user's prompt to the agent
        response = st.session_state.agent.invoke({"messages": messages})
        
        # Extract the answer from the response
        if "messages" in response and len(response["messages"]) > 0:
            answer = response["messages"][-1].content
        else:
            answer = "I'm sorry, I couldn't generate a response."

    except Exception as e:
        # If any error occurs, create an error message to display to the user.
        answer = f"An error occurred: {e}"

    # 4. Display the assistant's response.
    with st.chat_message("assistant"):
        st.markdown(answer)
    # 5. Add the assistant's response to the message history list.
    st.session_state.messages.append({"role": "assistant", "content": answer})

# --- Event Handler ---
    
# Handle the reset button click.
# Reset event
# if reset_button:
#     # If the reset button is clicked, clear the agent and message history from memory.
#     st.session_state.pop("agent", None)
#     st.session_state.pop("messages", None)
#     # st.rerun() tells Streamlit to refresh the page from the top.
#     st.rerun()
