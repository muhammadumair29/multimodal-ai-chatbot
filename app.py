import streamlit as st
import io
import google.generativeai as genai
from PIL import Image

from chatbot_logic import (
    HF_MODELS,
    generate_image_hf,
    initialize_gemini,
    get_gemini_response,
)


# --- Page Configuration ---
st.set_page_config(
    page_title="Multimodal AI Chatbot",
    page_icon="🤖",
    layout="wide",
)

# --- Load API Keys from Secrets ---
# This replaces the manual input. Streamlit automatically looks in .streamlit/secrets.toml
try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    hf_api_token = st.secrets["HF_API_TOKEN"]
except FileNotFoundError:
    st.error("Secrets file not found. Please create .streamlit/secrets.toml")
    st.stop()
except KeyError as e:
    st.error(f"Missing key in secrets file: {e}")
    st.stop()

 # --- Non-UI logic is imported from chatbot_core ---

# --- Streamlit UI ---

# st.title("🤖 Multimodal AI Chatbot")
# st.caption("Powered by Gemini 1.5 & Hugging Face Stable Diffusion")

# --- Custom Header ---
st.markdown("""
<style>
    /* --- 1. FONTS & IMPORTS --- */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');

    /* --- 2. GLOBAL VARIABLES --- */
    :root {
        --bg-deep: #02040a;
        --bg-panel: #0d1117;
        --neon-cyan: #00f3ff;
        --neon-pink: #bc13fe;
        --neon-blue: #0066ff;
        --text-main: #e6edf3;
    }

    /* --- 3. BACKGROUND & RESET --- */
    /* Main Background */
    .stApp {
        background-color: var(--bg-deep);
        background-image: 
            radial-gradient(circle at 15% 50%, rgba(0, 243, 255, 0.08), transparent 25%),
            radial-gradient(circle at 85% 30%, rgba(188, 19, 254, 0.08), transparent 25%);
        background-attachment: fixed;
    }
    
    /* Global Text */
    * {
        font-family: 'Rajdhani', sans-serif;
        color: var(--text-main);
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Orbitron', sans-serif;
        color: white !important;
    }

    /* --- 4. HEADER & FOOTER FIXES --- */
    /* Remove default white header decoration */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* FIX: Force the Sticky Bottom Container to be Dark */
    [data-testid="stBottom"] {
        background-color: var(--bg-deep) !important; 
        border-top: 1px solid rgba(0, 243, 255, 0.2);
    }
    
    /* FIX: This is the stable version of the code you found via inspect */
    /* It targets the inner container of the bottom bar */
    [data-testid="stBottom"] > div {
        background-color: var(--bg-deep) !important;
    }

    /* --- 5. CUSTOM HEADER COMPONENT --- */
    .cyber-header {
        background: rgba(13, 17, 23, 0.7);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(0, 243, 255, 0.3);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 0 20px rgba(0, 243, 255, 0.1);
    }
    .cyber-header h1 {
        background: linear-gradient(90deg, #fff, var(--neon-cyan), #fff);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 5s linear infinite;
        font-size: 3rem;
        font-weight: 900;
        text-transform: uppercase;
        margin: 0;
    }
    .cyber-header p {
        color: var(--neon-pink) !important;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    @keyframes shine { to { background-position: 200% center; } }

    /* --- 6. SIDEBAR & UPLOADER FIXES --- */
    [data-testid="stSidebar"] {
        background-color: var(--bg-panel);
        border-right: 1px solid rgba(0, 243, 255, 0.2);
    }
    
    /* File Uploader Container */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background-color: transparent !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        background-color: rgba(188, 19, 254, 0.05) !important;
        border: 1px dashed var(--neon-cyan) !important;
    }
    
    /* FIX: "Browse Files" Button */
    [data-testid="stFileUploader"] button {
        background-color: var(--bg-panel) !important;
        color: white !important;
        border: 1px dashed var(--neon-cyan) !important; 
    }
    [data-testid="stFileUploader"] button:hover {
        border-color: var(--neon-cyan) !important;
        color: var(--neon-cyan) !important;
    }
    
    /* Dropdown Styling */
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: #0b0f19 !important;
        color: 0d1117 !important;
        border: 1px solid #30363d !important;
    }
    [data-testid="stSidebar"] .stSelectbox svg { fill: white !important; }
            

    /* --- 7. CHAT MESSAGE STYLING --- */
    [data-testid="stChatMessage"] {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
    }
    [data-testid="stChatMessage"]:has(div[data-testid="stChatMessageContent"][class*="user-"]) {
        background: linear-gradient(90deg, rgba(0, 243, 255, 0.05), transparent);
        border-left: 3px solid var(--neon-cyan);
    }
    [data-testid="stChatMessage"]:has(div[data-testid="stChatMessageContent"][class*="assistant-"]) {
        background: linear-gradient(90deg, rgba(188, 19, 254, 0.05), transparent);
        border-left: 3px solid var(--neon-pink);
    }

    /* --- 8. CHAT INPUT BAR FIXES --- */
    /* Remove background from the input wrapper */
    [data-testid="stChatInput"] {
        background-color: transparent !important;
    }
    
    /* Style the actual input box */
    [data-testid="stChatInput"] > div {
        background-color: #0d1117 !important;
        border: 1px solid var(--neon-cyan) !important;
        color: white !important;
    }
    
    /* Input Text Color Fix */
    [data-testid="stChatInput"] textarea {
        background-color: #0d1117 !important;
        color: white !important;
        caret-color: var(--neon-cyan);
    }
    
    /* Placeholder Color */
    [data-testid="stChatInput"] textarea::placeholder {
        color: rgba(255, 255, 255, 0.5) !important;
    }
    
    /* Send Button Color */
    [data-testid="stChatInput"] button {
        color: var(--neon-cyan) !important;
    }

    /* --- 9. MISC --- */
    img { border: 1px solid var(--neon-cyan); box-shadow: 0 0 10px rgba(0, 243, 255, 0.2); }
</style>

<div class="cyber-header">
    <h1>Multimodal AI Chatbot</h1>
    <p>Powered by Gemini 2.5 & Hugging Face Stable Diffusion</p>
</div>
""", unsafe_allow_html=True)
# --- Sidebar for Settings ---
with st.sidebar:
    st.header("🛠️ Settings")
    
    # API Keys
    # gemini_api_key = st.text_input("Gemini API Key", type="password")
    # hf_api_token = st.text_input("Hugging Face API Token", type="password")

    # st.divider()

    

    # Model Selection
    st.subheader("🖼️ Image Generation")
    selected_hf_model_name = st.selectbox(
        "Choose a Stable Diffusion Model",
        options=list(HF_MODELS.keys())
    )
    # Get the corresponding model ID from the user-friendly name
    selected_hf_model_id = HF_MODELS[selected_hf_model_name]

    st.divider()

  

    st.subheader("🖼️ Image Upload")
    uploaded_image = st.file_uploader(
        "Upload an image for Gemini to analyze",
        type=["png", "jpg", "jpeg"]
    )

    if not gemini_api_key:
        st.warning("Please enter your Gemini API Key to start.")
    if not hf_api_token:
        st.warning("Please enter your Hugging Face Token for image generation.")

# --- Main Chat Interface ---

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "gemini_chat" not in st.session_state:
    st.session_state.gemini_chat = None

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("type") == "image":
            st.image(message["content"], caption=message.get("caption", None))
        else:
            st.markdown(message["content"])

# Main logic: Check for API key first
if gemini_api_key:
    # Initialize chat session once
    if st.session_state.gemini_chat is None:
        st.session_state.gemini_chat = initialize_gemini(gemini_api_key)
    
    # Check if initialization was successful
    if st.session_state.gemini_chat:
        
        
        # Get user prompt from chat input
        if prompt := st.chat_input("Ask or describe an image..."):
            
            # Convert prompt to lowercase for easy checking
            lower_prompt = prompt.lower()
            
            # List of trigger keywords. We'll check if the prompt starts with any of them.
            trigger_keywords = [
                "draw:", 
                "draw", 
                "draw an image:",
                "draw an image",
                "generate image:", 
                "generate image", 
                "create image:", 
                "create image",
                "create an image:",
                "create an image",
                "generate an image:",
                "generate an image",
                "generate an image:",
                "generate an image",
            ]
            
            matched_keyword = None
            for keyword in trigger_keywords:
                if lower_prompt.startswith(keyword):
                    matched_keyword = keyword
                    break # Found the longest matching keyword

            # 1. Check if an Image Generation command was found
            if matched_keyword:
                if not hf_api_token:
                    st.error("Please enter your Hugging Face API Token in the sidebar to generate images.")
                else:
                    # Extract the *actual* prompt text after the keyword
                    # We use len(matched_keyword) to cut it out from the *original* prompt
                    image_prompt = prompt[len(matched_keyword):].strip()
                    
                    # Also strip any leading colon just in case it wasn't part of the keyword
                    image_prompt = image_prompt.lstrip(":").strip() 
                    
                    # Handle empty prompt after keyword
                    if not image_prompt:
                        st.warning("Please describe the image you want to generate after the keyword (e.g., 'draw a cat').")
                        # Stop execution for this run and wait for new input
                        st.stop()
                    
                    # Add user prompt to chat
                    st.chat_message("user").markdown(prompt)
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    
                    # Generate image
                    with st.chat_message("assistant"):
                        with st.spinner(f"Generating: {image_prompt}..."):
                            generated_image = generate_image_hf(
                                image_prompt, 
                                hf_api_token, 
                                selected_hf_model_id
                            )
                        
                        if generated_image:
                            st.image(generated_image, caption=f"Generated from: {selected_hf_model_id}")
                            # Add image to history (store the PIL object)
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "type": "image", 
                                "content": generated_image,
                                "caption": image_prompt
                            })
                        else:
                            st.error("Image generation failed.")
                            st.session_state.messages.append({"role": "assistant", "content": "Image generation failed."})

            # 2. Default to Gemini for text, audio, and vision
            else:
                # (This 'else' block for Gemini remains exactly the same as before)
                
                # Add user prompt to chat
                st.chat_message("user").markdown(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})

                # Prepare files for Gemini
                files_to_send = []
                
                # a) Add uploaded image
                if uploaded_image:
                    img = Image.open(uploaded_image)
                    files_to_send.append(img)
                    with st.chat_message("user"):
                         st.image(img, caption="You uploaded this image.", width=250)
                
                
                # Get response from Gemini
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response_text = get_gemini_response(
                            st.session_state.gemini_chat,
                            prompt,
                            files_to_send
                        )
                        st.markdown(response_text)
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
else:
    st.info("Please enter your Gemini API Key in the sidebar to begin the chat.")