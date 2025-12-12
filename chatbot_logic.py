import io
import time
import google.generativeai as genai
import requests
import streamlit as st
from PIL import Image

# --- API KEYS (edit for local use or switch to env vars) ---
# GEMINI_API_KEY = "PASTE_YOUR_GEMINI_KEY_HERE"
# HF_API_TOKEN = "PASTE_YOUR_HUGGING_FACE_TOKEN_HERE"

HF_MODELS = {
    "Stable Diffusion XL (Default)": "stabilityai/stable-diffusion-xl-base-1.0",
    "Stable Diffusion v3.5-large": "stabilityai/stable-diffusion-3.5-large",
    "Stable Diffusion v2.1": "stabilityai/stable-diffusion-2-1",
    "Stable Diffusion v1.4": "CompVis/stable-diffusion-v1-4",
    "FLUX.1-Krea (New)": "black-forest-labs/FLUX.1-Krea-dev"
}

HF_INFERENCE_BASE_URL = "https://router.huggingface.co/hf-inference/models"

def _query_hf_api(prompt: str, api_token: str, model_url: str):
    """Send a request to the Hugging Face Inference API."""

    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {"inputs": prompt}

    try:
        return requests.post(model_url, headers=headers, json=payload)
    except Exception as exc:  # pragma: no cover - Streamlit feedback path
        st.error(f"API request failed: {exc}")
        return None


def generate_image_hf(prompt: str, api_token: str, model_id: str, retries: int = 3):
    model_url = f"{HF_INFERENCE_BASE_URL}/{model_id}"

    for attempt in range(retries):
        st.write(f"🎨 Generating image (attempt {attempt + 1}/{retries})...")
        response = _query_hf_api(prompt, api_token, model_url)

        if response is None:
            st.error("❌ API request failed.")
            continue

        if response.status_code == 410:
            st.warning("🔁 Detected deprecated endpoint; retrying with the new Hugging Face router...")
            model_url = f"{HF_INFERENCE_BASE_URL}/{model_id}"
            continue

        if response.status_code == 200:
            try:
                image_bytes = response.content
                return Image.open(io.BytesIO(image_bytes))
            except Exception as exc:
                st.error(f"❌ Error processing image: {exc}")
                continue

        if response.status_code == 503:
            st.warning("⏳ Model is loading, please wait...")
            try:
                estimated_time = response.json().get("estimated_time", 20)
                time.sleep(min(estimated_time + 5, 60))
            except Exception:
                time.sleep(20)
            continue

        try:
            error_msg = response.json().get("error", "Unknown error")
            st.error(f"❌ Error {response.status_code}: {error_msg}")
        except Exception:
            st.error(f"❌ Error {response.status_code}: {response.text}")
        time.sleep(5)

    st.error("❌ Failed to generate image after all retries.")
    return None


def initialize_gemini(api_key: str):
    """Configure and initialize a Gemini chat session."""

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        return model.start_chat(history=[])
    except Exception as exc:  # pragma: no cover - Streamlit feedback path
        st.error(f"Failed to initialize Gemini: {exc}")
        return None


def get_gemini_response(chat_session, prompt: str, uploaded_files):
    """Send a message with optional files to the Gemini chat session."""

    try:
        content_to_send = list(uploaded_files)
        content_to_send.append(prompt)
        response = chat_session.send_message(content_to_send)
        return response.text
    except Exception as exc:  # pragma: no cover - Streamlit feedback path
        return f"Error sending message to Gemini: {exc}"
