import streamlit as st
import speech_recognition as sr
import asyncio
import edge_tts
from playsound import playsound
import os
from ollama import chat
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- PREMIUM DASHBOARD CONFIG ---
st.set_page_config(
    page_title=" Voice AI Agent",
    page_icon="⚡",
    layout="wide"
)

# --- MODERN DUAL-LOOK STYLING (CUSTOM CSS) ---
st.markdown("""
    <style>
    .main { background-color: #fafbfc; }
    
    /* User Chat Bubble - Clean & Minimalist White/Gray */
    .user-bubble {
        background-color: fffff;
        border-left: 5px solid #6366F1;
        padding: 15px;
        border-radius: 0px 15px 15px 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        margin-bottom: 15px;
    }
    
    /* Assistant Chat Bubble - Premium Soft Indigo/Blue Tint */
    .assistant-bubble {
        background-color: fffff;
        border-left: 5px solid #10B981;
        padding: 15px;
        border-radius: 0px 15px 15px 15px;
        box-shadow: 0 4px 6px rgba(99, 102, 241, 0.05);
        margin-bottom: 15px;
    }
    
    /* Button Styling */
    .stButton>button { 
        width: 100%; border-radius: 8px; height: 3.2em; 
        background-color: #4F46E5; color: white; font-weight: bold;
        border: none; transition: all 0.3s ease;
    }
    .stButton>button:hover { background-color: #4338CA; box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3); }
    
    /* Analytics Cards */
    .metric-card {
        background: brown; padding: 15px; border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02); border-left: 5px solid #4F46E5;
    }
    </style>
    """, unsafe_allow_html=True)

# --- BACKEND INITIALIZATION ---
@st.cache_resource
def load_assets():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("vectorstore", embedding, allow_dangerous_deserialization=True)
    return db

try:
    db = load_assets()
except Exception as e:
    st.error(f"Knowledge base connection failed.")

recognizer = sr.Recognizer()

# --- VOICE FUNCTION ---
async def speak(text):
    file = "voice.mp3"
    tts = edge_tts.Communicate(text=text, voice="en-US-AriaNeural")
    await tts.save(file)
    playsound(file)
    if os.path.exists(file):
        os.remove(file)

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.info("🟢 System: Fully Operational")
    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.query_count = 0
        st.rerun()

# --- MAIN DASHBOARD INTERFACE ---
st.title("⚡ Voice AI Agent ")
st.caption("Hybrid Context-Aware Intelligent Assistant")

# --- LIVE METRICS COUNTERS ---
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(f'<div class="metric-card"><strong>Queries Processed</strong><br><span style="font-size:24px; color:#4F46E5; font-weight:bold;">{st.session_state.query_count}</span></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric-card"><strong>System Engine</strong><br><span style="font-size:24px; color:#10B981; font-weight:bold;">Hybrid LLM</span></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric-card"><strong>Voice Mode</strong><br><span style="font-size:24px; color:#F59E0B; font-weight:bold;">Bi-Directional</span></div>', unsafe_allow_html=True)

st.write("---")

# --- CONVERSATION VIEW (ALAG ALAG LOOK WALA LOGIC) ---
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            # User look: White background with Indigo Border & Custom Emoji
            with st.chat_message("user", avatar="👤"):
                st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            # AI look: Light Purple/Blue background with Green Border & Robot Emoji
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f'<div class="assistant-bubble">{msg["content"]}</div>', unsafe_allow_html=True)

# --- INTERACTIVE CONTROLS ---
st.write(" ") 
input_col1, input_col2 = st.columns([1, 4])

voice_triggered = False
with input_col1:
    if st.button("🎙️ Tap to Speak"):
        voice_triggered = True

user_input = st.chat_input("Type your message here or use the mic...")

# Handle Microphone Stream
if voice_triggered:
    with st.spinner("🎙️ Listening actively... Speak now"):
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.4)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
                user_input = recognizer.recognize_google(audio)
        except Exception:
            st.warning("⚠️ Could not capture audio.")

# --- DYNAMIC EXECUTION LOGIC ---
if user_input:
    st.session_state.query_count += 1
    
    # 1. Update UI with User Query (User Custom Look)
    st.session_state.messages.append({"role": "user", "content": user_input})
    with chat_container:
        with st.chat_message("user", avatar="👤"):
            st.markdown(f'<div class="user-bubble">{user_input}</div>', unsafe_allow_html=True)

    # 2. Advanced Status Indicator
    with st.spinner("Analyzing and generating response... 🧠"):
        try:
            docs = db.similarity_search(user_input, k=3)
            context = "\n".join([d.page_content for d in docs])

            response = chat(model="llama3.2", messages=[
                {"role": "system", "content": "Direct, professional, natural answer. Do not use phrases like 'based on the context'. If unrelated to context, answer with general knowledge seamlessly."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{user_input}"}
            ])
            answer = response["message"]["content"]

            # 3. Stream Text (AI Custom Look)
            with chat_container:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(f'<div class="assistant-bubble">{answer}</div>', unsafe_allow_html=True)
            
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # 4. Trigger Voice Stream
            st.toast("🔊 Streaming audio output...", icon="🎵")
            asyncio.run(speak(answer))
            st.rerun()

        except Exception as e:
            st.error(f"Execution Error: {e}")