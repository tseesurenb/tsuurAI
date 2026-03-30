"""
TsuurAI Chat App
Multi-turn conversation with local LLMs and OpenAI
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
SRC_DIR = Path(__file__).parent.parent.resolve()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.config import LOCAL_LLM_INFO, API_LLM_INFO
from common.models import load_local_llm
from common.llm import openai_client, chat_with_local_llm, chat_with_openai
from common.auth import show_login_page, show_user_sidebar, log_usage

st.set_page_config(page_title="TsuurAI - Chat", page_icon="💬", layout="wide")

# Check authentication
if not st.session_state.get("authenticated"):
    show_login_page()
    st.stop()

st.title("💬 TsuurAI Chat")
st.write("Chat with local LLMs or OpenAI models")

# Show user info in sidebar
show_user_sidebar()

# Sidebar configuration
with st.sidebar:
    st.header("Chat Configuration")

    # LLM type selection
    llm_type = st.radio(
        "LLM Type",
        ["Local LLM (GPU)", "OpenAI API"],
        index=0,
        horizontal=True,
    )

    if llm_type == "OpenAI API":
        use_local_llm = False
        if openai_client:
            llm_model = st.selectbox(
                "API Model",
                list(API_LLM_INFO.keys()),
                index=0,
            )
            info = API_LLM_INFO[llm_model]
            st.caption(f"{info['quality']} | {info['speed']} | {info['cost']}")
        else:
            llm_model = None
            st.warning("Set OPENAI_API_KEY environment variable")
    else:
        use_local_llm = True
        local_llm_name = st.selectbox(
            "Local Model",
            list(LOCAL_LLM_INFO.keys()),
            index=0,
        )
        llm_model = local_llm_name
        local_info = LOCAL_LLM_INFO[local_llm_name]
        st.caption(f"{local_info['description']}")
        st.caption(f"Size: {local_info['size_gb']} GB | Languages: {local_info['languages']}")
        arch_label = "Seq2Seq" if local_info.get("arch") == "seq2seq" else "Causal LM"
        st.caption(f"Architecture: {arch_label}")

    # Language preference
    language = st.selectbox(
        "Preferred Language",
        ["English", "Mongolian"],
        index=0,
    )

    # Temperature
    temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.1)
    st.caption("Lower = focused, Higher = creative")

    st.divider()

    # Clear chat button
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_messages = []
        st.rerun()

# Initialize chat history
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Load local LLM if selected
if use_local_llm and llm_model:
    with st.spinner(f"Loading {llm_model}..."):
        local_model, local_tokenizer = load_local_llm(llm_model)
    st.sidebar.success(f"{llm_model} ready!")

# Display chat history
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Type your message..."):
    # Show user message
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if use_local_llm and llm_model:
                response, error = chat_with_local_llm(
                    st.session_state.chat_messages,
                    llm_model,
                    local_model,
                    local_tokenizer,
                    language=language,
                    temperature=temperature,
                )
            elif llm_model:
                response, error = chat_with_openai(
                    st.session_state.chat_messages,
                    model_name=llm_model,
                    language=language,
                    temperature=temperature,
                )
            else:
                response, error = None, "No model selected"

        if error:
            st.error(f"Error: {error}")
        elif response:
            st.markdown(response)
            st.session_state.chat_messages.append({"role": "assistant", "content": response})

            log_usage(st.session_state.get("user_email"), "chat", {
                "model": llm_model,
                "language": language,
                "local": use_local_llm,
                "messages": len(st.session_state.chat_messages),
            })

st.divider()
st.caption("TsuurAI Chat Mode - Powered by Local LLMs & OpenAI")
