"""
UI components and utilities for Streamlit interface
"""
import streamlit as st
import time
import os
from api_client import chat


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "sessionid" not in st.session_state:
        st.session_state.sessionid = None


def save_uploaded_file(uploaded_file) -> str:
    """
    Save uploaded file to temp directory

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        Local file path where file was saved
    """
    temp_dir = os.path.join(os.getcwd(), "temp")
    os.makedirs(temp_dir, exist_ok=True)

    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def display_chat_history():
    """Display all previous chat messages"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def stream_response(response_text: str, placeholder) -> str:
    """
    Display response with typing effect

    Args:
        response_text: Full response text to display
        placeholder: Streamlit empty placeholder for updates

    Returns:
        Full response text
    """
    full_response = ""

    for chunk in response_text.split():
        full_response += chunk + " "
        time.sleep(0.05)
        placeholder.markdown(full_response + "▌")

    placeholder.markdown(full_response)
    return full_response


def handle_chat_input(prompt: str, document_path: str):
    """
    Process user chat input and generate response

    Args:
        prompt: User's question
        document_path: Path to document in storage
    """
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get and display assistant response
    with st.chat_message("assistant"):
        if st.session_state.sessionid is None:
            assistant_response, session_id = chat(
                prompt,
                data=document_path,
                session_id=None
            )
            st.session_state.sessionid = session_id
        else:
            assistant_response, session_id = chat(
                prompt,
                data=document_path,
                session_id=st.session_state.sessionid
            )

        message_placeholder = st.empty()
        full_response = stream_response(assistant_response, message_placeholder)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
