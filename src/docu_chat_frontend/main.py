"""
DocuChat Frontend - Streamlit Application

Simple chat interface for document Q&A using RAG backend.
Upload PDF/DOCX and ask questions about the content.
"""
import streamlit as st
from api_client import upload_file
from ui_components import (
    initialize_session_state,
    save_uploaded_file,
    display_chat_history,
    handle_chat_input
)


# Configure Streamlit page
st.set_page_config(
    page_title="DocuChat",
    page_icon=":books:",
    layout="wide"
)

# Initialize session state
initialize_session_state()

# File upload section
data_file = st.file_uploader(
    label="Upload Document (PDF/DOCX)",
    accept_multiple_files=False,
    type=["pdf", "docx"]
)

st.divider()

# Main chat interface
if data_file is not None:
    # Save uploaded file locally
    file_path = save_uploaded_file(data_file)

    # Upload to backend storage (S3)
    s3_upload_url = upload_file(file_path)
    document_name = s3_upload_url.split("/")[-1]

    # Display chat history
    display_chat_history()

    # Handle new user input
    if prompt := st.chat_input("Ask a question about your document"):
        handle_chat_input(prompt, document_name)
