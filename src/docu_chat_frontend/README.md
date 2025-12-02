# DocuChat Frontend

Streamlit-based web interface for the DocuChat RAG (Retrieval-Augmented Generation) application.

## Features

- 📄 **Document Upload**: Support for PDF and DOCX files
- 💬 **Interactive Chat**: Ask questions about uploaded documents
- 🔄 **Session Management**: Maintains conversation context
- ⚡ **Real-time Streaming**: Typing effect for responses
- 🎨 **Clean UI**: Simple and intuitive Streamlit interface

## Architecture

```
docu_chat_frontend/
├── main.py              # Streamlit application entry point
├── api_client.py        # Backend API communication
├── ui_components.py     # UI components and utilities
└── __init__.py
```

### Module Overview

**`main.py`** - Main Streamlit application
- Page configuration
- File upload interface
- Chat flow orchestration

**`api_client.py`** - Backend integration
- `chat()` - Send messages to RAG backend
- `upload_file()` - Upload documents to S3 storage

**`ui_components.py`** - UI utilities
- Session state management
- File handling
- Chat history display
- Response streaming with typing effect

## Prerequisites

- Python 3.10+
- Running DocuChat backend (port 8000)
- Backend services configured:
  - AWS S3 for document storage
  - MongoDB for conversation history
  - Perplexity AI for LLM responses

## Installation

```bash
# Navigate to project root
cd ai-applications

# Install dependencies (if not already done)
poetry install

# Ensure backend is running
poetry run uvicorn src.docu_chat.main:app --reload
```

## Usage

### Start the Frontend

```bash
# From project root
poetry run streamlit run src/docu_chat_frontend/main.py
```

The application will open at `http://localhost:8501`

### Using the Application

1. **Upload Document**
   - Click "Browse files" or drag & drop
   - Supported formats: PDF, DOCX
   - File is uploaded to backend S3 storage

2. **Ask Questions**
   - Type your question in the chat input
   - Press Enter to submit
   - AI will analyze the document and respond

3. **Continue Conversation**
   - Session maintains context automatically
   - Previous Q&A are visible in chat history
   - Upload new document to start fresh session

## Configuration

Backend URL is configured in `api_client.py`:

```python
BACKEND_URL = "http://localhost:8000"
```

Change this if your backend runs on different host/port.

## Features in Detail

### Session Management
- Automatically creates unique session IDs
- Maintains conversation history per document
- Stored in MongoDB via backend

### File Handling
- Local temp storage: `temp/` directory
- Automatic directory creation
- Files uploaded to S3 via backend API

### Response Streaming
- Simulated typing effect (0.05s per word)
- Visual cursor indicator (▌)
- Smooth user experience

## Troubleshooting

### Backend Connection Error
```
Error from backend: 500 - Internal Server Error
```
**Solution**: Ensure backend is running on port 8000

### File Upload Failed
```
Error uploading file: 500
```
**Solution**: Check backend S3 credentials in `.env`

### Import Errors
```
ModuleNotFoundError: No module named 'streamlit'
```
**Solution**: Install dependencies with `poetry install`

## Development

### Adding New Features

1. **New API endpoints** → Edit `api_client.py`
2. **UI components** → Edit `ui_components.py`
3. **Main flow** → Edit `main.py`

### Code Structure

Follow separation of concerns:
- **API layer**: `api_client.py` (no UI code)
- **UI layer**: `ui_components.py` (no direct API calls)
- **Application**: `main.py` (orchestration only)

## Related Documentation

- [Backend Setup](../docu_chat/README.md)
- [AWS Configuration](../../docs/AWS_SETUP.md)
- [MongoDB Configuration](../../docs/MONGODB_SETUP.md)

## Tech Stack

- **Streamlit** - Web framework
- **Requests** - HTTP client
- **Python 3.10+** - Runtime

## License

Private project
