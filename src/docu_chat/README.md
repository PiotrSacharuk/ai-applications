# AI Applications - docu_chat.py

## Description
FastAPI application for chatting with documents (PDF/DOCX) using RAG (Retrieval-Augmented Generation).

## What was fixed automatically:
- Syntax error: `file_name.tolower()` → `file_name.lower()`
- Variable error: `session_id` → `chats.session_id`
- Created `.env.example` with configuration templates
- Updated `.gitignore`
- Changed hardcoded credentials to environment variables
- Added `python-dotenv` to `pyproject.toml`
- Switched to HuggingFace embeddings (free, local)
- Configured Perplexity API support

## Manual steps required:

### 1. Copy configuration file
```bash
cp .env.example .env
```

### 2. Fill in `.env` with your data:
- **OPENAI_API_KEY** - Perplexity API key from https://www.perplexity.ai/settings/api
- **OPENAI_API_BASE** - https://api.perplexity.ai
- **LLM_MODEL** - Model name (e.g., `sonar`, `sonar-pro`, `sonar-reasoning`)
- **AWS_ACCESS_KEY_ID** - AWS access key
- **AWS_SECRET_ACCESS_KEY** - AWS secret key
- **AWS_S3_BUCKET** - S3 bucket name (default: docchat)
- **MONGODB_URL** - MongoDB Atlas connection string

### 3. Install dependencies
```bash
poetry install
```

### 4. Create S3 bucket (if it doesn't exist)
- Log in to AWS Console
- Create bucket with the name specified in `.env` (e.g., `docchat`)
- Create `documents/` folder in bucket

### 5. Configure MongoDB
- Create free cluster at https://www.mongodb.com/cloud/atlas
- Get connection string
- Paste into `.env` as `MONGODB_URL`

### 6. Run the application
```bash
poetry run uvicorn src.docu_chat.main:app --reload
```

## API Endpoints:

**POST /uploadFile**
- Upload document (PDF/DOCX) to S3
- Body: `multipart/form-data` with file

**POST /chat**
- Chat with document
- Body JSON:
```json
{
  "session_id": "uuid-or-null",
  "user_input": "question",
  "data_source": "filename.pdf"
}
```

## Testing:
```bash
curl http://localhost:8000/uploadFile -F "data_file=@document.pdf"
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"session_id": null, "user_input": "What is in this document?", "data_source": "document.pdf"}'
```

## Notes:
- Application uses Perplexity `sonar` model (configurable in `.env`)
- Uses HuggingFace embeddings (free, local - sentence-transformers/all-MiniLM-L6-v2)
- Chat history is saved in MongoDB
- Documents are stored in S3
