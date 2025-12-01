# AI Applications

Collection of AI-powered applications including chatbots, document Q&A, and business predictions.

## Projects

### 1. Document Chat (docu_chat)
RAG-based document Q&A system using FastAPI, Perplexity AI, and vector search.

**Documentation:** [src/docu_chat/README.md](src/docu_chat/README.md)

**Features:**
- Chat with PDF/DOCX documents
- Perplexity AI integration for responses
- Local HuggingFace embeddings (free)
- MongoDB conversation history
- S3 document storage

**Quick Start:**
```bash
# Setup (see detailed docs)
cp .env.example .env
# Fill in .env with your credentials

# Install dependencies
poetry install

# Run
poetry run uvicorn src.docu_chat.main:app --reload
```

### 2. Simple Chatbot (chatbot)
Intent-based chatbot with sentiment analysis using TextBlob.

**Features:**
- Keyword-based intent matching
- Sentiment analysis
- Simple CLI interface

**Run:**
```bash
poetry run python src/chatbot/main.py
```

### 3. Business Prediction (business_prediction)
Coffee shop location analysis and revenue prediction using machine learning.

**Features:**
- Population data analysis
- Location scoring
- Revenue prediction models
- Data visualization

**Run:**
```bash
# Method 1: As module
poetry run python -m src.business_prediction.main

# Method 2: Direct execution
poetry run python src/business_prediction/main.py
```

## Project Structure

```
ai-applications/
├── src/
│   ├── chatbot/                   # Simple intent-based chatbot
│   │   ├── main.py               # Chatbot implementation
│   │   └── __init__.py
│   ├── business_prediction/       # Business analytics & ML predictions
│   │   ├── main.py               # Main prediction pipeline
│   │   ├── data_loader.py        # Data loading utilities
│   │   ├── model_training.py     # ML model training
│   │   ├── predictions.py        # Prediction generation
│   │   ├── preprocessing.py      # Data preprocessing
│   │   └── __init__.py
│   └── docu_chat/                 # Document chat application
│       ├── main.py               # Main FastAPI app
│       └── __init__.py
├── docs/                          # Documentation
│   ├── SETUP_DOCU_CHAT.md        # Document chat setup guide
│   ├── AWS_SETUP.md              # AWS S3 configuration
│   └── MONGODB_SETUP.md          # MongoDB Atlas setup
├── data/                          # Data files
├── .env.example                   # Environment template
└── pyproject.toml                # Poetry dependencies
```

## Requirements

- Python 3.10+
- Poetry (package manager)

## Installation

```bash
# Clone repository
git clone <your-repo-url>
cd ai-applications

# Install dependencies
poetry install

# Configure environment
cp .env.example .env
# Edit .env with your credentials
```

## Environment Variables

See `.env.example` for required configuration. Each application may need different services:

- **docu_chat**: Perplexity API, AWS S3, MongoDB Atlas
- **chatbot**: No external services needed
- **business_prediction**: No external services needed

## Documentation

- [Document Chat Setup](src/docu_chat/README.md)
- [AWS Configuration](docs/AWS_SETUP.md)
- [MongoDB Configuration](docs/MONGODB_SETUP.md)

## License

Private project
