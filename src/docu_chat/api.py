"""
FastAPI application endpoints

Uses pluggable providers via Factory pattern:
- Switch storage: S3 / GCP / Azure
- Switch database: MongoDB / PostgreSQL / Redis
- Switch embeddings: HuggingFace / OpenAI
- Switch LLM: Perplexity / OpenAI / Anthropic

Configure in .env with:
    STORAGE_PROVIDER=s3
    DATABASE_PROVIDER=mongodb
    EMBEDDINGS_PROVIDER=huggingface
    LLM_PROVIDER=perplexity
"""
import os
import uuid
import traceback
import sys
from fastapi import FastAPI, UploadFile, status, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.callbacks import get_openai_callback
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from .factory import ProviderFactory
from .config import LLM_MODEL, LLM_TEMPERATURE, OPENAI_API_BASE


# Initialize providers from factory
storage_provider = ProviderFactory.create_storage()
database_provider = ProviderFactory.create_database()
embeddings_provider = ProviderFactory.create_embeddings()


class ChatMessageSent(BaseModel):
    session_id: str = None
    user_input: str
    data_source: str


app = FastAPI(
    title="Document Chat API",
    description="RAG-based document Q&A system with pluggable providers",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_session() -> str:
    """Generate new session ID"""
    return str(uuid.uuid4())


@app.post("/chat", summary="Chat with document")
async def create_chat_message(chats: ChatMessageSent):
    """
    Chat with uploaded document using RAG

    - **session_id**: Session ID (null for new session)
    - **user_input**: User question
    - **data_source**: Document filename in storage
    """
    try:
        # Generate session ID if not provided
        if chats.session_id is None:
            chats.session_id = get_session()

        # Extract filename from path
        file_name = chats.data_source.split("/")[-1]

        # Download document from storage
        local_file = storage_provider.download_document(file_name)

        # Load document
        if local_file.lower().endswith(".docx"):
            loader = Docx2txtLoader(file_path=local_file)
        elif local_file.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path=local_file)
        else:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Unsupported file format. Only PDF and DOCX are supported."}
            )

        data = loader.load()

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0,
            separators=["\n\n", "\n", " ", ""],
        )
        documents = text_splitter.split_documents(data)

        # Create vector store with embeddings provider
        # Extract the actual LangChain embeddings object from our wrapper
        if hasattr(embeddings_provider, 'embeddings'):
            # Our custom wrapper - extract the LangChain object
            langchain_embeddings = embeddings_provider.embeddings
        else:
            # Direct LangChain embeddings object
            langchain_embeddings = embeddings_provider

        vectorstore = FAISS.from_documents(documents, langchain_embeddings)

        # Create LLM
        llm = ChatOpenAI(
            model_name=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            openai_api_base=OPENAI_API_BASE
        )

        # Create QA chain with custom prompt to force document-only responses
        condense_question_prompt = PromptTemplate.from_template(
            """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
        )

        qa_prompt = PromptTemplate.from_template(
            """You are an AI assistant that ONLY answers questions based on the provided document context.
DO NOT use any external knowledge or internet sources.
If the answer is not in the document, say "I cannot find this information in the provided document."

IMPORTANT: Always answer in the SAME LANGUAGE as the question. If the question is in Polish, answer in Polish. If in English, answer in English.

Context from document:
{context}

Question: {question}

Answer based ONLY on the document above, in the SAME LANGUAGE as the question:"""
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=vectorstore.as_retriever(),
            condense_question_prompt=condense_question_prompt,
            combine_docs_chain_kwargs={"prompt": qa_prompt}
        )

        # Load conversation history from database (MongoDB/Postgres/Redis)
        chat_history = database_provider.load_history(chats.session_id)

        # Generate response
        with get_openai_callback() as cb:
            answer = qa_chain.invoke({
                "question": chats.user_input,
                "chat_history": chat_history,
            })

            # Remove <think> tags from Perplexity reasoning output
            import re
            if "answer" in answer and answer["answer"]:
                answer["answer"] = re.sub(r'<think>.*?</think>\s*', '', answer["answer"], flags=re.DOTALL)

            # Add token usage information
            answer["total_tokens_used"] = cb.total_tokens
            answer["prompt_tokens"] = cb.prompt_tokens
            answer["completion_tokens"] = cb.completion_tokens
            answer["total_cost"] = cb.total_cost

            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost (USD): ${cb.total_cost}")

        # Save conversation history to database
        database_provider.save_history(
            chats.session_id,
            [chats.user_input, answer.get("answer")]
        )

        # Cleanup local file
        if os.path.exists(local_file):
            os.remove(local_file)

        return JSONResponse(
            content={
                "session_id": str(chats.session_id),
                "response": answer,
            }
        )

    except Exception as e:
        print(traceback.format_exc())
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}",
        )


@app.post("/uploadFile", summary="Upload document to storage")
async def upload_file(data_file: UploadFile):
    """
    Upload document (PDF/DOCX) to storage (S3/GCP/Azure)

    Returns filename and storage path
    """
    file_name = data_file.filename.split("/")[-1]
    print(f"Uploading file: {file_name}")

    try:
        # Save file temporarily
        with open(file_name, "wb") as out_file:
            content = await data_file.read()
            out_file.write(content)

        # Upload to storage (S3/GCP/Azure)
        storage_path = storage_provider.upload_document(file_name)

        # Cleanup local file
        os.remove(file_name)

        return JSONResponse(content={
            "filename": file_name,
            "file_path": storage_path
        })

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading file: {str(e)}"
        )


@app.get("/", summary="Health check")
async def root():
    """API health check endpoint"""
    return {
        "status": "ok",
        "service": "Document Chat API",
        "providers": {
            "storage": storage_provider.__class__.__name__,
            "database": database_provider.__class__.__name__,
            "embeddings": embeddings_provider.__class__.__name__,
            "llm": LLM_MODEL
        }
    }
