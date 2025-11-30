from pydantic import BaseModel
import pymongo

import traceback
import os, sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, UploadFile, status, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.callbacks import get_openai_callback
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

import gc
import awswrangler as wr
import boto3

# Configuration from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# API Configuration (supports OpenAI, Perplexity, etc.)
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")  # e.g., https://api.perplexity.ai
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo-16k")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

S3_KEY = os.getenv("AWS_ACCESS_KEY_ID")
S3_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET = os.getenv("AWS_S3_BUCKET", "docchat")
S3_REGION = os.getenv("AWS_S3_REGION", "us-east-1")
S3_PATH = os.getenv("AWS_S3_PATH", "documents/")

try:
    MONGO_URL = os.getenv("MONGODB_URL", "mongodb+srv://admin:admin@cluster0.jyupp.mongodb.net/?retryWrites=true&w=majority&ssl=true")
    MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "chat_with_doc")
    MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "chat_history")

    client = pymongo.MongoClient(MONGO_URL, uuidRepresentation="standard")
    db = client[MONGODB_DATABASE]
    conversation_col = db[MONGODB_COLLECTION]

    conversation_col.create_index([("session_id")], unique=True)
except:
    print(traceback.format_exc())
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, exc_tb.tb_lineno)


class ChatMessageSent(BaseModel):
    session_id: str = None
    user_input: str
    data_source: str

def get_response(
    file_name: str,
    session_id: str,
    query: str,
    model: str = None,
    temperature: float = 0.0,
):
    print(f"file name is {file_name}")
    file_name = file_name.split("/")[-1]

    # Use configured model or default
    if model is None:
        model = LLM_MODEL

    # Use local HuggingFace embeddings (free, no API needed)
    # Perplexity API does NOT support embeddings!
    print("Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    wr.s3.download(
        path=f"s3://{S3_BUCKET}/{S3_PATH}{file_name}",
        local_file=file_name,
        boto3_session=aws_s3
    )

    if file_name.lower().endswith(".docx"):
        loader = Docx2txtLoader(file_path=file_name)
    else:
        loader = PyPDFLoader(file_path=file_name)

    data = loader.load()
    print("splitting...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        separators=["\n\n", "\n", " ", ""],
    )

    all_splits = text_splitter.split_documents(data)

    vectorstore = FAISS.from_documents(all_splits, embeddings)

    # Configure LLM (supports custom API base)
    llm_kwargs = {"model_name": model, "temperature": temperature}
    if OPENAI_API_BASE:
        llm_kwargs["openai_api_base"] = OPENAI_API_BASE
    llm = ChatOpenAI(**llm_kwargs)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectorstore.as_retriever()
    )

    with get_openai_callback() as cb:
        answer = qa_chain(
            {
                "question": query,
                "chat_history": load_memory_to_pass(session_id=session_id),
            }
        )

        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")
        answer["total_tokens_used"] = cb.total_tokens
    gc.collect()
    return answer

def load_memory_to_pass(session_id: str):
    """
    Load conversation history for a given session ID

    Args:
        session_id (str): Unique session identifier

    Returns:
        list: List of conversational history as a list of tuples (user_message, bot_response)
    """
    data = conversation_col.find_one({"session_id": session_id})
    history = []
    if data:
        data = data.get("conversation", [])
        for x in range(0, len(data), 2):
            history.extend([(data[x], data[x + 1])])

    print(f"Loaded history: {history}")
    return history

import uuid
def get_session() -> str:
    return str(uuid.uuid4())

def add_session_history(session_id: str, new_values: list):
    """
    Add new conversation entries to the session history

    Args:
        session_id (str): Unique session identifier
        new_values (list): List of new conversation entries to add
    """
    document = conversation_col.find_one({"session_id": session_id})
    if document:
        conversation = document.get("conversation", [])
        conversation.extend(new_values)
        conversation_col.update_one(
            {"session_id": session_id}, {"$set": {"conversation": conversation}},
        )
    else:
        conversation_col.insert_one(
            {"session_id": session_id, "conversation": new_values}
        )


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

aws_s3 = boto3.Session(
    aws_access_key_id=S3_KEY,
    aws_secret_access_key=S3_SECRET,
    region_name=S3_REGION,
)


@app.post("/chat")
async def create_chat_message(chats: ChatMessageSent):
    try:
        if chats.session_id is None:
            chats.session_id = get_session()

            payload = ChatMessageSent(
                session_id=chats.session_id,
                user_input=chats.user_input,
                data_source=chats.data_source,
            )
            payload = payload.model_dump()

            response = get_response(
                file_name=payload.get("data_source"),
                session_id=payload.get("session_id"),
                query=payload.get("user_input"),
            )

            add_session_history(
                session_id=chats.session_id,
                new_values=[payload.get("user_input"), response.get("answer")],
            )

            return JSONResponse(
                content={
                    "session_id": str(chats.session_id),
                    "response": response,
                }
            )

        else:
            payload = ChatMessageSent(
                session_id=str(chats.session_id),
                user_input=chats.user_input,
                data_source=chats.data_source,
            )
            payload = payload.dict()

            response = get_response(
                file_name=payload.get("data_source"),
                session_id=payload.get("session_id"),
                query=payload.get("user_input"),
            )

            add_session_history(
                session_id=str(chats.session_id),
                new_values=[payload.get("user_input"), response.get("answer")],
            )

            return JSONResponse(
                content={
                    "session_id": str(chats.session_id),
                    "response": response,
                }
            )
    except Exception:
        print(traceback.format_exc())
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        raise HTTPException(
            status_code=status.HTTP_204_NO_CONTENT,
            detail="error",
        )

@app.post("/uploadFile")
async def upload_to_s3(data_file: UploadFile):
    file_name = data_file.filename.split("/")[-1]
    print(file_name)
    try:
        with open(file_name, "wb") as out_file:
            content = await data_file.read()
            out_file.write(content)
        s3_file_path = f"s3://{S3_BUCKET}/{S3_PATH}{file_name}"
        wr.s3.upload(
            local_file=file_name,
            path=s3_file_path,
            boto3_session=aws_s3,
        )
        os.remove(file_name)
        response ={
            "filename": file_name,
            "file_path": s3_file_path
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")

    return JSONResponse(content=response)

import uvicorn
if __name__ == "__main__":
    uvicorn.run(app)