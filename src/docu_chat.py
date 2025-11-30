from pydantic import BaseModel
import pymongo

import traceback
import os, sys

from fastapi import FastAPI, UploadFile, status, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import S3FileLoader, PyPDFLoader, Docx2txtLoader
from langchain_community.callbacks import get_openai_callback
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

import gc
import urllib.parse
import awswrangler as wr
import boto3

