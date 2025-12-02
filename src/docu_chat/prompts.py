"""
Prompt templates for RAG conversation chain

These templates control how the LLM processes questions and generates answers.
"""
from langchain_core.prompts import PromptTemplate


# Template for condensing follow-up questions into standalone questions
CONDENSE_QUESTION_TEMPLATE = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(CONDENSE_QUESTION_TEMPLATE)


# Template for answering questions based on document context
QA_TEMPLATE = """You are an AI assistant that ONLY answers questions based on the provided document context.
DO NOT use any external knowledge or internet sources.
If the answer is not in the document, say "I cannot find this information in the provided document."

IMPORTANT: Always answer in the SAME LANGUAGE as the question. If the question is in Polish, answer in Polish. If in English, answer in English.

Context from document:
{context}

Question: {question}

Answer based ONLY on the document above, in the SAME LANGUAGE as the question:"""

QA_PROMPT = PromptTemplate.from_template(QA_TEMPLATE)
