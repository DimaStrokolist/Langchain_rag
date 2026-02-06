import asyncio
import json
from typing import TypedDict, Dict, Any, List
from langchain_chroma import Chroma
import httpx
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

from langgraph.graph import StateGraph, END

from langchain_ollama import ChatOllama, OllamaEmbeddings
from sqlalchemy.testing.suite.test_reflection import metadata

llm = ChatOllama(
            model="qwen2.5:32b",  # или "codellama:7b"
            temperature=0.1,
            base_url="http://localhost:11434"  # порт LLM контейнера
        )

embeddings = OllamaEmbeddings(
            model="nomic-embed-text",  # загруженная в ollama_embeddings
            base_url="http://localhost:11435"  # порт эмбеддинг контейнера!
        )

vectorstore = Chroma(
                persist_directory="./hh_chroma",
                embedding_function=embeddings
            )
docs = []
ids = []

docs.append(Document(page_content="hello world",
                     metadata={"source":"123"}))
ids.append("#123")


vectorstore.add_documents(docs,ids = ids)

print(vectorstore.similarity_search("456")[0].page_content)

print(vectorstore.get(where={"source":"123"}))

# vectorstore.delete(ids=["#123"])

print(vectorstore.get(where={"source":"123"}))

def get_docs(query: str, k: int =5) -> List[Document]:
    return vectorstore.similarity_search(query=query, k=k)

def build_context(docs: List[Document]) -> str:
    return "/n/n".join(f"{doc.page_content}" for doc in docs)

print(get_docs(""))

d = get_docs("")

s = build_context(d)

print(s)