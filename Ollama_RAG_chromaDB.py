import asyncio
import json
from typing import TypedDict, Dict, Any, List
from langchain_chroma import Chroma
import httpx
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

from langgraph.graph import StateGraph, END

llm = ChatOllama(
            model="tinyllama:1.1b",  # или "codellama:7b"
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

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert assistant.
Answer the question ONLY using the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
)

def get_docs(query: str, k: int =5) -> List[Document]:
    return vectorstore.similarity_search(query=query, k=k)

def build_context(docs: List[Document]) -> str:
    return "/n/n".join(f"{doc.page_content}" for doc in docs)

def rag_answer(question):
    docs = get_docs(question)
    context = build_context(docs)
    prompt = RAG_PROMPT.format(context=context, question=question)
    responce = llm.invoke(prompt)
    return responce.content

print(rag_answer("курс доллара"))


def search_doctors(query: str, k: int = 3):
    """Поиск врачей по запросу"""

    results = vectorstore.similarity_search(
        query=query,
        k=k
    )

    print(f"\nРезультаты поиска для запроса: '{query}'")
    print("=" * 50)

    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc.metadata['doctor_name']}")
        print(f"   Специализация: {doc.metadata['specialization']}")
        print(f"   Кабинет: {doc.metadata['cabinet_number']} (этаж {doc.metadata['floor']})")
        print(f"   Описание: {doc.metadata['description'][:100]}...")

    return results
