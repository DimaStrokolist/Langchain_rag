import asyncio
import json
from typing import TypedDict, Dict, Any
import httpx
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama, OllamaEmbeddings

# ==================================================
# TECH CATEGORIES
# ==================================================

TECH_CATEGORIES = [
    "Python-разработчик",
    "ML-инженер",
    "Data Scientist",
    "Backend-разработчик",
    "Frontend-разработчик",
    "DevOps-инженер",
]

# ==================================================
# STATE
# ==================================================

class AgentState(TypedDict):
    description: str
    job_type: str
    search_type: str
    category: str
    hh_stats: Dict[str, Any]
    rag_context: str
    confidence: Dict[str, float]
    processed: bool

# ==================================================
# MULTI AGENT SYSTEM
# ==================================================

class MultiAgentSystem:
    def __init__(self):
        # LLM
        self.llm = ChatOllama(
            model="qwen2.5:32b",
            base_url="http://localhost:11434",
            temperature=0.1
        )

        # Embeddings
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )

        # Vector DB (local)
        self.vectorstore = Chroma(
            collection_name="hh_vacancies",
            embedding_function=self.embeddings,
            persist_directory="./hh_chroma"
        )

        # Graph
        self.graph = self._build_graph()

    # --------------------------------------------------
    # GRAPH
    # --------------------------------------------------

    def _build_graph(self):
        g = StateGraph(AgentState)

        g.add_node("hr_agent", self.hr_agent)
        g.add_node("tech_agent", self.tech_agent)
        g.add_node("hh_agent", self.hh_agent)
        g.add_node("rag_agent", self.rag_agent)
        g.add_node("finalize", self.finalize)

        g.set_entry_point("hr_agent")
        g.add_edge("hr_agent", "tech_agent")
        g.add_edge("tech_agent", "hh_agent")
        g.add_edge("hh_agent", "rag_agent")
        g.add_edge("rag_agent", "finalize")
        g.add_edge("finalize", END)

        return g.compile()

    # --------------------------------------------------
    # HR AGENT
    # --------------------------------------------------

    async def hr_agent(self, state: AgentState) -> Dict[str, Any]:
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
Ты HR-эксперт.

Описание:
{text}

Ответь JSON:
{{
  "job_type": "проектная работа | постоянная работа",
  "search_type": "поиск работы | поиск исполнителя"
}}
"""
        )

        res = await self.llm.ainvoke([
            HumanMessage(content=prompt.format(text=state["description"]))
        ])

        try:
            return json.loads(res.content)
        except Exception:
            return {
                "job_type": "постоянная работа",
                "search_type": "поиск работы"
            }

    # --------------------------------------------------
    # TECH AGENT
    # --------------------------------------------------

    async def tech_agent(self, state: AgentState) -> Dict[str, Any]:
        cats = "\n".join(f"- {c}" for c in TECH_CATEGORIES)

        prompt = PromptTemplate(
            input_variables=["text", "cats"],
            template="""
Ты технический эксперт.

Описание:
{text}

Категории:
{cats}

Выбери ОДНУ категорию и напиши её точно.
"""
        )

        res = await self.llm.ainvoke([
            HumanMessage(content=prompt.format(
                text=state["description"],
                cats=cats
            ))
        ])

        cat = res.content.strip()
        if cat not in TECH_CATEGORIES:
            cat = TECH_CATEGORIES[0]

        return {"category": cat}

    # --------------------------------------------------
    # HH AGENT (LOAD DATA + RAG)
    # --------------------------------------------------

    async def hh_agent(self, state: AgentState) -> Dict[str, Any]:
        query = state["category"]

        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(
                "https://api.hh.ru/vacancies",
                params={
                    "text": query,
                    "area": 113,  # Россия
                    "per_page": 30
                },
                headers={"User-Agent": "HH-RAG-Agent"}
            )

        data = r.json()
        items = data.get("items", [])

        docs = []
        salaries = []

        for v in items:
            content = f"""
Название вакансии: {v.get('name')}
Компания: {v.get('employer', {}).get('name')}
Требования: {v.get('snippet', {}).get('requirement')}
Обязанности: {v.get('snippet', {}).get('responsibility')}
Ссылка: {v.get('alternate_url')}
"""
            docs.append(Document(page_content=content))

            if v.get("salary") and v["salary"].get("from"):
                salaries.append(v["salary"]["from"])

        if docs:
            self.vectorstore.add_documents(docs)
            self.vectorstore.persist()

        avg_salary = sum(salaries) / len(salaries) if salaries else None

        return {
            "hh_stats": {
                "query": query,
                "found": data.get("found", 0),
                "avg_salary": avg_salary
            }
        }

    # --------------------------------------------------
    # RAG AGENT (ANALYSIS)
    # --------------------------------------------------

    async def rag_agent(self, state: AgentState) -> Dict[str, Any]:
        query = f"Навыки и требования для {state['category']}"
        docs = self.vectorstore.similarity_search(query, k=5)
        context = "\n\n".join(d.page_content for d in docs)

        prompt = PromptTemplate(
            input_variables=["context"],
            template="""
Ты карьерный консультант.

На основе вакансий ниже:
{context}

Сделай анализ:
1. ТОП-5 навыков
2. Типичные требования
3. Что обязательно указать в резюме
4. Как выделиться среди кандидатов

Пиши кратко и по делу.
"""
        )

        res = await self.llm.ainvoke([
            HumanMessage(content=prompt.format(context=context))
        ])

        return {"rag_context": res.content}

    # --------------------------------------------------
    # FINAL
    # --------------------------------------------------

    async def finalize(self, state: AgentState) -> Dict[str, Any]:
        return {
            "confidence": {
                "hr": 0.9,
                "tech": 0.85,
                "market": 0.9
            },
            "processed": True
        }

    # --------------------------------------------------
    # API
    # --------------------------------------------------

    async def run(self, text: str) -> Dict[str, Any]:
        state: AgentState = {
            "description": text,
            "job_type": "",
            "search_type": "",
            "category": "",
            "hh_stats": {},
            "rag_context": "",
            "confidence": {},
            "processed": False
        }
        return await self.graph.ainvoke(state)

# ==================================================
# DEMO
# ==================================================

async def main():
    system = MultiAgentSystem()

    print("\n🧠 GRAPH:")
    print(system.graph.get_graph().draw_mermaid())

    text = "Ищу работу Python разработчиком, интересует backend"
    result = await system.run(text)

    print("\n📊 HH СТАТИСТИКА:")
    print(json.dumps(result["hh_stats"], ensure_ascii=False, indent=2))

    print("\n🧠 RAG АНАЛИЗ РЫНКА:")
    print(result["rag_context"])

    print("\n✅ ГОТОВО. Используй этот анализ для резюме и откликов.")

if __name__ == "__main__":
    asyncio.run(main())

#в отдельном файле подключиться к chroma создать и вставить документ, изменить, удалить и вывести список по статье https://docs.trychroma.com/docs/overview/getting-started