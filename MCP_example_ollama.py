import asyncio
import json
from enum import Enum
from typing import TypedDict, Dict, Any, List

import httpx
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama


# =========================
# Константы и категории
# =========================

CATEGORIES: List[str] = [
    "Backend-разработчик",
    "Frontend-разработчик",
    "Fullstack-разработчик",
    "Python-разработчик",
    "Data Scientist",
    "ML-инженер",
    "DevOps-инженер",
    "QA-инженер",
    "Мобильный разработчик",
    "3D-аниматор",
    "2D-аниматор",
    "Дизайнер",
    "Маркетолог",
    "SEO-специалист",
    "Копирайтер",
    "Контент-менеджер",
    "Бизнес-аналитик",
    "Продакт-менеджер",
    "Проект-менеджер"
]


class JobType(Enum):
    PROJECT = "проектная работа"
    PERMANENT = "постоянная работа"


class SearchType(Enum):
    LOOKING_FOR_WORK = "поиск работы"
    LOOKING_FOR_PERFORMER = "поиск исполнителя"


# =========================
# Состояние агента
# =========================

class State(TypedDict):
    description: str
    job_type: str
    category: str
    search_type: str
    confidence_scores: Dict[str, float]
    processed: bool


# =========================
# Агент
# =========================

class VacancyClassificationAgent:
    def __init__(self):
        self.llm = ChatOllama(
    model="llama3.2",
    temperature=0.1,
    base_url="http://localhost:11434")

        self.graph = self._build_graph()

    # ---------- Graph ----------

    def _build_graph(self):
        graph = StateGraph(State)

        graph.add_node("job_type", self._classify_job_type)
        graph.add_node("category", self._classify_category)
        graph.add_node("search_type", self._classify_search_type)
        graph.add_node("confidence", self._calculate_confidence)
        #graph.add_node("hh", self.hh_agent)


        graph.set_entry_point("job_type")
        graph.add_edge("job_type", "category")
        graph.add_edge("category", "search_type")
        graph.add_edge("search_type", "confidence")
        graph.add_edge("confidence", END)

        return graph.compile()

    # ---------- Nodes ----------
    async def hh_agent(self, state: State) -> Dict[str, Any]:
        query = state["category"]

        async with httpx.AsyncClient() as client:
            r = await client.get(
                "https://api.hh.ru/vacancies",
                params={
                    "text": query,
                    "per_page": 20
                },
                headers={"User-Agent": "LangGraph-Agent"}
            )

        data = r.json()
        items = data.get("items", [])

        salaries = [
            v["salary"]["from"]
            for v in items
            if v.get("salary") and v["salary"].get("from")
        ]

        avg_salary = sum(salaries) / len(salaries) if salaries else None

        return {
            "hh_stats": {
                "found": data.get("found", 0),
                "avg_salary": avg_salary,
                "query": query
            }
        }

    async def _classify_job_type(self, state: State) -> Dict[str, Any]:
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
Определи тип работы.

Описание:
{text}

Ответь строго одним вариантом:
- проектная работа
- постоянная работа
"""
        )

        msg = HumanMessage(content=prompt.format(text=state["description"]))
        res = await self.llm.ainvoke([msg])
        answer = res.content.lower()

        if "проект" in answer or "фриланс" in answer or "разов" in answer:
            value = JobType.PROJECT.value
        else:
            value = JobType.PERMANENT.value

        return {"job_type": value}

    async def _classify_category(self, state: State) -> Dict[str, Any]:
        categories = "\n".join(f"- {c}" for c in CATEGORIES)

        prompt = PromptTemplate(
            input_variables=["text", "categories"],
            template="""
Определи категорию профессии.

Описание:
{text}

Список категорий:
{categories}

Выбери ТОЛЬКО одну категорию из списка.
"""
        )

        msg = HumanMessage(
            content=prompt.format(
                text=state["description"],
                categories=categories
            )
        )
        res = await self.llm.ainvoke([msg])
        answer = res.content.strip()

        if answer not in CATEGORIES:
            answer = self._fallback_category(answer)

        return {"category": answer}

    async def _classify_search_type(self, state: State) -> Dict[str, Any]:
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
Определи тип поиска.

Описание:
{text}

Ответь строго:
- поиск работы
- поиск исполнителя
"""
        )

        msg = HumanMessage(content=prompt.format(text=state["description"]))
        res = await self.llm.ainvoke([msg])
        answer = res.content.lower()

        if "ищу" in answer or "резюме" in answer:
            value = SearchType.LOOKING_FOR_WORK.value
        else:
            value = SearchType.LOOKING_FOR_PERFORMER.value

        return {"search_type": value}

    async def _calculate_confidence(self, state: State) -> Dict[str, Any]:
        prompt = PromptTemplate(
            input_variables=["text", "job", "cat", "search"],
            template="""
Оцени уверенность классификации (0.0–1.0).

Описание:
{text}

Тип работы: {job}
Категория: {cat}
Тип поиска: {search}

Ответь ТОЛЬКО JSON:
{{
  "job_type_confidence": 0.0,
  "category_confidence": 0.0,
  "search_type_confidence": 0.0
}}
"""
        )

        msg = HumanMessage(content=prompt.format(
            text=state["description"],
            job=state["job_type"],
            cat=state["category"],
            search=state["search_type"]
        ))

        res = await self.llm.ainvoke([msg])

        try:
            confidence = json.loads(res.content)
        except Exception:
            confidence = {
                "job_type_confidence": 0.7,
                "category_confidence": 0.7,
                "search_type_confidence": 0.7
            }

        return {
            "confidence_scores": confidence,
            "processed": True
        }

    # ---------- Helpers ----------

    def _fallback_category(self, predicted: str) -> str:
        p = predicted.lower()
        for c in CATEGORIES:
            if p in c.lower() or c.lower() in p:
                return c
        return CATEGORIES[0]

    # ---------- Public API ----------

    async def classify(self, text: str) -> Dict[str, Any]:
        state: State = {
            "description": text,
            "job_type": "",
            "category": "",
            "search_type": "",
            "confidence_scores": {},
            "processed": False
        }

        result = await self.graph.ainvoke(state)

        return {
            "job_type": result["job_type"],
            "category": result["category"],
            "search_type": result["search_type"],
            "confidence_scores": result["confidence_scores"],
            "success": result["processed"]
        }


# =========================
# Demo
# =========================

async def main():
    agent = VacancyClassificationAgent()

    examples = [
        "Требуется Python разработчик для постоянной работы в стартапе",
        "Ищу заказы на разработку Telegram-ботов на Python",
        "Нужен 3D-аниматор для разового рекламного проекта",
        "Резюме: ML-инженер, ищу удалённую работу",
    ]

    for text in examples:
        print("📝 Описание:", text)
        result = await agent.classify(text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())

#переписать под поиск автомобилей по параметрам