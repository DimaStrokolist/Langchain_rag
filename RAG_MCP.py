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
        # LLM - используем модель, которую загрузили
        self.llm = ChatOllama(
            model="qwen2.5:32b",  # или "codellama:7b"
            temperature=0.1,
            base_url="http://localhost:11434"  # порт LLM контейнера
        )

        # Embeddings - используем отдельный контейнер
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",  # загруженная в ollama_embeddings
            base_url="http://localhost:11435"  # порт эмбеддинг контейнера!
        )

        # Vector DB (local) - Создаем пустое хранилище или загружаем существующее
        try:
            # Пробуем загрузить существующее хранилище
            self.vectorstore = Chroma(
                persist_directory="./hh_chroma",
                embedding_function=self.embeddings
            )
            print("Загружено существующее векторное хранилище")
        except Exception as e:
            print(f"Создаю новое векторное хранилище: {e}")
            # Создаем пустое хранилище с dummy документом
            self.vectorstore = Chroma.from_documents(
                documents=[Document(page_content="init", metadata={"source": "init"})],
                embedding=self.embeddings,
                persist_directory="./hh_chroma"
            )
            # Удаляем dummy документ
            self.vectorstore.delete(ids=["0"])

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
        except Exception as e:
            print(f"Ошибка парсинга JSON в hr_agent: {e}")
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
            # Пытаемся найти совпадение
            for tech_cat in TECH_CATEGORIES:
                if tech_cat.lower() in cat.lower() or cat.lower() in tech_cat.lower():
                    cat = tech_cat
                    break
            else:
                cat = TECH_CATEGORIES[0]

        print(f"Выбрана категория: {cat}")
        return {"category": cat}

    # --------------------------------------------------
    # HH AGENT (LOAD DATA)
    # --------------------------------------------------

    async def hh_agent(self, state: AgentState) -> Dict[str, Any]:
        query = state["category"]
        print(f"Поиск вакансий для: {query}")

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.get(
                    "https://api.hh.ru/vacancies",
                    params={
                        "text": query,
                        "area": 113,  # Россия
                        "per_page": 10  # Уменьшил для теста
                    },
                    headers={"User-Agent": "HH-RAG-Agent/1.0"}
                )
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            print(f"Ошибка при запросе к HH API: {e}")
            return {
                "hh_stats": {
                    "query": query,
                    "found": 0,
                    "avg_salary": None,
                    "error": str(e)
                }
            }

        items = data.get("items", [])
        print(f"Найдено вакансий: {len(items)}")

        docs = []
        salaries = []

        for idx, v in enumerate(items[:5]):  # Обрабатываем только первые 5 для теста
            try:
                content = f"""
Название: {v.get('name', 'Не указано')}
Компания: {v.get('employer', {}).get('name', 'Не указано')}
Требования: {v.get('snippet', {}).get('requirement', 'Не указано')}
Обязанности: {v.get('snippet', {}).get('responsibility', 'Не указано')}
Зарплата: {v.get('salary', {}).get('from', 'Не указана')} - {v.get('salary', {}).get('to', '')} {v.get('salary', {}).get('currency', '')}
"""

                docs.append(Document(
                    page_content=content,
                    metadata={
                        "id": v.get("id", str(idx)),
                        "name": v.get("name", ""),
                        "source": "hh.ru",
                        "query": query
                    }
                ))

                salary_from = v.get("salary", {}).get("from")
                if salary_from:
                    salaries.append(salary_from)
            except Exception as e:
                print(f"Ошибка обработки вакансии {idx}: {e}")

        # Добавляем документы в векторное хранилище
        if docs:
            try:
                print(f"Добавляю {len(docs)} документов в векторное хранилище...")
                self.vectorstore.add_documents(docs)
                print("Документы успешно добавлены")
            except Exception as e:
                print(f"Ошибка при добавлении документов: {e}")

        avg_salary = sum(salaries) / len(salaries) if salaries else None

        return {
            "hh_stats": {
                "query": query,
                "found": data.get("found", 0),
                "processed": len(docs),
                "avg_salary": avg_salary
            }
        }

    # --------------------------------------------------
    # RAG AGENT (ANALYSIS)
    # --------------------------------------------------

    async def rag_agent(self, state: AgentState) -> Dict[str, Any]:
        query = f"Навыки и требования для {state['category']}"
        print(f"RAG запрос: {query}")

        try:
            # Проверяем количество документов в хранилище
            collection_count = self.vectorstore._collection.count()
            print(f"Документов в хранилище: {collection_count}")

            if collection_count == 0:
                return {"rag_context": "Нет данных для анализа. Векторное хранилище пустое."}

            docs = self.vectorstore.similarity_search(query, k=3)  # Уменьшил k для теста
            print(f"Найдено релевантных документов: {len(docs)}")

            if not docs:
                return {"rag_context": "Не найдено релевантных вакансий для анализа."}

            context = "\n\n---\n\n".join(f"Вакансия {i + 1}:\n{d.page_content}"
                                         for i, d in enumerate(docs))
        except Exception as e:
            print(f"Ошибка в RAG поиске: {e}")
            return {"rag_context": f"Ошибка анализа: {str(e)}"}

        prompt = PromptTemplate(
            input_variables=["context", "category"],
            template="""
Ты карьерный консультант.

На основе анализа вакансий для позиции "{category}":

{context}

Сделай краткий анализ:
1. ТОП-3 самых частых требований
2. Ключевые навыки
3. Что важно указать в резюме

Пиши кратко, по делу, на русском.
"""
        )

        try:
            res = await self.llm.ainvoke([
                HumanMessage(content=prompt.format(
                    context=context,
                    category=state['category']
                ))
            ])
            return {"rag_context": res.content}
        except Exception as e:
            print(f"Ошибка в LLM запросе: {e}")
            return {"rag_context": "Не удалось выполнить анализ из-за ошибки LLM."}

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

        try:
            result = await self.graph.ainvoke(state)
            return result
        except Exception as e:
            print(f"Ошибка выполнения графа: {e}")
            return {
                "error": str(e),
                "processed": False
            }


# ==================================================
# DEMO
# ==================================================

async def main():
    try:
        system = MultiAgentSystem()

        print("\n🧠 MULTI-AGENT RAG SYSTEM FOR HH.RU")
        print("=" * 50)

        # Тестовые запросы
        test_queries = [
            "Ищу работу Python разработчиком, интересует backend",
            "Хочу стать Data Scientist, есть опыт в аналитике",
            "Ищу ML инженера для проекта",
        ]

        for text in test_queries[:1]:  # Тестируем только первый для начала
            print(f"\n📝 Запрос: {text}")
            print("-" * 50)

            result = await system.run(text)

            if "error" in result:
                print(f"❌ Ошибка: {result['error']}")
                continue

            print("\n📊 HH СТАТИСТИКА:")
            if "hh_stats" in result:
                stats = result["hh_stats"]
                print(f"  Запрос: {stats.get('query', 'N/A')}")
                print(f"  Найдено вакансий: {stats.get('found', 0)}")
                print(f"  Обработано: {stats.get('processed', 0)}")
                avg_salary = stats.get('avg_salary')
                if avg_salary:
                    print(f"  Средняя зарплата: {avg_salary:.0f} руб.")
                else:
                    print(f"  Средняя зарплата: Нет данных")

            print("\n🧠 RAG АНАЛИЗ РЫНКА:")
            if "rag_context" in result:
                print(result["rag_context"])

            print("\n" + "=" * 50)
            print("✅ ГОТОВО. Используй этот анализ для резюме и откликов.")

    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())


#решить проблему с нейронками