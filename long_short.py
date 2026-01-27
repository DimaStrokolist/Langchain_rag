from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, after_model
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from typing import Any


class CustomAgentState(AgentState):
    user_id: str
    preferences: dict
# =====================
# 1. Инструмент (необязательно, но пусть будет)
# =====================

@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]

    if len(messages) <= 3:
        return None  # No changes needed

    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }
@after_model
def delete_old_messages2(state: AgentState, runtime: Runtime) -> dict | None:
    """Remove old messages to keep conversation manageable."""
    messages = state["messages"]
    if len(messages) > 2:
        # remove the earliest two messages
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:0]]}
    return None

def delete_messages(state):
    messages = state["messages"]
    if len(messages) > 2:
        # remove the earliest two messages
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}


def delete_messages1(state : AgentState, runtime: Runtime) -> dict | None:
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}

@tool
def echo(text: str) -> str:
    """Просто возвращает текст"""
    return text


# =====================
# 2. Модель Ollama
# =====================

model = ChatOllama(
    model="llama3.2",
    temperature=0.3,
    base_url="http://localhost:11434")



# =====================
# 3. Создание агента с памятью
# =====================

agent = create_agent(
    model=model,
    tools=[echo],
    middleware=[delete_old_messages2],
    state_schema= CustomAgentState,
    system_prompt="Ты дружелюбный ассистент. Запоминай, что говорит пользователь.",
    checkpointer=InMemorySaver(),  # ⭐ ВАЖНО
)


# =====================
# 4. Один thread (один диалог)
# =====================

config: RunnableConfig = {"configurable": {"thread_id": "1"}}


# =====================
# 5. Диалог
# =====================

agent.invoke({
    "messages": [{"role": "user", "content": "Привет! Меня зовут Алекс"}],
    "user_id": "user_123",
    "preferences": {"theme": "dark"}},
    config,
)

agent.invoke(
    {"messages": [{"role": "user", "content": "Запомни моё имя"}]},
    config,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Как меня зовут?"}]},
    config,
)

print(result["messages"][-1].content)
print(len(result["messages"]))
for i in result["messages"]:
    print(i.content)
# статья https://docs.langchain.com/oss/python/langchain/short-term-memory#trim-messages
# написать код с кратковременной памятью (записать, достать)