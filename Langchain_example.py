from langchain.agents import create_agent
from langchain_community.llms.ollama import Ollama
from langchain.tools import tool
from langchain_ollama import ChatOllama
import json

llm = ChatOllama(
    model="llama3.2",
    temperature=0.3,
    base_url="http://localhost:11434")

@tool(
    "calculator",
    description="Performs arithmetic calculations. Use this for any math problems."
)
def calc(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression)+1)

agent = create_agent(llm,tools=[calc])
# result = agent.invoke(
#     {"messages": [{"role": "user", "content": "2+2"}]})
# messages = result["messages"]
# for msg in messages:
#     if msg.__class__.__name__ == "ToolMessage":
#         print(f"ToolMessage: {msg.content}")
#     elif msg.__class__.__name__ == "AIMessage":
#         print(f"AIMessage: {msg.content}")
#
# print("\nФинальный результат агента:")
#
# last_tool = next(
#     msg for msg in reversed(messages)
#     if msg.__class__.__name__ == "ToolMessage"
# )
#
# print(last_tool.content)

def agent_invoke(m):
    result = agent.invoke(
        {"messages": [{"role": "user", "content": m}]})
    return result["messages"]
print(agent_invoke("10+1"))

#все операции со словарями сделать https://metanit.com/python/tutorial/3.3.php

from typing import Any


def run_agent_and_get_final(user_input: str) -> str:
    """
    Запускает агента, печатает ход рассуждений и
    возвращает финальный результат (ToolMessage или AIMessage)
    """

    # 1. Вызов агента
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]}
    )

    messages = result.get("messages", [])

    print("=== Ход выполнения агента ===")

    # 2. Лог всех сообщений
    for msg in messages:
        cls = msg.__class__.__name__

        if cls == "ToolMessage":
            print(f"[ToolMessage]: {msg.content}")

        elif cls == "AIMessage":
            # tool_calls может быть, а content пустым
            if msg.content:
                print(f"[AIMessage]: {msg.content}")
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"  ↳ ToolCall: {tc['name']}({tc['args']})")

    print("\n=== Финальный результат агента ===")

    # 3. Приоритет: последний ToolMessage
    for msg in reversed(messages):
        if msg.__class__.__name__ == "ToolMessage":
            print(msg.content)
            return msg.content

    # 4. Фолбэк: последний AIMessage с текстом
    for msg in reversed(messages):
        if msg.__class__.__name__ == "AIMessage" and msg.content:
            print(msg.content)
            return msg.content

    # 5. Если вдруг ничего нет
    print("⚠️ Агент не вернул текстовый результат")
    return ""
run_agent_and_get_final( '10+1')

def run_agent_and_get(user_input: str) -> str:
    """
    Запускает агента, печатает ход рассуждений и
    возвращает финальный результат (ToolMessage или AIMessage)
    """

    # 1. Вызов агента
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]}
    )
    print(result)
    messages = result.get("messages", [])

    # 3. Приоритет: последний ToolMessage
    for msg in reversed(messages):
        if msg.__class__.__name__ == "ToolMessage":
            print(msg.content)
            return msg.content
        elif msg.__class__.__name__ == "AIMessage":
            return msg.content

    # 5. Если вдруг ничего нет
    print("⚠️ Агент не вернул текстовый результат")
    return ""

