from langchain.messages import RemoveMessage
from langchain_core.messages import HumanMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, after_model
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from typing import Any
from langchain.messages import RemoveMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain.tools import tool

llm = ChatOllama(
    model="llama3.2",
    temperature=0.3,
    base_url="http://localhost:11434")


@after_model
def delete_old_messages2(state: AgentState, runtime: Runtime) -> dict | None:
    """Remove old messages to keep conversation manageable."""
    messages = state["messages"]
    if len(messages) > 2:
        # remove the earliest two messages
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
    return None

class CustomState(AgentState):
    user_id: str
    preferences: dict

@before_model
def reset_messages_keep_state(state: CustomState, runtime: Runtime):
    system = [m for m in state["messages"] if isinstance(m, SystemMessage)]

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *system,
        ]
    }
from langchain.agents.middleware import after_model
from langchain.messages import ToolMessage

@tool
def reset_memory() -> str:
    """Полностью очистить память диалога"""
    return "__RESET_MEMORY__"

@before_model
def handle_reset_tool(
    state: AgentState,
    runtime: Runtime,
) -> dict[str, Any] | None:

    messages = state["messages"]

    if not messages:
        return None

    last = messages[-1]

    if isinstance(last, ToolMessage) and last.content == "__RESET_MEMORY__":
        system_messages = [
            m for m in messages if isinstance(m, SystemMessage)
        ]

        if system_messages:
            return {
                "messages": [
                    RemoveMessage(id=REMOVE_ALL_MESSAGES),
                    system_messages[0],
                ]
            }

        return {
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]
        }

    return None
@before_model
def reset_on_user_command(
    state: AgentState,
    runtime: Runtime,
) -> dict[str, Any] | None:

    messages = state["messages"]

    if not messages:
        return None

    last = messages[-1]

    if isinstance(last, HumanMessage) and last.content.strip() == "/reset":
        system_messages = [
            m for m in messages if isinstance(m, SystemMessage)
        ]

        if system_messages:
            return {
                "messages": [
                    RemoveMessage(id=REMOVE_ALL_MESSAGES),
                    system_messages[0],
                ]
            }

        return {
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]
        }

    return None

@before_model
def reset_memory_keep_system(
    state: AgentState,
    runtime: Runtime
) -> dict[str, Any]:
    """
    Полностью очищает память перед моделью,
    но оставляет system message (иначе Ollama упадёт)
    """

    system_messages = [
        m for m in state["messages"]
        if isinstance(m, SystemMessage)
    ]

    # если system почему-то нет — просто чистим всё
    if not system_messages:
        return {
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]
        }

    system_msg = system_messages[0]

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            system_msg,
        ]
    }

agent = create_agent(
    llm,
    tools = [reset_memory],
    middleware=[handle_reset_tool, reset_on_user_command],
    checkpointer=InMemorySaver(),
    system_prompt=(
        # "Ты ассистент. Команда /reset очищает память.",
        "Если пользователь хочет отчистить пямять вызови tool reset_memory"
    )
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": "hi, my name is bob"}, config)
# agent.invoke({"messages": "write a short poem about cats"}, config)
# agent.invoke({"messages": "now do the same but for dogs"}, config)
# agent.invoke({"messages": "/reset"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)
final_response["messages"][-1].pretty_print()
# agent.invoke({"messages": "очисти память "}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

print(final_response["messages"][-1].content)
"""
================================== Ai Message ==================================

Your name is Bob. You told me that earlier.
If you'd like me to call you a nickname or use a different name, just say the word.
"""
#дз - https://metanit.com/python/tutorial/3.6.php