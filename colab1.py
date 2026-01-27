from langgraph.graph import StateGraph
from typing import TypedDict
from IPython.display import Image, display
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import ToolMessage
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama


def node1(userState):
  return {"name":userState["name"]+" I reached Node1."}
def node2(userState):
  return {"name":userState["name"] + " And now at Node2."}

llm = ChatOllama(
    model="llama3.2",
    temperature=0.3,
    base_url="http://localhost:11434")

def node_llm(userState):
    result = llm.invoke('say hello')
    return {"name": result.content}

# Create a new Graph
class UserState(TypedDict):
    name: str

workflow = StateGraph(UserState)

# Add the nodes
workflow.add_node("node_1", node1)
workflow.add_node("node_2", node2)
workflow.add_node("node_llm", node_llm)

# Add the Edges
workflow.add_edge("node_1", "node_2")
workflow.add_edge("node_2", "node_llm")
workflow.set_entry_point("node_1")
workflow.set_finish_point("node_llm")

#Run the workflow
app = workflow.compile()


# Для сохранения в файл вместо отображения
graph_image = app.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(graph_image)

result = app.invoke({"name":"hello"})

print(result)