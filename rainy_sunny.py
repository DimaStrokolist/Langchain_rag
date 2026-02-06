import random
from typing import Literal
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


class UserState(TypedDict):
    name: str

def weather(userState):
    return userState
def rainy_weather(userState):
  return {"name":userState["name"]+ " Its going to rain today. Carry an umbrella."}
def sunny_weather(userState):
  return {"name":userState["name"]+ " Sun"}

def forecast_weather(userState)->Literal["rainy", "sunny"]:
  if random.random() < 0.5:
    return "rainy"
  else:
    return "sunny"


workflow = StateGraph(UserState)

workflow.add_node("weather", weather)
workflow.add_node("rainy", rainy_weather)
workflow.add_node("sunny", sunny_weather)

workflow.add_edge(START, "weather")
workflow.add_conditional_edges("weather", forecast_weather)
workflow.add_edge("rainy", END)
workflow.add_edge("sunny", END)

app = workflow.compile()

graph_image = app.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(graph_image)

result = app.invoke({"name":"какая погода?"})

print(result)

#написчать граф самому по аналогии этому, но в ручную