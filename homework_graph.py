from langgraph.graph import StateGraph, START, END
from typing import TypedDict


class UserState(TypedDict):
    number: int

def plus1(userState):
    return {"number": userState["number"] + 1}

def minus2(userState):
    return {"number": userState["number"] - 2}

def multiply2(userState):
    return {"number": userState["number"] * 2}

workflow = StateGraph(UserState)

workflow.add_node("+",plus1)
workflow.add_node("-",minus2)
workflow.add_node("*",multiply2)

workflow.add_edge(START,"+")
workflow.add_edge("+","-")
workflow.add_edge("-","*")
workflow.add_edge("*",END)

app = workflow.compile()

result = app.invoke({"number":5})

print(result)
