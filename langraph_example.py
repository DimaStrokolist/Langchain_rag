from typing import TypedDict
from datetime import date, timedelta
from langgraph.graph import StateGraph, START, END


class UserState(TypedDict):
    name: str
    surname: str
    birth_date: date
    today: date
    age: int
    message: str


def calculate_age(state: UserState) -> dict:
    today = state["today"]
    age = today.year - state["birth_date"].year
    if (today.month, today.day) < (state["birth_date"].month, state["birth_date"].day):
        age -= 1
    return {"age": age}


def check_drive(state: UserState) -> str:
    return "можно" if state["age"] >= 18 else "нельзя"


def generate_success_message(state: UserState) -> dict:
    return {
        "message": f"Поздравляем, {state['name']} {state['surname']}! "
                   f"Вам уже {state['age']} лет и вы можете водить!"
    }


def generate_failure_message(state: UserState) -> dict:
    return {
        "message": f"К сожалению, {state['name']} {state['surname']}, "
                   f"вам ещё только {state['age']} лет и вы не можете водить."
    }


def increment_date(state: UserState) -> dict:
    new_date = state["today"] + timedelta(days=1)
    print(state["today"], "->", new_date)
    return {"today": new_date}


graph = StateGraph(UserState)

graph.add_node("calculate_age", calculate_age)
graph.add_node("success_message", generate_success_message)
graph.add_node("increment_date", increment_date)

graph.add_edge(START, "calculate_age")

graph.add_conditional_edges(
    "calculate_age",
    check_drive,
    {
        "можно": "success_message",
        "нельзя": "increment_date"
    }
)

graph.add_edge("increment_date", "calculate_age")
graph.add_edge("success_message", END)

app = graph.compile()

result = app.invoke(
    {
        "name": "Алексей",
        "surname": "Яковенко",
        "birth_date": date.fromisoformat("2008-02-19"),
        "today": date.today()
    },
    {"recursion_limit": 1000}
)

print(result)

#написать граф и пройти его в длину и ширину
#переписать код вручную
