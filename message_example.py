from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import ToolMessage
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

@tool(description='используется для прогноза погоды')
def get_weather(city: str) -> str:
    return f"В {city} сейчас солнечно"

llm = ChatOllama(
    model="llama3.2",
    temperature=0.3,
    base_url="http://localhost:11434")

agent = create_agent(
    llm,
    tools = [get_weather])

system_msg = SystemMessage("You are a helpful coding assistant.")

messages = [
    system_msg,
    HumanMessage("How do I create a REST API?")
]
message1 = [
    SystemMessage('Ты сервис прогноза погоды'),
    HumanMessage('Какая погода в Москве?')
]
response = llm.invoke(messages)
print(response)
response1 = (agent.invoke({
    'messages':message1}))
print(response1)

# String content
human_message = HumanMessage("Hello, how are you?")

# Provider-native format (e.g., OpenAI)
human_message = HumanMessage(content=[
    {"type": "text", "text": "Hello, how are you?"},
    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
])

# List of standard content blocks
human_message = HumanMessage(content_blocks=[
    {"type": "text", "text": "Hello, how are you?"},
    {"type": "image", "url": "https://example.com/image.jpg"},
])
responce = llm.invoke(human_message)
print(responce)