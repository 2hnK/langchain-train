from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain

from config import get_model

template = ChatPromptTemplate.from_messages(
    [
        ("system", " 당신은 친절한 친구입니다."),
        ("human", "{question}"),
    ]
)

model = get_model()

chatbot = template | model

for part in chatbot.stream({"question": "LLM은 어디서 제공하나요?"}):
    print(part)
