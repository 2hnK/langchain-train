from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain

import asyncio
from config import get_model


# 구성요소
template = ChatPromptTemplate.from_messages(
    [
        ("system", " 당신은 친절한 친구입니다."),
        ("human", "{question}"),
    ]
)

model = get_model()


# 함수로 결합한다.
# 데코레이터 @chain을 추가해 작성한 함수에 Runnable 인터페이스를 추가한다.
@chain
async def chatbot(values):
    prompt = await template.ainvoke(values)
    return await model.ainvoke(prompt)


async def main():
    result = await chatbot.ainvoke({"question": "LLM은 어디서 제공하나요?"})
    print(result.content)