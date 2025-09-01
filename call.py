from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "환경 변수 OPENAI_API_KEY가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY를 추가하거나 OS 환경 변수로 설정하세요."
    )

model = ChatOpenAI(model="gpt-5-nano", api_key=api_key)

result = model.invoke("하늘이")
print(result)
print(result.content)