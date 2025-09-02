import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def get_api_key():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "환경 변수 OPENAI_API_KEY가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY를 추가하거나 OS 환경 변수로 설정하세요."
        )

    return api_key


def get_model():
    return ChatOpenAI(model="gpt-5-nano", api_key=get_api_key())
