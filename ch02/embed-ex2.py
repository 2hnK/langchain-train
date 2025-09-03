from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("./test.txt", encoding="utf-8")
doc = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(doc)

model = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings = model.embed_documents([chunk.page_content for chunk in chunks])

print(embeddings)

"""
한 번에 하나씩 임베딩하기보다 동시 임베딩하는 편이 더 좋다.
모델 구성상 동시 임베딩이 더 효율적이기 때문이다.
결과는 여러 개의 숫자 리스트를 포함하는 리스트로, 각 내부 리스트는 앞서 설명한 벡터로 표현한다.

embeddings = [
    [0.1, -0.2, 0.5, ...], # 청크1의 임베딩 벡터
    [0.3, 0.1, -0.4, ...], # 청크2의 임베딩 벡터
    [0.2, -0.1, 0.8, ...], # 청크3의 임베딩 벡터
    # ... 더 많은 청크들
]

"""
