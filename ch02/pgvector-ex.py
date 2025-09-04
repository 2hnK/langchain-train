import uuid

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

connection = "postgresql+psycopg://langchain:1234@localhost:6024/langchain"

raw_documents = TextLoader("./test.txt", encoding="utf-8").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)

embeddings_model = OpenAIEmbeddings()

db = PGVector.from_documents(documents, embeddings_model, connection=connection)


ids = [str(uuid.uuid4()), str(uuid.uuid4())]
db.add_documents(
    [
        Document(
            page_content="there are cats int the pond",
            metadata={"location": "pond", "topic": "animals"},
        ),
        Document(
            page_content="ducks are also found in the pond",
            metadata={"location": "pond", "topic": "animals"},
        ),
    ],
    ids=ids,
)

results = db.similarity_search("query", k=2)
print(results)

# db.delete(ids=[1])s

"""
데이터가 지속적으로 변화하는 상황에서 벡터DB는 데이터를 다시 인덱싱한다.
이로 인해 계산 비용이 지속적으로 발생하고, 기존 콘텐츠가 중복되기도 한다.
"""
