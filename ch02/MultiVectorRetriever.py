import uuid

from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import MultiVectorRetriever
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.stores import InMemoryStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
collection_name = "summaries"
embeddings_model = OpenAIEmbeddings()

# 문서 로드
loader = TextLoader("./test.txt", encoding="utf-8")
docs = loader.load()

print("length of loaded docs: ", len(docs[0].page_content))

# 문서 분할
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

prompt_text = "다음 문서의 요약을 생성하세요:\n\n{doc}"

"""
*체인 구성 순서 규칙*
1. 입력 매핑/전처리
2. 프롬프트 렌더링 - `ChatPromptTemplate()`
3. 모델 호출 - `ChatOpenAI()`
4. 출력 파싱 - `StrOutputParser()`

*lambda 입력 매핑*
lambda를 사용하지 않고 입력값을 구성하면 x가 없기 때문에 오류가 발생한다.
`{"doc": x.page_content}`

*callable 입력 매핑*
callable을 사용하면 실제 입력이 들어오는 실행 시점에 x에서 값을 뽑아 `{doc:...}`를 만들 수 있다.
callable이란 함수처럼 호출될 수 있는 객체이다.
- 함수/람다
- 메서드/빌트인 함수
- 클래스 자체(MyClass() 인스턴스를 생성하는 생성자를 의미)
- call_ 메서드를 가진 객체

*Lambda*
파이썬에서의 lambda는 즉석에서 함수 객체를 만들어내는 문법이다.
f = lambda x: x.page_content
Function<Doc, String> f = x -> x.getPageContent();
"""
prompt = ChatPromptTemplate.from_template(prompt_text)
llm = ChatOpenAI(model="gpt-5-nano")
summarize_chain = {"doc": lambda x: x.page_content} | prompt | llm | StrOutputParser()

# LLM 호출 - batch
summaries = summarize_chain.batch(chunks, {"max_concurrency": 5})

# 벡터 저장소는 하위 청크를 인덱싱하는 데 사용
# k-NN 유사도 검색을 수행해 유사도가 높은 청크를 찾는다.
vectorstore = PGVector(
    embeddings=embeddings_model,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

# 상위 문서를 위한 스토리지 레이어
# 메모리 기반 key-value 저장소
store = InMemoryStore()
id_key = "doc_id"

# 원본 문서를 저장소에 보관하면서 벡터 저장소에 요약을 인덱싱
# PGVector에서 임베딩 유사도 검색 수행
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
    search_kwargs={"k": 3},
)

# 문서와 동일한 길이가 필요하므로 summaries에서 chunks로 변경
doc_ids = [str(uuid.uuid4()) for _ in chunks]

# 각 요약은 doc_id를 통해 원본 문서와 연결
# list[Document(page_content=summary, metadata={id_key: doc_id})]
summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)  # i: 인덱스, s: 요약
]

# 유사도 검색을 위해 벡터 저장소에 문서 요약을 추가
# retriever의 vectorstore 호출하는 이유:
# - retriever의 vectorstore가 바뀌어도 영향을 주지 않기 위함
# - retriever가 내부적으로 참조하는 vectorstore를 추가하려는 의도를 표현하기 위함함
retriever.vectorstore.add_documents(summary_docs)  # PGVector에 저장

# doc_ids를 통해 요약과 연결된 원본 문서를 문서 저장소에 저장
# 이를 통해 먼저 요약을 효율적으로 검색한 다음, 필요할 때 전체 문서를 가져옴
# mset(): 여러 개의 키-값 쌍을 한 번에 저장
retriever.docstore.mset(list(zip(doc_ids, chunks)))  # InMemoryStore에 저장

# 벡터 저장소가 요약을 검색
# 유사도가 높은 Document 리스트를 반환
sub_docs = retriever.vectorstore.similarity_search("darkness", k=3)

print("sub docs (all):")
for i, d in enumerate(sub_docs):
    print(f"[{i}] doc_id={d.metadata.get(id_key)} length={len(d.page_content)}")
    print(d.page_content)

# retriever는 더 큰 원본 문서 청크를 반환
# 유사도 검색 결과의 id_key를 통해 원본 문서를 찾아 반환
retrieved_docs = retriever.invoke("darkness")

print("retrieved docs (all):")
for i, d in enumerate(retrieved_docs):
    print(f"[{i}] length={len(d.page_content)}")
    print(d.page_content)
