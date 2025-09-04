import uuid

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.indexes import SQLRecordManager, index
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
collection_name = "my_docs"
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
namespace = "my_docs_namespace"

# DB연결 및 테이블 생성
vectorstore = PGVector(
    embeddings=embeddings_model,  # 임베딩 모델
    collection_name=collection_name,  # 테이블 이름
    connection=connection,  # 데이터베이스 연결 정보
    use_jsonb=True,
)

"""
SQLRecordManager의 역할
- 중복 방지: 같은 source_id를 가진 문서가 이미 있는지 확인
- 버전 관리: 문서가 수정되었을 때 기존 버전을 삭제하고 새 버전을 저장
- 증분 업데이트: `cleanup="incremental"` 옵션과 함께 사용하여 효율적인 문서 관리

namespace의 역할
- 논리적 격리: 같은 데이터베이스 내에서 서로 다른 문서 그룹을 분리
- 중복 관리 범위 설정: 같은 namespace 내에서만 문서 중복을 확인
- 데이터베이스 테이블 구분: namespace 내에서만 문서 중복을 확인
"""
record_manager = SQLRecordManager(
    namespace,
    db_url="postgresql+psycopg://langchain:langchain@localhost:6024/langchain",
)

"""
스키마(테이블)가 없으면 생성
- uuid: 문서 고유 식별자
- key: 문서 출처 식별자
- namespace: 문서 네임스페이스
- updated_at: 문서 수정 시간
- group_id: 문서 그룹 식별자
"""
record_manager.create_schema()

# 인덱싱할 문서 생성
docs = [
    Document(
        page_content="there are cats in the pond",
        metadata={"id": 1, "source": "cats.txt"},
    ),
    Document(
        page_content="ducks are also found in the pond",
        metadata={"id": 2, "source": "ducks.txt"},
    ),
]

# 문서 인덱싱 1회차: 새로운 문서
index_1 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
)
print("인덱싱 1회차:", index_1)

# 문서 인덱싱 2회차: 변경 없는 문서
index_2 = index(
    docs, # 인덱싱할 문서 목록
    record_manager, # 인덱싱 상태 관리 객체: SQLRecordManager
    vectorstore, # 벡터 저장소: PGVector
    cleanup="incremental", # 인덱싱 후 중복 문서 정리 방법
    source_id_key="source", # 문서 출처 식별자 키
)
print("인덱싱 2회차:", index_2)

docs[0].page_content = "I just modified this document!"

# 문서 인덱싱 3회차: 수정된 문서
# 문서를 수정하면 새 버전을 저장하고, 출처가 같은 기존 문서는 삭제한다.
index_3 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
)
print("인덱싱 3회차:", index_3)
