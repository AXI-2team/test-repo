from langchain_core.documents import Document
from app.maria_vector_db import MariaDBVectorStore
from app.sample_data import sample_documents
from app.config import settings
import pymysql


TOP_K = 5
SIMILARITY_THRESHOLD = 0.7 # 유사도가 0.7 이상인 문서만 사용하겠다.

## rag에서 openai에서 embeddingmodel과 유사도를 찾아서 llm을 이용하여 답변되도록 2번 이용된다. 
# 청킹 : 500자씩 겹치는건 100자씩 보통
# csv -> 청킹 -> 임베딩 -> db 저장
# query vector(유사도)를 하나 생성, vector a(저장), b(질문) 사이의 cos값, a와 b가 비슷하면(각도가 0도가 됨) cos = 1 
# (x, y, z) 3차원에 하나의 차원을 추가 한다고 한다면. (t, 위치) -> 이렇게 차원축소 가능함.
# 임베딩에서 말하는 차원 : 의미를 표현하는 좌표축의 개수
# 임베딩은 글자 구분이 아닌 의미 구분을 위해 고차원 벡터를 사용함.
# 따라서 RAG의 핵심 개념은 텍스트를 임베딩해서 고차원 공간의 점으로 생성, 가까운 점일 수록 의미가 비슷하고 유사 문서로 검색됨
# 차원이 많을 수록 모델은 의미를 더 풍부하게 표현 가능 > 저장공간 많아야하고 속도 저하
    # 그래서 실무에서는 벡터DB가 등장한다. 100만개 벡터 중 전부 비교해서 각 문서당 1536차원 계산하는 것이 아닌 100만개 벡터를 전부 비교하지 않고
    # 비슷할 것 같은 후보만 찾아 사용함.

"""
현재 코드 : 책 100만권, 질문-> 책 100만권 전부 읽음 -> 답 제공

벡터 DB : 도서관 색인 -> 관련 있을 법한 책 20권만 추림 -> 그것만 읽어서 답 제공

-> 그래서 실무에서는 차원 수보다 검색알고리즘이 더 중요하다. 

임베딩 차원이 높을수록 의미 표현력은 좋아지지만, 유사도 계산 비용도 증가합니다. 
따라서 대규모 데이터에서는 벡터 DB의 인덱싱 기법을 사용해 검색 성능을 확보합니다. 

"""


class VectorService:
    def __init__(self):
        self.store = MariaDBVectorStore()


    # (1) 샘플 문서 저장
    def load_sample_docs(self):
        docs = [
            Document(page_content=d['text'], metadata=d['metadata'])
            for d in sample_documents
        ]

        self.store.add_document(docs)

        return len(docs)
    

    # (2) basic_serce : 기본 similaryitySearch
    def basic_search (self, query: str):
        return self.store.similarity_search(query, k=5)
    

    # (3) threshold + filterExpression
    def search2(self, query:str):
        filter_ = {"author": "john", "article_type": "blog"}
        return self.store.similarity_search(query, k=TOP_K, filter=filter_)
    

    # score 출력
    def search_with_scores(self, query: str):
        return self.store.similarity_search_with_score(query, k=2)
    

    # native sql 조회
    def native_query(self):
        conn = self.store.connect()
        cur = conn.cursor()
        cur.execute(f"select * from {settings.DB_TABLE} limit 20")
        rows = cur.fetchall()
        conn.close()
        return rows
