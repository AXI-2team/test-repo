import json
import pymysql
from langchain_openai import OpenAIEmbeddings
from app.config import settings
import math


"""

user <-> service <-> DB store(handler) <-> DB

                         ^
                         |
                     DB config

구조 : 사용자 -> 검색어 입력 -> openai embedding생성 -> mariadb 저장된 벡터와 비교 -> 유사한 문서 반환 (즉, 검색 엔진)

                     """

class MariaDBVectorStore:
    # 임베딩 : 문자를 숫자로 바꿈 > 유사한 문서를 찾을 수 있게 됨
    def __init__(self, table_name="ai_vector_store", embedding_model='text-embedding-3-small'):
        self.table = table_name
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=settings.OPENAI_API_KEY
        )

        self._create_table()

    
    def _create_table(self):
        sql = f"""
        create table if not exists {self.table} (
            id bigint primary key auto_increment,
            text text not null,
            embedding JSON not null,
            metadata JSON,
            created_at timestamp default current_timestamp
        );

        """

        conn =self.connect()
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()
        conn.close()
        

    def connect(self):
        return pymysql.connect(
            host=settings.DB_HOST,
            user=settings.DB_USER,
            port=int(settings.DB_PORT),
            password=settings.DB_PASS,
            database=settings.DB_NAME,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
            # cursorclass=pymysql.cursors.DictCursor
        )
    
    # 문서 저장
    def add_document(self, documents):
        conn = self.connect()
        cur = conn.cursor()

        for doc in documents:
            embedding = self.embeddings.embed_query(doc.page_content)
            sql = f"""
                insert into {self.table} (text, embedding, metadata) values (%s, %s, %s)
            """
            cur.execute(sql, (doc.page_content, json.dumps(embedding), json.dumps(doc.metadata)))

        conn.commit()
        conn.close()
    
    # 코사인 유사도 검색
    def _cosine_similarity(self, v1, v2):
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

    # 기본 검색
    def similarity_search(self, query, k=5, filter=None):
        
        # 1) 쿼리 임베딩 생성
        query_vec = self.embeddings.embed_query(query)

        # 모든 문서 가져오기
        conn = self.connect()
        cur = conn.cursor()

        sql = f"select id, text, embedding, metadata from {self.table}"

        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()

        scored_docs = []
        for row in rows:
            vec = json.loads(row['embedding'])
            score = self._cosine_similarity(query_vec, vec)

            # metadata 필터적용
            if filter:
                match = True
                for key, val in filter.items():
                    if not row['metadata']:
                        match = False
                        break
                    meta = json.loads(row['metadata'])
                    if key not in meta or meta[key] != val:
                        match = False
                        break
                
                if not match:
                    continue
            
            scored_docs.append((score, row))
        
        # 내림차순 정렬
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        scored_docs = scored_docs[:k]

        #langchain document로 변환
        results = []

        from langchain_core.documents import Document

        for score, row in scored_docs:
            doc = Document(
                page_content = row['text'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
            doc.metadata['score'] = score
            results.append(doc)

        return results
    
    def similarity_search_with_score(self, query, k=5):
        docs = self.similarity_search(query, k=k)
        return [(d, d.metadata['score']) for d in docs]