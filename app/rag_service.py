from openai import OpenAI
from app.config import settings
from app.maria_vector_db import MariaDBVectorStore

class RagService:
    def __init__(self):
        self.vector_store = MariaDBVectorStore()
        self.llm = OpenAI(api_key=settings.OPENAI_API_KEY)

    

    def generate_answer(self, question: str):
        
        # 1. vector 검색
        docs = self.vector_store.similarity_search(question, k=5)
        if not docs:
            return '검색된 문서가 없어어 RAG응답을 만들 수 없음'
        
        # 2. 검색된 문서를 하나의 context로 병합
        context_text = '\n\n'.join([f'-{d.page_content}' for d in docs])


        # 3. 모델에 전달할 프롬프트 구성
        prompt = f"""
            당신은 검색문서로 기반으로 답변하는 RAG 전문가이다. 
            
            [관련 문서들]
            {context_text}

            [사용자 질문]
            {question}

            위 문서를 근거로 정확하고 알아듣기 쉽게 한국어로 답변 생성하세요.
        """

        # 4. llm호출하기
        completion = self.llm.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {"role": "system", "content": "당신은 문서기반 RAG QA전문가이다."},
                {"role": "user", "content": prompt},
            ]
        )

        return completion.choices[0].message.content


