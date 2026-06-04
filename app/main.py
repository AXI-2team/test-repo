import sys
import os
import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.service import VectorService
from app.rag_service import RagService

service = VectorService()
rag = RagService()

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="RAG Demo",
    page_icon="🤖",
    layout="wide"
)


st.title("🤖[GITHUB] RAG Demo")
st.caption("MariaDB VectorStore + OpenAI Embedding + RAG")

# =========================
# Vector Store 관리
# =========================
with st.expander("📦 Vector Store 관리", expanded=True):
    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button("샘플 문서 저장", use_container_width=True):
            with st.spinner("문서 임베딩 및 저장 중..."):
                count = service.load_sample_docs()

            st.success(f"{count}개 문서 저장 완료")

    with col2:
        st.info("샘플 문서를 임베딩하여 MariaDB VectorStore에 저장합니다.")

st.divider()

# =========================
# 유사도 검색
# =========================
st.subheader("🔍 Similarity Search")

search_col1, search_col2 = st.columns([1, 3])
with search_col1:
    method = st.selectbox(
        "검색 방식",
        ["basic_search", "search2"]
    )
with search_col2:
    query = st.text_input(
        "검색어 입력",
        placeholder="예) 파이썬"
    )

if st.button("검색 실행", use_container_width=True):
    with st.spinner("검색 중..."):
        if method == "basic_search":
            results = service.basic_search(query)
        else:
            results = service.search2(query)

    st.success(f"{len(results)}건 검색 완료")

    for idx, doc in enumerate(results, start=1):
        with st.container(border=True):
            st.markdown(f"### 📄 문서 {idx}")
            st.markdown("**본문**")
            st.write(doc.page_content)

            st.markdown("**Metadata**")
            st.json(doc.metadata)

st.divider()

# =========================
# Score 확인
# =========================

st.subheader("📊 Similarity Score")
score_query = st.text_input(
    "Score 검색어",
    placeholder="예) LangChain"
)

if st.button("Score 확인", use_container_width=True):
    with st.spinner("유사도 계산 중..."):
        results = service.search_with_scores(score_query)

    for idx, (doc, score) in enumerate(results, start=1):
        with st.container(border=True):
            st.markdown(f"### 🏆 Rank {idx}")
            st.write(doc.page_content)
            st.progress(float(score))
            st.metric(
                label="Similarity Score",
                value=f"{score:.4f}"
            )

st.divider()

# =========================
# Native SQL
# =========================

st.subheader("🗄 Native SQL 조회")
if st.button("조회 실행", use_container_width=True):

    rows = service.native_query()
    st.dataframe(
        rows,
        use_container_width=True
    )

st.divider()

# =========================
# RAG
# =========================

st.header("💬 RAG 질의응답")
question = st.text_area(
    "질문 입력",
    height=120,
    placeholder="예) LangChain이 무엇인가요?"
)

if st.button("RAG 실행", type="primary", use_container_width=True):
    with st.spinner("LLM 응답 생성 중..."):
        answer = rag.generate_answer(question)
    st.success("답변 생성 완료")
    with st.container(border=True):
        st.markdown("### 🤖 답변")
        st.write(answer)