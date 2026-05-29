# KRX 규정 RAG 챗봇 (v2 고도화 버전)

한국거래소(KRX) 규정 및 관련 법령을 검색하고 AI가 답변하는 고도화된 RAG(Retrieval-Augmented Generation) 챗봇입니다.

## 🚀 주요 고도화 특징 (v2)

- **하이브리드 검색 (Hybrid Search)**: SQLite FTS5(키워드)와 BGE-M3(의미) 검색을 결합하여 정확도 극대화
- **고성능 임베딩**: `BAAI/bge-m3` 모델을 통한 다국어/법률 특화 의미 추출
- **RRF(Reciprocal Rank Fusion)**: 서로 다른 검색 알고리즘의 결과를 최적으로 결합하는 순위 재정렬 적용
- **조(Article) 단위 계층적 청킹**: 규정의 구조([편>장>절>조])를 유지하여 AI의 맥락 이해도 향상
- **핵심 법령 포함**: 자본시장법, 상법 등 KRX 규정과 연계된 10대 주요 법령 데이터 구축
- **폐쇄망 최적화**: 로컬 임베딩 모델(`local_model/`) 사용으로 인터넷 없이도 벡터 검색 가능

## 🛠 임베딩 및 검색 기술 상세

### 1. 벡터 임베딩 프로세스
- **모델**: `BAAI/bge-m3`
- **전략**: 조문 제목과 본문을 결합하여 임베딩을 생성함으로써 검색 품질을 향상시켰습니다.
- **저장**: SQLite에 `float32` 벡터를 BLOB 형태로 저장하여 별도의 벡터 DB 없이 가볍게 동작합니다.

### 2. 하이브리드 검색 아키텍처
1. **쿼리 확장**: `synonyms.py`를 이용해 사용자 질문의 키워드를 금융 도메인에 맞게 확장합니다.
2. **병렬 검색**: 
   - **FTS5**: Trigram 기반의 정확한 키워드/조항 번호 매칭
   - **Vector**: 코사인 유사도 기반의 의미적 매칭
3. **결합(RRF)**: 두 결과의 순위를 역수 합산 방식으로 결합하여 최종 Top-K를 산출합니다.

### 3. 계층적 데이터 구조
- 모든 조문은 상위 장/절 정보를 메타데이터로 보유합니다.
- AI는 답변 생성 시 "제N편 제M장 제X조"와 같은 정확한 출처 정보를 함께 제공받습니다.

## 프로젝트 구조

```
krx-reg-chatbot/
├── agent/
│   ├── app.py              # FastAPI 엔드포인트 (SSE 스트리밍)
│   ├── graph.py             # LangGraph 기반 에이전트 워크플로우
│   ├── indexer.py           # 마크다운 → SQLite 인덱서
│   ├── llm_client.py        # Google Gemini API 클라이언트
│   └── tools/
│       ├── regulation_search.py  # 하이브리드(FTS5 + Vector) 검색 도구
│       └── synonyms.py           # 동의어 사전 및 쿼리 확장
├── data/
│   ├── krx_rag.db          # 원본 데이터 + 벡터 임베딩 통합 DB
│   └── krx_regulations/     # 원본 마크다운 파일
├── local_model/            # BGE-M3 로컬 모델 가중치
├── generate_embeddings_real.py # 벡터 임베딩 생성 스크립트
├── requirements.txt         # 의존성 패키지
└── README.md
```

## 아키텍처

```
사용자 질문
    ↓
[Query Expansion] 동의어 확장
    ↓
[Hybrid Search] ───┐
    ├── FTS5 (Keyword)
    └── Vector (Semantic)
    ↓
[RRF Ranking] 결과 재정렬
    ↓
[LLM Reasoning] Gemini-2.0-Flash가 검색된 조문을 분석하여 답변 생성 (스트리밍)
```

### v1 vs v2 비교

| 구성요소 | v1 (기본) | v2 (고도화) |
|---------|------|------|
| **검색 방식** | FTS5 전문 검색 | **Hybrid (FTS5 + Vector)** |
| **임베딩 모델** | 없음 | **BAAI/bge-m3 (Local)** |
| **순위 산정** | 단순 BM25 | **RRF (Reciprocal Rank Fusion)** |
| **청킹 전략** | 단순 길이 분할 | **조(Article) 단위 계층 구조 유지** |
| **데이터 범위** | KRX 규정 15종 | **KRX 규정 + 자본시장법 등 10대 법령** |

## 빠른 시작

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 임베딩 생성 (최초 1회)
로컬 모델을 통해 규정 데이터를 벡터화합니다.
```bash
python generate_embeddings_real.py
```

### 3. 실행
```bash
python run.py
```

---
> 본 프로젝트는 한국거래소 규정 및 관련 법령에 대한 AI 기반 질의응답을 지원하며, 모든 처리는 로컬 DB와 보안이 강화된 API 호출을 통해 이루어집니다.
