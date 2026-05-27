# KRX 규정 RAG 챗봇

한국거래소(KRX) 규정을 검색하고 AI가 답변하는 RAG(Retrieval-Augmented Generation) 챗봇입니다.

## 특징

- **Docker/Milvus 불필요**: SQLite FTS5 기반 전문 검색으로 외부 서비스 없이 동작
- **폐쇄망 호환**: 인터넷 없이도 규정 검색 가능 (LLM API만 필요)
- **15개 규정 수록**: 유가증권·코스닥·코넥스 시장의 상장규정, 공시규정, 업무규정 및 시행세칙
- **SSE 스트리밍**: 실시간 토큰 단위 답변 출력
- **Tool Agent Loop**: LangGraph 기반 검색→판단→답변 에이전트 아키텍처

## 프로젝트 구조

```
krx-reg-chatbot/
├── agent/
│   ├── app.py              # FastAPI 엔드포인트 (SSE 스트리밍)
│   ├── config.py            # 환경설정
│   ├── graph.py             # LangGraph StateGraph 정의
│   ├── indexer.py           # 마크다운 → SQLite FTS5 인덱서
│   ├── llm_client.py        # Google Generative AI 클라이언트
│   ├── models.py            # Pydantic 데이터 모델
│   ├── nodes/
│   │   ├── policy.py        # 다음 Action 결정 (LLM)
│   │   ├── executor.py      # Tool 실행 및 Observation 수집
│   │   └── synthesizer.py   # 최종 답변 합성 (LLM 스트리밍)
│   └── tools/
│       ├── tool_runtime.py  # Tool 디스패치
│       └── regulation_search.py  # SQLite FTS5 규정 검색
├── data/
│   └── krx_regulations/     # 규정 마크다운 파일 (15개)
├── frontend/
│   └── index.html           # 채팅 UI
├── .env.example             # 환경변수 템플릿
├── requirements.txt         # Python 의존성
├── run.py                   # 원클릭 실행 스크립트
└── README.md
```

## 빠른 시작

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. API Key 설정

```bash
# .env 파일 생성
copy .env.example .env

# .env 파일에서 GOOGLE_API_KEY 수정
GOOGLE_API_KEY=AIza...
```

### 3-1. 백엔드 실행

```bash
python run.py
```

첫 실행 시 자동으로:
1. 규정 마크다운 15개를 파싱하여 조문 단위(2,245개 청크)로 분할
2. SQLite FTS5 인덱스 구축 (`data/krx_regulations.db`)
3. FastAPI 서버 시작 (http://localhost:8000)

### 3-2. 프론트 실행

새로운 터미널 추가 하시고

```bash
npx serve
```

### 4. 사용

브라우저에서 http://localhost:3000 접속

## 아키텍처

```
사용자 질문
    ↓
[normalize_goal] 목표 정규화
    ↓
[policy] LLM이 다음 행동 결정 ←─────────┐
    ↓                                    │
    ├── CALL_TOOL → [executor] 규정 검색 → budget 체크 →┘
    ├── WRITE_NOTE → [executor] 메모 작성 → budget 체크 →┘
    ├── SYNTHESIZE → [synthesizer] 최종 답변 생성 (스트리밍)
    └── STOP → 즉시 종료
```

### 원본(basic-tool-calling-agent) 대비 변경사항

| 구성요소 | 원본 | 변경 |
|---------|------|------|
| 벡터DB | Milvus (Docker) | SQLite FTS5 (파일 1개) |
| 임베딩 | sentence-transformers | 불필요 (trigram 토크나이저) |
| 검색 도구 | vector_search + web_search | regulation_search |
| 인프라 | Docker Compose (etcd+minio+milvus) | 불필요 |
| 데이터 | 외부 주입 | 마크다운 자동 인덱싱 |

## 수록 규정 목록

### 유가증권시장
- 상장규정 / 상장규정 시행세칙
- 공시규정 / 공시규정 시행세칙
- 업무규정 / 업무규정 시행세칙

### 코스닥시장
- 상장규정 / 상장규정 시행세칙
- 공시규정 / 공시규정 시행세칙
- 업무규정 / 업무규정 시행세칙

### 코넥스시장
- 상장규정
- 공시규정 / 공시규정 시행세칙

## 환경변수

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| GOOGLE_API_KEY | Google Generative AI API 키 (필수) | - |
| LLM_MODEL | 사용할 Gemini 모델 | gemini-2.0-flash |
| MAX_STEPS | 에이전트 최대 루프 횟수 | 5 |
| RAG_TOP_K | 검색 결과 반환 수 | 5 |

## 폐쇄망 설치 가이드

1. 인터넷 PC에서 패키지 다운로드:
```bash
pip download -r requirements.txt -d ./packages
```

2. USB로 전체 프로젝트 폴더 복사

3. 폐쇄망 PC에서 설치:
```bash
pip install --no-index --find-links=./packages -r requirements.txt
```

4. `.env` 파일에 API Key 설정 후 `python run.py` 실행

> 참고: Google Generative AI API 호출을 위해 `generativelanguage.googleapis.com`에 대한 네트워크 접근은 필요합니다.
