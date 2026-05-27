"""KRX 규정 RAG 챗봇 실행 스크립트.

1단계: 규정 마크다운 → SQLite FTS5 인덱싱 (최초 1회)
2단계: FastAPI 서버 시작

사용법:
  python run.py
"""

import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    # 프로젝트 루트를 sys.path에 추가
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from agent.config import settings

    # ─── 1단계: 인덱싱 ───
    db_path = settings.db_path
    if not os.path.exists(db_path):
        logger.info("=" * 60)
        logger.info("📦 규정 데이터 인덱싱 시작 (최초 1회)")
        logger.info("=" * 60)

        from agent.indexer import build_index
        count = build_index()
        logger.info("✅ 인덱싱 완료: %d개 청크", count)
    else:
        logger.info("✅ 기존 인덱스 사용: %s", db_path)

    # ─── API Key 확인 ───
    if not settings.google_api_key:
        logger.warning("⚠️  GOOGLE_API_KEY가 설정되지 않았습니다.")
        logger.warning("   .env 파일에 GOOGLE_API_KEY=AIza... 를 추가하세요.")
        logger.warning("   또는 환경변수로 설정: set GOOGLE_API_KEY=AIza...")

    # ─── 2단계: 서버 시작 ───
    logger.info("=" * 60)
    logger.info("🚀 KRX 규정 RAG 챗봇 서버 시작")
    logger.info("   URL: http://localhost:8000")
    logger.info("   API 문서: http://localhost:8000/docs")
    logger.info("=" * 60)

    import uvicorn
    uvicorn.run(
        "agent.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
