"""KRX 규정 마크다운을 파싱하여 SQLite FTS5 데이터베이스로 인덱싱한다.

Docker/Milvus 없이 순수 Python + SQLite만으로 동작하는 RAG 인덱서.
폐쇄망에서도 외부 의존성 없이 사용 가능.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from pathlib import Path

from agent.config import settings

logger = logging.getLogger(__name__)


def _parse_frontmatter(content: str) -> tuple[dict[str, str], str]:
    """마크다운 프론트매터(YAML)를 파싱한다.

    Returns:
        (메타데이터 딕셔너리, 본문 텍스트)
    """
    meta: dict[str, str] = {}
    body = content

    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            # 프론트매터 파싱
            for line in parts[1].strip().split("\n"):
                if ":" in line:
                    key, val = line.split(":", 1)
                    meta[key.strip()] = val.strip().strip("'\"")
            body = parts[2]

    return meta, body


def _detect_market(filepath: str, meta: dict[str, str]) -> str:
    """시장 분류를 감지한다."""
    text = filepath + meta.get("분류", "") + meta.get("제목", "")
    if "유가증권" in text:
        return "유가증권시장"
    elif "코스닥" in text:
        return "코스닥시장"
    elif "코넥스" in text:
        return "코넥스시장"
    return "기타"


def _detect_reg_type(meta: dict[str, str]) -> str:
    """규정 유형을 감지한다."""
    title = meta.get("제목", "")
    if "시행세칙" in title:
        return "시행세칙"
    elif "상장규정" in title:
        return "상장규정"
    elif "공시규정" in title:
        return "공시규정"
    elif "업무규정" in title:
        return "업무규정"
    return "기타"


def _chunk_by_article(body: str, regulation_name: str, market: str) -> list[dict]:
    """마크다운 본문을 조(article) 단위로 청킹한다.

    ##### 제N조(제목) 패턴으로 분리하며, 각 청크에 메타데이터를 부착한다.
    """
    chunks: list[dict] = []

    # 장/절 헤더 추적
    current_chapter = ""
    current_section = ""

    # ##### 제N조 또는 ## 제N장 패턴으로 분리
    lines = body.split("\n")
    current_article_title = ""
    current_article_lines: list[str] = []

    def _flush():
        if current_article_lines:
            text = "\n".join(current_article_lines).strip()
            if text and len(text) > 10:
                chunks.append({
                    "regulation_name": regulation_name,
                    "market": market,
                    "chapter": current_chapter,
                    "section": current_section,
                    "article_title": current_article_title or current_chapter,
                    "text": text,
                })

    for line in lines:
        stripped = line.strip()

        # 장 헤더: ## 제N장 ...
        if re.match(r"^#{1,3}\s*제\d+장", stripped):
            current_chapter = re.sub(r"^#+\s*", "", stripped)
            continue

        # 절 헤더: ### 제N절 ...
        if re.match(r"^#{1,4}\s*제\d+절", stripped):
            current_section = re.sub(r"^#+\s*", "", stripped)
            continue

        # 조 헤더: ##### 제N조(제목) 또는 ##### 제N조의N(제목)
        if re.match(r"^#{1,6}\s*제\d+조", stripped):
            _flush()
            current_article_title = re.sub(r"^#+\s*", "", stripped)
            current_article_lines = [current_article_title]
            continue

        current_article_lines.append(line)

    _flush()

    # 청크가 없으면 전체 텍스트를 하나의 청크로
    if not chunks and body.strip():
        # 큰 본문은 적당히 분할
        text = body.strip()
        chunk_size = 2000
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i:i + chunk_size]
            if chunk_text.strip():
                chunks.append({
                    "regulation_name": regulation_name,
                    "market": market,
                    "chapter": "",
                    "section": "",
                    "article_title": f"본문 (파트 {i // chunk_size + 1})",
                    "text": chunk_text,
                })

    return chunks


def build_index(regulations_dir: str | None = None, db_path: str | None = None) -> int:
    """규정 마크다운 파일들을 읽어 SQLite FTS5 인덱스를 구축한다.

    Args:
        regulations_dir: 규정 마크다운 폴더 경로
        db_path: SQLite DB 파일 경로

    Returns:
        인덱싱된 총 청크 수
    """
    regulations_dir = regulations_dir or settings.regulations_dir
    db_path = db_path or settings.db_path

    logger.info("인덱싱 시작: %s → %s", regulations_dir, db_path)

    # DB 디렉토리 생성
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 기존 테이블 삭제 후 재생성
    cur.execute("DROP TABLE IF EXISTS regulations_fts")
    cur.execute("DROP TABLE IF EXISTS regulations")

    # 원본 데이터 테이블
    cur.execute("""
        CREATE TABLE regulations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            regulation_name TEXT NOT NULL,
            market TEXT NOT NULL,
            chapter TEXT DEFAULT '',
            section TEXT DEFAULT '',
            article_title TEXT DEFAULT '',
            text TEXT NOT NULL
        )
    """)

    # FTS5 가상 테이블 (한국어 토크나이저: trigram)
    cur.execute("""
        CREATE VIRTUAL TABLE regulations_fts USING fts5(
            regulation_name,
            market,
            article_title,
            text,
            content='regulations',
            content_rowid='id',
            tokenize='trigram'
        )
    """)

    # 자동 동기화 트리거
    cur.execute("""
        CREATE TRIGGER regulations_ai AFTER INSERT ON regulations BEGIN
            INSERT INTO regulations_fts(rowid, regulation_name, market, article_title, text)
            VALUES (new.id, new.regulation_name, new.market, new.article_title, new.text);
        END
    """)

    total_chunks = 0

    # 마크다운 파일 순회
    for root, _dirs, files in os.walk(regulations_dir):
        for fname in files:
            if not fname.endswith(".md") or fname == "README.md":
                continue

            filepath = os.path.join(root, fname)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                logger.warning("파일 읽기 실패: %s (%s)", filepath, e)
                continue

            meta, body = _parse_frontmatter(content)
            regulation_name = meta.get("제목", os.path.splitext(fname)[0])
            market = _detect_market(filepath, meta)

            chunks = _chunk_by_article(body, regulation_name, market)

            for chunk in chunks:
                cur.execute(
                    """INSERT INTO regulations
                       (regulation_name, market, chapter, section, article_title, text)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        chunk["regulation_name"],
                        chunk["market"],
                        chunk["chapter"],
                        chunk["section"],
                        chunk["article_title"],
                        chunk["text"],
                    ),
                )

            total_chunks += len(chunks)
            logger.info("  %s: %d 청크", regulation_name, len(chunks))

    conn.commit()

    # 통계
    cur.execute("SELECT COUNT(*) FROM regulations")
    count = cur.fetchone()[0]
    logger.info("인덱싱 완료: 총 %d 청크 (DB: %d rows)", total_chunks, count)

    conn.close()
    return total_chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    n = build_index()
    print(f"\n✅ 인덱싱 완료: {n}개 청크")
