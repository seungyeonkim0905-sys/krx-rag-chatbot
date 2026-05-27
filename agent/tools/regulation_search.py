"""SQLite FTS5 기반 KRX 규정 검색 Tool.

Milvus/Docker 없이 순수 SQLite FTS5(trigram)로 한국어 전문 검색을 수행한다.
폐쇄망 환경에서도 외부 의존성 없이 동작한다.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from typing import Any

from agent.config import settings
from agent.models import RegulationSearchHit, RegulationSearchInput, RegulationSearchOutput

logger = logging.getLogger(__name__)


def _ensure_db() -> str:
    """DB 파일 존재를 확인하고, 없으면 인덱싱을 수행한다."""
    db_path = settings.db_path
    if not os.path.exists(db_path):
        logger.info("DB 파일 없음 - 자동 인덱싱 시작")
        from agent.indexer import build_index
        build_index()
    return db_path


def _build_fts_query(query_text: str, market: str | None, reg_type: str | None) -> tuple[str, list]:
    """FTS5 검색 쿼리를 구성한다.

    한국어 trigram 토크나이저에 맞게 검색어를 처리한다.
    """
    # FTS5 쿼리 구성: 핵심 키워드 추출
    # trigram 토크나이저는 3글자 단위로 매칭하므로 원본 텍스트를 그대로 사용
    fts_query = query_text.strip()

    # 시장/규정 유형 필터가 있으면 추가
    conditions = []
    params = []

    if market:
        conditions.append("r.market = ?")
        params.append(market)

    if reg_type:
        conditions.append("r.regulation_name LIKE ?")
        params.append(f"%{reg_type}%")

    return fts_query, conditions, params


def regulation_search(args: dict[str, Any]) -> dict[str, Any]:
    """SQLite FTS5를 사용하여 KRX 규정을 검색한다.

    전략:
    1. FTS5 MATCH (trigram) 검색 시도
    2. 결과 부족 시 키워드를 붙여서 재시도 (예: "상장 요건" → "상장요건")
    3. 그래도 부족하면 LIKE 폴백
    """
    input_params = RegulationSearchInput(**args)
    db_path = _ensure_db()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    _, conditions, filter_params = _build_fts_query(
        input_params.query_text, input_params.market, input_params.reg_type
    )

    where_clause = ""
    if conditions:
        where_clause = "AND " + " AND ".join(conditions)

    rows = []

    # 시도 1: 원본 쿼리 FTS5
    fts_queries = [input_params.query_text.strip()]
    # 시도 2: 공백 제거 버전 (한국어 복합어 처리)
    no_space = input_params.query_text.replace(" ", "")
    if no_space != fts_queries[0]:
        fts_queries.append(no_space)
    # 시도 3: 개별 키워드 OR 검색
    keywords = input_params.query_text.strip().split()
    if len(keywords) > 1:
        fts_queries.append(" OR ".join(keywords))

    for fts_query in fts_queries:
        try:
            sql = f"""
                SELECT
                    r.id, r.regulation_name, r.market,
                    r.article_title, r.text,
                    regulations_fts.rank AS score
                FROM regulations_fts
                JOIN regulations r ON r.id = regulations_fts.rowid
                WHERE regulations_fts MATCH ?
                {where_clause}
                ORDER BY regulations_fts.rank
                LIMIT ?
            """
            all_params = [fts_query] + filter_params + [input_params.top_k]
            cur.execute(sql, all_params)
            rows = cur.fetchall()
            if rows:
                logger.info("FTS5 검색 성공 (query='%s'): %d건", fts_query[:30], len(rows))
                break
        except sqlite3.OperationalError as e:
            logger.debug("FTS5 MATCH 실패 (%s): %s", fts_query[:30], e)
            continue

    # 폴백: LIKE 검색
    if not rows:
        logger.info("FTS5 결과 없음 - LIKE 폴백")
        rows = _fallback_like_search(cur, input_params, conditions, filter_params)

    hits: list[RegulationSearchHit] = []
    for row in rows:
        hit = RegulationSearchHit(
            id=row["id"],
            score=abs(float(row["score"])) if row["score"] else 0.0,
            text=row["text"],
            regulation_name=row["regulation_name"],
            market=row["market"],
            article_title=row["article_title"],
        )
        hits.append(hit)

    conn.close()

    output = RegulationSearchOutput(hits=hits)
    logger.info(
        "regulation_search 완료: query='%s', hits=%d",
        input_params.query_text[:50],
        len(hits),
    )
    return output.model_dump()


def _fallback_like_search(
    cur: sqlite3.Cursor,
    params: RegulationSearchInput,
    conditions: list[str],
    filter_params: list,
) -> list:
    """FTS5 검색 실패 시 LIKE 기반 폴백 검색."""
    # 검색어를 공백으로 분리하여 각각 LIKE 매칭
    keywords = params.query_text.strip().split()
    like_conditions = []
    like_params = []

    for kw in keywords[:5]:  # 최대 5개 키워드
        like_conditions.append("(r.text LIKE ? OR r.article_title LIKE ?)")
        like_params.extend([f"%{kw}%", f"%{kw}%"])

    where_parts = []
    if like_conditions:
        where_parts.append("(" + " AND ".join(like_conditions) + ")")
    if conditions:
        where_parts.extend(conditions)

    where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""
    all_params = like_params + filter_params + [params.top_k]

    sql = f"""
        SELECT
            r.id,
            r.regulation_name,
            r.market,
            r.article_title,
            r.text,
            0.0 AS score
        FROM regulations r
        {where_clause}
        LIMIT ?
    """

    cur.execute(sql, all_params)
    return cur.fetchall()
