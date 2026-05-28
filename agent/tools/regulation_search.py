"""SQLite FTS5 기반 KRX 규정 + 핵심 법령 검색 Tool (동의어 확장 적용).

기존 regulation_search.py를 대체한다. 인터페이스(입출력)는 동일하게 유지.

개선점:
  1. 동의어 사전(synonyms.py)으로 실무용어(VI, 서킷브레이커 등)를 정식용어로 확장
  2. 확장된 다중 쿼리 결과를 통합/재정렬 (원본 질의어 우선 가중)
  3. chapter/section 까지 인덱싱된 새 DB 스키마 활용
순수 SQLite FTS5(trigram). 외부 의존성 없음.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from typing import Any

from agent.config import settings
from agent.models import RegulationSearchHit, RegulationSearchInput, RegulationSearchOutput

# 동의어 사전 (같은 tools 폴더에 synonyms.py 배치)
try:
    from agent.tools.synonyms import expand_query
except ImportError:
    # 사전이 없어도 동작하도록 안전장치
    def expand_query(q: str) -> list[str]:
        return [q]

logger = logging.getLogger(__name__)


def _ensure_db() -> str:
    """DB 파일 존재 확인."""
    db_path = settings.db_path
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"DB 파일이 없습니다: {db_path}\n"
            f"먼저 build_index.py를 실행하거나, 제공된 krx_rag.db를 해당 경로에 두세요."
        )
    return db_path


def _fts_escape(q: str) -> str:
    """FTS5 구문 검색용 이스케이프. 따옴표 제거 후 phrase로 감싼다."""
    q = q.replace('"', "").strip()
    return f'"{q}"' if q else q


def _filter_clause(market: str | None, reg_type: str | None) -> tuple[str, list]:
    conditions, params = [], []
    if market:
        conditions.append("r.market = ?")
        params.append(market)
    if reg_type:
        conditions.append("r.regulation_name LIKE ?")
        params.append(f"%{reg_type}%")
    where = ("AND " + " AND ".join(conditions)) if conditions else ""
    return where, params


def regulation_search(args: dict[str, Any]) -> dict[str, Any]:
    """동의어 확장 + FTS5 통합 검색.

    전략:
      1. 질의어를 동의어 사전으로 확장 (원본 + 정식용어들)
      2. 각 확장 쿼리를 FTS5(trigram)로 검색
      3. 결과를 통합하고, 원본 질의어 매칭에 가중치를 주어 재정렬
      4. 전부 비면 공백제거/LIKE 폴백
    """
    params = RegulationSearchInput(**args)
    db_path = _ensure_db()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    top_k = params.top_k

    # 1) 동의어 확장
    expanded = expand_query(params.query_text.strip())

    def _run_fts(where: str, filter_params: list) -> dict:
        """주어진 필터로 FTS5 다중쿼리 검색을 수행하고 결과를 통합 반환."""
        merged: dict[int, tuple[float, sqlite3.Row]] = {}
        for priority, q in enumerate(expanded):
            fts_q = _fts_escape(q)
            if not fts_q:
                continue
            sql = f"""
                SELECT r.id, r.regulation_name, r.market, r.article_title,
                       r.text, regulations_fts.rank AS score
                FROM regulations_fts
                JOIN regulations r ON r.id = regulations_fts.rowid
                WHERE regulations_fts MATCH ? {where}
                ORDER BY regulations_fts.rank
                LIMIT ?
            """
            try:
                cur.execute(sql, [fts_q] + filter_params + [top_k * 2])
                for row in cur.fetchall():
                    combined = row["score"] + priority * 0.5
                    rid = row["id"]
                    if rid not in merged or combined < merged[rid][0]:
                        merged[rid] = (combined, row)
            except sqlite3.OperationalError as e:
                logger.debug("FTS5 실패 (%s): %s", q[:20], e)
                continue
        return merged

    # 2) 1차 검색: Policy가 지정한 필터(market/reg_type) 적용
    where, filter_params = _filter_clause(params.market, params.reg_type)
    merged = _run_fts(where, filter_params)

    # 3) 안전장치: 필터를 걸었는데 0건이면, 필터를 빼고 재검색
    #    (Policy가 잘못된 reg_type/market 필터를 거는 경우를 자동 보정)
    if not merged and (params.market or params.reg_type):
        logger.info("필터 적용 시 0건 - 필터 제거 후 재검색 (market=%s, reg_type=%s)",
                    params.market, params.reg_type)
        merged = _run_fts("", [])

    # 4) 폴백: 그래도 0건이면 LIKE (필터 없이)
    if not merged:
        logger.info("FTS5 결과 없음 - LIKE 폴백")
        no_filter = RegulationSearchInput(query_text=params.query_text, top_k=top_k)
        for row in _fallback_like(cur, no_filter, "", []):
            merged[row["id"]] = (0.0, row)

    # 4) 재정렬 후 상위 top_k
    ranked = sorted(merged.values(), key=lambda x: x[0])[:top_k]

    hits = [
        RegulationSearchHit(
            id=row["id"],
            score=abs(float(score)) if score else 0.0,
            text=row["text"],
            regulation_name=row["regulation_name"],
            market=row["market"],
            article_title=row["article_title"],
        )
        for score, row in ranked
    ]

    conn.close()
    logger.info("regulation_search 완료: query='%s', hits=%d (확장 %d쿼리)",
                params.query_text[:40], len(hits), len(expanded))
    return RegulationSearchOutput(hits=hits).model_dump()


def _fallback_like(cur, params, where, filter_params) -> list:
    keywords = params.query_text.strip().split()
    like_conds, like_params = [], []
    for kw in keywords[:5]:
        like_conds.append("(r.text LIKE ? OR r.article_title LIKE ?)")
        like_params += [f"%{kw}%", f"%{kw}%"]
    where_parts = []
    if like_conds:
        where_parts.append("(" + " AND ".join(like_conds) + ")")
    # where 는 'AND ...' 형태이므로 접두어 제거 후 결합
    extra = where.replace("AND ", "", 1) if where.startswith("AND ") else where
    if extra:
        where_parts.append(extra)
    where_clause = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""
    sql = f"""
        SELECT r.id, r.regulation_name, r.market, r.article_title, r.text, 0.0 AS score
        FROM regulations r {where_clause} LIMIT ?
    """
    cur.execute(sql, like_params + filter_params + [params.top_k])
    return cur.fetchall()
