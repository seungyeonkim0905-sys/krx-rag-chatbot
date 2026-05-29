"""SQLite FTS5 + 진짜 벡터 하이브리드 검색 Tool.

개선점:
  1. 동의어 사전(synonyms.py)으로 쿼리 확장
  2. FTS5(어휘) 검색 + 진짜 벡터(의미) 검색을 RRF로 결합
  3. local_model 폴더의 임베딩 모델 사용 (BAAI/bge-m3 또는 klue/roberta 등)
"""

from __future__ import annotations

import logging
import os
import sqlite3
import ssl
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer, models

from agent.config import settings
from agent.models import RegulationSearchHit, RegulationSearchInput, RegulationSearchOutput

# SSL 인증서 오류 방지
ssl._create_default_https_context = ssl._create_unverified_context

# 동의어 사전
try:
    from agent.tools.synonyms import expand_query
except ImportError:
    def expand_query(q: str) -> list[str]:
        return [q]

logger = logging.getLogger(__name__)

# 전역 모델 캐시 (메모리 절약 및 속도 향상)
_model_cache: SentenceTransformer | None = None

def _get_model() -> SentenceTransformer:
    global _model_cache
    if _model_cache is None:
        model_path = settings.embedding_model_path
        logger.info(f"임베딩 모델 로드 중: {model_path}")
        try:
            # 기본 SentenceTransformer 로드 시도
            _model_cache = SentenceTransformer(model_path)
        except Exception as e:
            logger.warning(f"기본 로드 실패({e}), 수동 구성 시도...")
            # Pooling 설정 누락 시 대응
            word_embedding_model = models.Transformer(model_path)
            pooling_model = models.Pooling(word_embedding_model.get_embedding_dimension())
            _model_cache = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        logger.info("모델 로드 완료.")
    return _model_cache

def _ensure_db() -> str:
    db_path = settings.db_path
    # settings.db_path가 krx_regulations.db로 되어 있을 수 있으므로 확인
    # generate_embeddings_real.py가 사용하는 krx_rag.db가 우선
    rag_db = os.path.join(os.path.dirname(db_path), "krx_rag.db")
    if os.path.exists(rag_db):
        return rag_db
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB 파일이 없습니다: {db_path}")
    return db_path

def _fts_escape(q: str) -> str:
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

def _reciprocal_rank_fusion(fts_ranks: dict[int, int], vector_ranks: dict[int, int], k: int = 60) -> dict[int, float]:
    all_ids = set(fts_ranks.keys()) | set(vector_ranks.keys())
    rrf_scores = {}
    for rid in all_ids:
        score = 0.0
        if rid in fts_ranks:
            score += 1.0 / (k + fts_ranks[rid])
        if rid in vector_ranks:
            score += 1.0 / (k + vector_ranks[rid])
        rrf_scores[rid] = score
    return rrf_scores

def regulation_search(args: dict[str, Any]) -> dict[str, Any]:
    params = RegulationSearchInput(**args)
    db_path = _ensure_db()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    top_k = params.top_k
    query_text = params.query_text.strip()
    
    # 1) 동의어 확장
    expanded = expand_query(query_text)
    where, filter_params = _filter_clause(params.market, params.reg_type)

    # 2) FTS5 검색
    fts_results = {}
    for priority, q in enumerate(expanded):
        fts_q = _fts_escape(q)
        if not fts_q: continue
        sql = f"""
            SELECT r.id, r.regulation_name, r.market, r.article_title, r.text
            FROM regulations_fts
            JOIN regulations r ON r.id = regulations_fts.rowid
            WHERE regulations_fts MATCH ? {where}
            ORDER BY regulations_fts.rank
            LIMIT ?
        """
        try:
            cur.execute(sql, [fts_q] + filter_params + [top_k * 3])
            for rank, row in enumerate(cur.fetchall(), start=1):
                rid = row["id"]
                if rid not in fts_results or rank < fts_results[rid][0]:
                    fts_results[rid] = (rank, row)
        except Exception as e:
            logger.debug(f"FTS5 실패: {e}")

    # 3) 벡터 검색 (의미 검색)
    vector_results = {}
    try:
        model = _get_model()
        qv = model.encode(query_text, normalize_embeddings=True)
        
        # 필터링된 모든 조문 벡터 가져오기
        sql = "SELECT id, regulation_name, market, article_title, text, embedding FROM regulations WHERE embedding IS NOT NULL"
        if where:
            # where clause는 'AND ...' 형태이므로 'WHERE'로 시작하게 변경
            sql += " " + where.replace("AND", "AND", 1)
            cur.execute(sql, filter_params)
        else:
            cur.execute(sql)
            
        scored = []
        for row in cur.fetchall():
            dv = np.frombuffer(row["embedding"], dtype=np.float32)
            # 코사인 유사도
            sim = float(np.dot(qv, dv)) # normalize_embeddings=True 이므로 dot이 cosine
            scored.append((sim, row))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        for rank, (sim, row) in enumerate(scored[:top_k * 3], start=1):
            vector_results[row["id"]] = (rank, row)
            
    except Exception as e:
        logger.warning(f"벡터 검색 실패: {e}")

    # 4) RRF 결합
    fts_ranks = {rid: rank for rid, (rank, _) in fts_results.items()}
    vector_ranks = {rid: rank for rid, (rank, _) in vector_results.items()}
    rrf_scores = _reciprocal_rank_fusion(fts_ranks, vector_ranks)
    
    sorted_ids = sorted(rrf_scores.items(), key=lambda x: -x[1])

    hits = []
    for rid, score in sorted_ids[:top_k]:
        if rid in fts_results:
            row = fts_results[rid][1]
        else:
            row = vector_results[rid][1]
            
        hits.append(RegulationSearchHit(
            id=row["id"],
            score=score,
            text=row["text"],
            regulation_name=row["regulation_name"],
            market=row["market"],
            article_title=row["article_title"]
        ))

    # 5) 안전장치: 결과 0건이면 필터 제거 후 재검색 (재귀 방지를 위해 수동 수행)
    if not hits and (params.market or params.reg_type):
        logger.info("필터 적용 결과 없음 - 필터 제거 후 재시도")
        new_args = args.copy()
        new_args["market"] = None
        new_args["reg_type"] = None
        return regulation_search(new_args)

    conn.close()
    return RegulationSearchOutput(hits=hits).model_dump()
