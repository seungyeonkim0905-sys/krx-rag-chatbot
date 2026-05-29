"""
진짜 임베딩이 효과 있는지 확인하는 테스트.

generate_embeddings_real.py 를 먼저 실행한 뒤 이걸 돌리세요:
    python test_embedding_quality.py

확인 포인트:
  - "상장폐지 사유" → 본문이 긴 핵심 조문(제48조 등)이 상위에 오는가?
  - "변동성완화장치" → 매매계약체결방법 관련 조문이 의미로 잡히는가?
기존 FTS5(글자검색)와 결과를 나란히 비교해서 보여줍니다.
"""

import os
import sqlite3
import numpy as np
import ssl

# SSL 인증서 오류 방지
ssl._create_default_https_context = ssl._create_unverified_context

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "krx_rag.db")
MODEL_NAME = os.path.join(BASE_DIR, "local_model")  # 로컬 모델 경로
# ─────────────────────────────────────────────────────

TEST_QUERIES = [
    ("상장폐지 사유", "유가증권시장"),
    ("변동성완화장치", None),
    ("내부자거래 처벌", None),
]


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    from sentence_transformers import SentenceTransformer, models
    print(f"모델 로드: {MODEL_NAME} ...")
    
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception:
        # Pooling 설정 누락 시 대응
        word_embedding_model = models.Transformer(MODEL_NAME)
        pooling_model = models.Pooling(word_embedding_model.get_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    for query, market in TEST_QUERIES:
        print("\n" + "=" * 70)
        print(f"질문: '{query}'" + (f"  (시장={market})" if market else ""))
        print("=" * 70)

        # ── (A) 기존 FTS5 글자검색 ──
        print("\n[기존 FTS5 - 글자검색]")
        where = "AND r.market=?" if market else ""
        prm = [f'"{query}"'] + ([market] if market else []) + [5]
        try:
            fts = cur.execute(f"""
                SELECT r.article_title, r.regulation_name, LENGTH(r.text) len
                FROM regulations_fts JOIN regulations r ON r.id=regulations_fts.rowid
                WHERE regulations_fts MATCH ? {where}
                ORDER BY regulations_fts.rank LIMIT ?
            """, prm).fetchall()
            for r in fts:
                mark = "⭐" if r["len"] > 2500 else "  "
                print(f"  {mark}({r['len']:5d}자) {r['regulation_name'][:14]} {r['article_title'][:30]}")
        except Exception as e:
            print(f"  (오류: {e})")

        # ── (B) 진짜 의미 임베딩 검색 ──
        print("\n[진짜 임베딩 - 의미검색]")
        qv = model.encode(query, normalize_embeddings=True)
        sql = "SELECT id, article_title, regulation_name, market, LENGTH(text) len, embedding FROM regulations WHERE embedding IS NOT NULL"
        if market:
            sql += " AND market=?"
            cur.execute(sql, (market,))
        else:
            cur.execute(sql)
        scored = []
        for row in cur.fetchall():
            dv = np.frombuffer(row["embedding"], dtype=np.float32)
            scored.append((cosine(qv, dv), row))
        scored.sort(reverse=True, key=lambda x: x[0])
        for s, r in scored[:5]:
            mark = "⭐" if r["len"] > 2500 else "  "
            print(f"  {mark}유사도{s:.3f} ({r['len']:5d}자) {r['regulation_name'][:14]} {r['article_title'][:30]}")

    conn.close()
    print("\n\n⭐ 표시 = 본문 2500자 이상의 핵심 조문")
    print("의미검색에서 ⭐가 상위에 오면 성공입니다.")


if __name__ == "__main__":
    main()
