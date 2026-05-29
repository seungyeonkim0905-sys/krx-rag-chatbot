"""
조문을 '진짜 의미 임베딩(벡터)'으로 변환해 DB에 저장하는 스크립트.

★ 중요: TF-IDF가 아니라 sentence-transformers의 진짜 의미 임베딩을 사용합니다.
        (TF-IDF는 글자 매칭이라 우리 문제를 못 풉니다.)

사용법 (PC PowerShell에서):
    cd "D:\최종코딩파일\krx-rag-chatbot v2"
    python generate_embeddings_real.py

처음 실행하면 모델(약 1.1GB, bge-m3)을 자동 다운로드합니다. (1회성)
그 후 조문 1만여 개를 벡터로 변환해 data/krx_rag.db 의 embedding 컬럼에 저장합니다.
"""

import os
import sqlite3
import time
import numpy as np
import ssl

# SSL 인증서 오류 방지 (필요 시)
ssl._create_default_https_context = ssl._create_unverified_context

# ── 설정 ─────────────────────────────────────────────
# 절대 경로로 변경하여 어디서든 실행 가능하게 함
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "krx_rag.db")

# 로컬 모델 폴더 사용 (인터넷 연결 없이 실행 가능)
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "local_model")
if os.path.exists(LOCAL_MODEL_PATH):
    MODEL_NAME = LOCAL_MODEL_PATH
    USE_LOCAL = True
    print(f"✓ 로컬 모델 경로 확인: {MODEL_NAME}")
else:
    MODEL_NAME = "BAAI/bge-m3"
    USE_LOCAL = False
    print(f"! 로컬 모델을 찾을 수 없어 온라인 모델({MODEL_NAME}) 사용 시도")

BATCH_SIZE = 32          # 한 번에 처리할 조문 수
# ─────────────────────────────────────────────────────


def main():
    from sentence_transformers import SentenceTransformer, models

    if not os.path.exists(DB_PATH):
        print(f"❌ DB를 찾을 수 없습니다: {DB_PATH}")
        print("   이 스크립트를 chatbot 폴더 안에서 실행했는지 확인하세요.")
        return

    print(f"[1/4] 임베딩 모델 로드: {MODEL_NAME}")
    t0 = time.time()
    
    try:
        # 일반적인 방식으로 로드 시도
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        if USE_LOCAL:
            print(f"      ! 기본 로드 실패 ({e}), 수동 구성 시도...")
            # Pooling 설정이 없는 경우 수동으로 추가 (klue/roberta 등 일반 모델 대응)
            word_embedding_model = models.Transformer(MODEL_NAME)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        else:
            raise e
            
    dim = model.get_sentence_embedding_dimension()
    print(f"      ✓ 로드 완료 ({time.time()-t0:.0f}초), 벡터 차원={dim}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # embedding 컬럼이 없으면 추가
    cols = [r[1] for r in cur.execute("PRAGMA table_info(regulations)")]
    if "embedding" not in cols:
        print("[2/4] embedding 컬럼 추가")
        cur.execute("ALTER TABLE regulations ADD COLUMN embedding BLOB")
        conn.commit()
    else:
        print("[2/4] embedding 컬럼 이미 존재")

    # 아직 임베딩 안 된 조문만 (중단 후 재실행해도 이어서 진행)
    rows = cur.execute(
        "SELECT id, article_title, text FROM regulations WHERE embedding IS NULL"
    ).fetchall()
    total = len(rows)
    if total == 0:
        print("✓ 모든 조문이 이미 임베딩되어 있습니다. 끝.")
        conn.close()
        return

    print(f"[3/4] 임베딩 생성 시작: {total:,}개 조문")
    t0 = time.time()
    done = 0
    for i in range(0, total, BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        # 제목+본문을 합쳐 임베딩 (검색 품질 향상)
        texts = [f"{r[1]}\n{r[2]}" for r in batch]
        vecs = model.encode(texts, normalize_embeddings=True)
        for (rid, _, _), vec in zip(batch, vecs):
            blob = np.asarray(vec, dtype=np.float32).tobytes()
            cur.execute("UPDATE regulations SET embedding=? WHERE id=?", (blob, rid))
        conn.commit()
        done += len(batch)
        elapsed = time.time() - t0
        speed = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / speed if speed > 0 else 0
        print(f"      {done:,}/{total:,}  ({done*100//total}%)  "
              f"경과 {elapsed:.0f}s  남은예상 {eta:.0f}s", flush=True)

    print(f"[4/4] 완료! 총 {done:,}개, {time.time()-t0:.0f}초 소요")
    conn.close()
    print("\n✅ 다음 단계: python test_embedding_quality.py 로 효과를 확인하세요.")


if __name__ == "__main__":
    main()
