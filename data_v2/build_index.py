"""KRX 규정 + 대한민국 법령 마크다운을 파싱하여 SQLite FTS5 DB로 인덱싱한다.

두 가지 데이터 소스를 모두 처리한다:
  1) output_clean (kr_clean/): KRX 거래소 규정 (frontmatter 영문 키, category_path 리스트)
  2) kr (kr/): 대한민국 법령 (frontmatter 한글 키, 편/장/절/조 계층)

조 헤더는 두 소스 모두 '##### 제N조' (레벨5)로 일관됨.
순수 Python + SQLite(FTS5 trigram)만 사용. 외부 의존성 최소화(PyYAML만).
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frontmatter 파싱 (YAML, 한/영 키 모두 지원)
# ---------------------------------------------------------------------------

def parse_frontmatter(content: str) -> tuple[dict, str]:
    """마크다운 YAML frontmatter를 파싱한다. \r\n 줄바꿈도 처리."""
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    meta: dict = {}
    body = content

    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                loaded = yaml.safe_load(parts[1])
                if isinstance(loaded, dict):
                    meta = loaded
            except yaml.YAMLError as e:
                logger.warning("YAML 파싱 실패, 단순 파서로 대체: %s", e)
                for line in parts[1].strip().split("\n"):
                    if ":" in line and not line.startswith((" ", "-")):
                        k, v = line.split(":", 1)
                        meta[k.strip()] = v.strip().strip("'\"")
            body = parts[2]

    return meta, body


# ---------------------------------------------------------------------------
# 메타데이터 정규화: 두 소스의 서로 다른 필드를 공통 스키마로 매핑
# ---------------------------------------------------------------------------

def normalize_meta(meta: dict, filepath: str, source: str) -> dict:
    """소스별 frontmatter를 공통 스키마로 변환한다.

    공통 스키마:
        title, doc_type, category, market, enforce_date, promulgation_no, department
    """
    if source == "krx":  # output_clean — 영문 키
        title = meta.get("title", "")
        cat_path = meta.get("category_path", []) or []
        category = " > ".join(str(c) for c in cat_path) if isinstance(cat_path, list) else str(cat_path)
        market = _detect_market(title, category)
        return {
            "title": title or _fallback_title(filepath),
            "doc_type": meta.get("book_cd", ""),          # 규정/세칙/지침 등
            "category": category,
            "market": market,
            "enforce_date": str(meta.get("enforce_date", "")),
            "promulgation_no": str(meta.get("promulgation_no", "")),
            "department": str(meta.get("department", "")),
            "source": "KRX규정",
        }
    else:  # law — kr.zip, 한글 키
        title = meta.get("제목", "")
        bun = meta.get("소관부처", "")
        if isinstance(bun, list):
            bun = ",".join(str(b) for b in bun)
        return {
            "title": title or _fallback_title(filepath),
            "doc_type": meta.get("법령구분", ""),          # 법률/시행령/시행규칙 등
            "category": meta.get("법령분야", "") or "대한민국 법령",
            "market": "법령",
            "enforce_date": str(meta.get("시행일자", "")),
            "promulgation_no": str(meta.get("공포번호", "")),
            "department": str(bun),
            "source": "대한민국법령",
        }


def _fallback_title(filepath: str) -> str:
    return os.path.splitext(os.path.basename(filepath))[0]


def _detect_market(title: str, category: str) -> str:
    text = (title or "") + (category or "")
    if "유가증권" in text:
        return "유가증권시장"
    if "코스닥" in text:
        return "코스닥시장"
    if "코넥스" in text:
        return "코넥스시장"
    if "파생상품" in text:
        return "파생상품시장"
    if "채권" in text or "환매조건부" in text:
        return "채권시장"
    if "금시장" in text or "석유" in text or "배출권" in text:
        return "일반상품시장"
    return "공통/기타"


# ---------------------------------------------------------------------------
# 조(article) 단위 청킹 — 편/장/절/조 계층 추적
# ---------------------------------------------------------------------------

# 조 가지번호까지 포함: 제41조의2
RE_PART    = re.compile(r"^#{1,6}\s*(제\d+편\b.*)$")
RE_CHAPTER = re.compile(r"^#{1,6}\s*(제\d+장\b.*)$")
RE_SECTION = re.compile(r"^#{1,6}\s*(제\d+절\b.*)$")
RE_ARTICLE = re.compile(r"^#{1,6}\s*(제\d+조(?:의\d+)?\b.*)$")


def chunk_by_article(body: str, base: dict) -> list[dict]:
    """본문을 조 단위로 청킹. 각 청크에 편/장/절 계층 메타 부착."""
    chunks: list[dict] = []
    cur_part = cur_chapter = cur_section = ""
    cur_title = ""
    buf: list[str] = []

    def clean(s: str) -> str:
        return re.sub(r"^#+\s*", "", s).strip()

    def flush():
        if buf:
            text = "\n".join(buf).strip()
            if text and len(text) > 10:
                # chapter 컬럼에 편+장을 합쳐 보존
                chapter_full = " ".join(x for x in [cur_part, cur_chapter] if x)
                chunks.append({
                    **base,
                    "chapter": chapter_full,
                    "section": cur_section,
                    "article_title": cur_title or cur_chapter or base["title"],
                    "text": text,
                })

    for line in body.split("\n"):
        s = line.strip()
        if RE_PART.match(s):
            cur_part = clean(s); cur_chapter = ""; cur_section = ""
            continue
        if RE_CHAPTER.match(s):
            cur_chapter = clean(s); cur_section = ""
            continue
        if RE_SECTION.match(s):
            cur_section = clean(s)
            continue
        if RE_ARTICLE.match(s):
            flush()
            cur_title = clean(s)
            buf = [cur_title]
            continue
        buf.append(line)
    flush()

    # 조 헤더가 전혀 없는 문서(별표/계약서 등)는 크기 기준 분할
    if not chunks and body.strip():
        text = body.strip()
        size = 2000
        for i in range(0, len(text), size):
            part = text[i:i + size]
            if part.strip():
                chunks.append({
                    **base,
                    "chapter": "",
                    "section": "",
                    "article_title": f"{base['title']} (파트 {i // size + 1})",
                    "text": part,
                })

    return chunks


# ---------------------------------------------------------------------------
# DB 구축
# ---------------------------------------------------------------------------

def init_db(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript("""
        DROP TRIGGER IF EXISTS regulations_ai;
        DROP TABLE IF EXISTS regulations_fts;
        DROP TABLE IF EXISTS regulations;

        CREATE TABLE regulations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,            -- KRX규정 / 대한민국법령
            regulation_name TEXT NOT NULL,   -- 규정/법령명 (title)
            doc_type TEXT DEFAULT '',        -- 규정/세칙/지침/법률/시행령 등
            market TEXT DEFAULT '',          -- 유가/코스닥/코넥스/파생/법령 등
            category TEXT DEFAULT '',        -- 분류 경로
            chapter TEXT DEFAULT '',         -- 편+장
            section TEXT DEFAULT '',         -- 절
            article_title TEXT DEFAULT '',   -- 제N조(제목)
            enforce_date TEXT DEFAULT '',
            promulgation_no TEXT DEFAULT '',
            department TEXT DEFAULT '',
            text TEXT NOT NULL
        );

        -- chapter, section 까지 인덱싱에 포함 (기존엔 누락)
        CREATE VIRTUAL TABLE regulations_fts USING fts5(
            regulation_name, market, chapter, section, article_title, text,
            content='regulations',
            content_rowid='id',
            tokenize='trigram'
        );

        CREATE TRIGGER regulations_ai AFTER INSERT ON regulations BEGIN
            INSERT INTO regulations_fts(
                rowid, regulation_name, market, chapter, section, article_title, text)
            VALUES (new.id, new.regulation_name, new.market,
                    new.chapter, new.section, new.article_title, new.text);
        END;
    """)
    conn.commit()
    return conn


def index_zip(conn, zip_path, inner_root, source, only_dirs=None) -> tuple[int, int]:
    """zip 안의 md 파일을 인덱싱. (파일 수, 청크 수) 반환.

    only_dirs: 지정 시 해당 법령 디렉토리만 인덱싱 (경량화용 화이트리스트)
    """
    import zipfile
    cur = conn.cursor()
    n_files = n_chunks = 0

    def norm(s):
        return s.replace("·", "").replace("ㆍ", "").replace(" ", "")

    only_norm = {norm(d) for d in only_dirs} if only_dirs else None

    with zipfile.ZipFile(zip_path) as z:
        md_names = [n for n in z.namelist()
                    if n.endswith(".md") and os.path.basename(n) != "README.md"]
        for name in md_names:
            # 화이트리스트 필터 (법령 디렉토리명 기준)
            if only_norm is not None:
                parts = name.split("/")
                if len(parts) < 2 or norm(parts[1]) not in only_norm:
                    continue
            with z.open(name) as f:
                content = f.read().decode("utf-8", errors="replace")
            meta, body = parse_frontmatter(content)
            base = normalize_meta(meta, name, source)
            base["regulation_name"] = base.pop("title")
            # base에서 청킹에 불필요한 키 정리 후 컬럼 매핑용으로 보존
            chunks = chunk_by_article(body, {**base, "title": base["regulation_name"]})
            for c in chunks:
                cur.execute("""
                    INSERT INTO regulations
                    (source, regulation_name, doc_type, market, category,
                     chapter, section, article_title,
                     enforce_date, promulgation_no, department, text)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    c["source"], c["regulation_name"], c["doc_type"], c["market"],
                    c["category"], c["chapter"], c["section"], c["article_title"],
                    c["enforce_date"], c["promulgation_no"], c["department"], c["text"],
                ))
            n_files += 1
            n_chunks += len(chunks)
            if n_files % 500 == 0:
                conn.commit()
                logger.info("  ... %d 파일 처리", n_files)

    conn.commit()
    return n_files, n_chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # KRX 규정이 자주 인용하는 핵심 법령만 선별 (경량화)
    CORE_LAWS = [
        "자본시장과금융투자업에관한법률",
        "상법",
        "금융회사의지배구조에관한법률",
        "주식ㆍ사채등의전자등록에관한법률",
        "독점규제및공정거래에관한법률",
        "주식회사등의외부감사에관한법률",
        "채무자회생및파산에관한법률",
        "외국환거래법",
        "은행법",
        "금융지주회사법",
    ]

    DB = "/home/claude/workspace/output/krx_rag.db"
    conn = init_db(DB)

    print("=" * 60)
    print("[1/2] KRX 규정 인덱싱 (전체 80개)")
    f1, c1 = index_zip(conn, "/mnt/user-data/uploads/output_clean_no_date.zip",
                       "kr_clean", "krx")
    print(f"  → {f1} 파일, {c1:,} 청크")

    print(f"[2/2] 핵심 법령 인덱싱 ({len(CORE_LAWS)}개만 선별)")
    f2, c2 = index_zip(conn, "/mnt/user-data/uploads/kr.zip", "kr", "law",
                       only_dirs=CORE_LAWS)
    print(f"  → {f2} 파일, {c2:,} 청크")

    cur = conn.cursor()
    cur.execute("INSERT INTO regulations_fts(regulations_fts) VALUES('optimize')")
    conn.commit()
    cur.execute("SELECT COUNT(*) FROM regulations")
    total = cur.fetchone()[0]
    print("=" * 60)
    print(f"✅ 완료: 총 {total:,} 청크 (파일 {f1+f2:,}개)")
    conn.close()
