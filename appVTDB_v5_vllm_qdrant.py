"""
QCD Tool v5 - Bản tối ưu: vLLM + Qdrant
- AI: vLLM host công ty (ưu tiên) | Gemini API (fallback)
- Vector DB: Qdrant local (nhanh hơn ChromaDB)
- Jira: token mới (đã test OK)

Cách chạy:
    streamlit run appVTDB_v5_vllm_qdrant.py
"""

import streamlit as st
import pandas as pd
import json
import os
import requests
import re
import io
import time
import hashlib
import threading
from datetime import datetime
from dotenv import load_dotenv
import urllib3
import logging

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# ==== LOAD .ENV ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH, override=True)

# ==== CẤU HÌNH ====
st.set_page_config(page_title="QCD Tool v5 – vLLM + Qdrant", page_icon="🔍", layout="wide")

# Jira
JIRA_URL = os.getenv("JIRA_URL", "").rstrip("/")
JIRA_TOKEN = os.getenv("JIRA_API_TOKEN", "").strip()

# AI Engine
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "").strip()
VLLM_MODEL = os.getenv("VLLM_MODEL", "qwen3").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Qdrant
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_db")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "jira_issues_v5")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

# Thresholds
SIMILARITY_THRESHOLD = 0.55  # Cosine similarity (higher = more similar)
DESC_RAW_MAX_CHARS = 6000
AI_TIMEOUT_SECONDS = 60

# ==== KHỞI TẠO AI ENGINE ====
USE_VLLM = bool(VLLM_BASE_URL and VLLM_MODEL)

if USE_VLLM:
    from openai import OpenAI
    ai_client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")
    st.success(f"✅ **vLLM**: {VLLM_MODEL} @ {VLLM_BASE_URL}")
elif GEMINI_API_KEY:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    st.info(f"ℹ️ **Gemini**: fallback (chưa có vLLM từ IT)")
else:
    st.error("❌ Cần cấu hình VLLM_BASE_URL hoặc GEMINI_API_KEY trong .env")
    st.stop()

def call_llm(prompt: str) -> str:
    """Gọi AI: vLLM nếu có, fallback Gemini."""
    if USE_VLLM:
        resp = ai_client.chat.completions.create(
            model=VLLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=4096
        )
        return resp.choices[0].message.content.strip()
    else:
        model = genai.GenerativeModel("gemini-2.5-flash", generation_config={"temperature": 0, "max_output_tokens": 8192})
        return model.generate_content(prompt).text.strip()

def call_llm_with_timeout(prompt: str, timeout: int = AI_TIMEOUT_SECONDS) -> str:
    result = {"text": None, "error": None}
    def _call():
        try:
            result["text"] = call_llm(prompt)
        except Exception as e:
            result["error"] = e
    t = threading.Thread(target=_call, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise TimeoutError(f"AI không phản hồi sau {timeout}s.")
    if result["error"]:
        raise result["error"]
    return result["text"]

def embed_text(text: str) -> list[float]:
    """Tạo vector embedding."""
    cleaned = clean_jira_text(text)
    if not cleaned:
        cleaned = "empty"
    if USE_VLLM:
        # vLLM có thể dùng embedding model nếu có
        try:
            resp = ai_client.embeddings.create(model=VLLM_MODEL, input=cleaned)
            return resp.data[0].embedding
        except Exception:
            pass
    # Fallback Gemini embedding
    if GEMINI_API_KEY:
        import google.generativeai as genai
        resp = genai.embed_content(model="models/text-embedding-004", content=cleaned)
        return resp["embedding"]
    return [0.0] * EMBED_DIM

# ==== JIRA SESSION ====
@st.cache_resource
def create_jira_session():
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {JIRA_TOKEN}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    })
    return session

jira_session = create_jira_session()

# ==== QDRANT ====
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(path=QDRANT_PATH)

def ensure_collection():
    qc = get_qdrant_client()
    existing = [c.name for c in qc.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        qc.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
        )
    return qc

def key_to_id(key: str) -> int:
    return int(hashlib.md5(key.encode()).hexdigest()[:15], 16)

# ==== TEXT PROCESSING ====
def clean_jira_text(text: str) -> str:
    if not isinstance(text, str): text = str(text or "")
    text = re.sub(r'!\S+\.(png|jpg|jpeg|gif|bmp|svg)(\|[^!]*)?!', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\{[a-zA-Z][^}]{0,80}\}', '', text)
    text = re.sub(r'\*{1,3}', ' ', text)
    text = re.sub(r'\b_([^_]+)_\b', r'\1', text)
    text = re.sub(r'-{4,}', ' ', text)
    text = re.sub(r'(?m)^[#!@]+\s*', '', text)
    text = text.replace('\xa0', ' ').replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
    return re.sub(r'\s+', ' ', text).strip()[:7500]

def extract_zones(raw_desc: str) -> dict:
    text = raw_desc or ""
    OBS_MARKERS = ["observation:", "actual observation:", "actual result:", "actual results:",
                    "iv. actual", "4. actual result", "2.4. actual", "actual:", "4. actual", "[actual]"]
    EXP_MARKERS = ["expected result:", "expected results:", "expect result:", "expectation:",
                    "iii. expect", "3. expect", "2.3. expected", "test expected", "expected:"]
    PROC_MARKERS = ["test procedure:", "test procedure :", "step/test procedure:", "steps to reproduce",
                    "procedure:", "test step", "2.2. test step", "how to reproduce:", "test steps:",
                    "ii. test condition", "step:", "steps:", "[step]", "[test procedure]"]
    END_MARKERS = ["recovery:", "recover:", "frequency:", "occurrences:", "attachment:",
                    "remark:", "reference:", "2.5", "2.6", "v. frequency", "vi. recover"]
    lower = text.lower()

    def find_last(markers):
        best_pos, best_len = -1, 0
        for m in markers:
            idx = lower.rfind(m)
            if idx > best_pos: best_pos, best_len = idx, len(m)
        return best_pos, best_len

    def find_first(markers, after=0):
        best_pos, best_len = len(text)+1, 0
        for m in markers:
            idx = lower.find(m, after)
            if idx != -1 and idx < best_pos: best_pos, best_len = idx, len(m)
        return (best_pos if best_pos <= len(text) else -1), best_len

    def extract_section(sp, sl, stop):
        if sp == -1: return ""
        cs = sp + sl
        ep = len(text)
        for m in stop:
            idx = lower.find(m, cs)
            if idx != -1 and idx < ep: ep = idx
        return text[cs:ep].strip()

    obs_p, obs_l = find_last(OBS_MARKERS)
    if obs_p != -1:
        observation = extract_section(obs_p, obs_l, EXP_MARKERS + END_MARKERS)
    else:
        lines = text.split('\n')
        el = [l for l in lines if any(k in l.lower() for k in ['fail','error','wrong','not','issue','bug','bị'])]
        observation = " ".join(el) if el else text[:1500]

    exp_p, exp_l = find_first(EXP_MARKERS)
    proc_p, proc_l = find_first(PROC_MARKERS)
    expected = extract_section(exp_p, exp_l, OBS_MARKERS + PROC_MARKERS + END_MARKERS)
    procedure = extract_section(proc_p, proc_l, EXP_MARKERS + OBS_MARKERS + END_MARKERS)
    return {"observation": observation, "expected": expected, "procedure": procedure}

def extract_core_summary(summary: str) -> str:
    if not isinstance(summary, str): return ""
    s = summary
    for _ in range(8):
        prev = s
        s = re.sub(r'^\s*\[[^\]]*\]\s*', '', s).strip()
        s = re.sub(r'^\s*VF6\w*[-_]?\s*[|\-]?\s*', '', s, flags=re.IGNORECASE).strip()
        s = re.sub(r'^\s*VF\d\w*[-_]?\s*[|\-]?\s*', '', s, flags=re.IGNORECASE).strip()
        s = re.sub(r'^\s*EE[_\s]\w+[-_]?\s*[|\-]?\s*', '', s, flags=re.IGNORECASE).strip()
        s = re.sub(r'^\s*EEC-\w+[-_]?\s*[|\-]?\s*', '', s, flags=re.IGNORECASE).strip()
        s = re.sub(r'^\s*(ePT|FPT|ODX|CTF|AUTO)[-_]?\w*\s*[|\-]?\s*', '', s, flags=re.IGNORECASE).strip()
        s = re.sub(r'^\s*FRS\s*\S+\s*[|\-]?\s*', '', s, flags=re.IGNORECASE).strip()
        if s == prev: break
    if s.count('|') >= 2:
        parts = [p.strip() for p in s.split('|')]
        s = parts[-1] if parts[-1] else parts[-2]
    return s or summary

def build_vector_document(summary: str, zones: dict) -> str:
    parts = []
    clean_sum = extract_core_summary(summary)
    obs = zones.get("observation","").strip()
    exp = zones.get("expected","").strip()
    if obs: parts.append(f"Fault: {obs[:500]}"); parts.append(f"Actual: {obs[:300]}")
    if exp: parts.append(f"Expected: {exp[:250]}")
    if clean_sum: parts.append(f"Summary: {clean_sum}")
    return " | ".join(parts)

# ==== JIRA FETCH ====
def fetch_jira_issues(jql: str) -> pd.DataFrame:
    all_issues = []
    start_at = 0
    while True:
        params = {
            "jql": jql, "startAt": start_at, "maxResults": 100,
            "fields": "key,summary,description,status,issuetype,customfield_14506,customfield_12101,customfield_market"
        }
        resp = jira_session.get(f"{JIRA_URL}/rest/api/2/search", params=params, verify=False, timeout=30)
        if resp.status_code != 200:
            st.error(f"Jira lỗi {resp.status_code}: {resp.text[:200]}")
            break
        data = resp.json()
        issues = data.get("issues", [])
        all_issues.extend(issues)
        if start_at + len(issues) >= data.get("total", 0) or not issues:
            break
        start_at += len(issues)

    rows = []
    for issue in all_issues:
        f = issue.get("fields", {})
        market_raw = (f.get("customfield_14506") or f.get("customfield_12101") or
                      f.get("customfield_markets") or f.get("customfield_market") or "")
        if isinstance(market_raw, dict): market_raw = market_raw.get("value","")
        elif isinstance(market_raw, list):
            market_raw = ", ".join([m.get("value", m) if isinstance(m,dict) else str(m) for m in market_raw])
        market_raw = str(market_raw or "")
        if not market_raw:
            search_text = f"{f.get('summary','')} {f.get('description','') or ''}"
            m_match = re.search(r'\[\s*(KZ|VN|EU|ME|UAE|AU|US|CAN|NA)\s*\]|\b(KZ|VN|EU|ME|UAE|AU|US|CAN|NA)\b', search_text, re.IGNORECASE)
            if m_match: market_raw = (m_match.group(1) or m_match.group(2)).upper()
        rows.append({
            "Key": issue.get("key",""), "Summary": f.get("summary",""),
            "Description": f.get("description","") or "", "Status": f.get("status",{}).get("name",""),
            "Markets": market_raw,
        })
    return pd.DataFrame(rows)

# ==== SYNC TO QDRANT ====
def sync_to_qdrant(df: pd.DataFrame, progress_ph=None) -> dict:
    qc = ensure_collection()
    total = len(df)
    skipped, points = 0, []
    for i, row in df.iterrows():
        key = str(row.get("Key", f"row-{i}")).strip()
        if not key: skipped += 1; continue
        summary = str(row.get("Summary",""))
        desc = str(row.get("Description",""))
        status = str(row.get("Status",""))
        market = str(row.get("Markets",""))
        zones = extract_zones(desc)
        doc = build_vector_document(summary, zones)
        if not doc.strip(): skipped += 1; continue
        points.append(PointStruct(
            id=key_to_id(key),
            vector=embed_text(doc),
            payload={
                "key": key, "summary_raw": summary,
                "observation": zones["observation"], "expected": zones["expected"],
                "procedure": zones["procedure"],
                "description_raw": desc[:DESC_RAW_MAX_CHARS],
                "status": status, "markets": market,
                "synced_at": datetime.now().isoformat()
            }
        ))
        if progress_ph and i % 20 == 0:
            progress_ph.progress(min((i+1)/total, 1.0))
    if points:
        batch_size = 50
        for b in range(0, len(points), batch_size):
            qc.upsert(QDRANT_COLLECTION, points[b:b+batch_size])
    return {"total": total, "upserted": len(points), "skipped": skipped}

def get_db_stats() -> dict:
    try:
        qc = get_qdrant_client()
        info = qc.get_collection(QDRANT_COLLECTION)
        sample = qc.scroll(QDRANT_COLLECTION, limit=3, with_payload=True)[0]
        dates = [p.payload.get("synced_at","") for p in sample if p.payload.get("synced_at")]
        keys = [p.payload.get("key","") for p in sample]
        return {"count": info.points_count, "last_synced": max(dates)[:19].replace("T"," ") if dates else "—", "sample_keys": keys}
    except Exception:
        return {"count": 0, "last_synced": "—", "sample_keys": []}

# ==== RERANK PROMPT ====
def slim_for_ai(item: dict, max_field: int = 600) -> dict:
    out = {}
    for k, v in item.items():
        if k == "description_full": continue
        out[k] = (str(v or "")[:max_field]).strip()
    return out

def build_rerank_prompt(new_item: dict, candidates: list) -> str:
    new_slim = slim_for_ai(new_item)
    cand_slim = [slim_for_ai(c) for c in candidates]
    return f"""You are a Senior Automotive QA expert for VinFast electric vehicles.
Compare the NEW TICKET against TOP CANDIDATES from the Jira database.

Each ticket has 5 pre-extracted fields:
  * summary   — cleaned fault description (tags stripped)
  * actual    — what the system actually did wrong (Observation / Actual Result)
  * expected  — what the system SHOULD do (defines the ticket's purpose/scope)
  * procedure — test steps used to reproduce
  * markets   — target market (KZ/VN/EU/ME/UAE); may be empty

PHASE 1 - SCOPE CHECK (Expected Result):
  SAME SCOPE: both tickets expect the same system behavior/outcome
  DIFFERENT SCOPE: tickets have fundamentally different purposes
  SCOPE RULE: If DIFFERENT SCOPE, cap final score at 55 regardless of Actual similarity.

PHASE 2 - SIMILARITY SCORING (Actual Result - primary driver):
Keyword scoring per match:
  +3 pts — SPECIFIC: exact DTC code (P0xxx, U0xxx), exact signal value, exact fault state
  +2 pts — component + symptom PAIR: "MHU freeze", "ACC not activate", "DMS false alarm"
  +1 pt  — single generic term: VCU, BMS, DTC, MHU, warning, display

CLASSIFICATION:
  85-100 → DUPLICATE | 70-84 → NEAR_DUP | 50-69 → SIMILAR | <50 → NOT_RELATED

Return ONLY a valid JSON array. No markdown, no code block.
[
  {{
    "jira_key": "<key>",
    "is_duplicate": true/false,
    "confidence_score": <0-100>,
    "classification": "DUPLICATE" | "NEAR_DUP" | "SIMILAR" | "NOT_RELATED",
    "scope_match": true/false,
    "reason": "<1-2 câu giải thích bằng tiếng Việt>"
  }}
]

NEW TICKET: {json.dumps(new_slim, ensure_ascii=False)}
TOP CANDIDATES FROM JIRA: {json.dumps(cand_slim, ensure_ascii=False)}"""

def parse_ai_json(text: str) -> list:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text).strip()
    try: result = json.loads(text); return result if isinstance(result, list) else [result]
    except: pass
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try: result = json.loads(match.group()); return result if isinstance(result, list) else [result]
        except: pass
    return []

# ==== UI HELPERS ====
def metric_card(label, value, color):
    return f"""<div class="metric-card" style="border-left-color:{color}">
        <div class="metric-label">{label}</div>
        <div class="metric-num" style="color:{color}">{value}</div>
    </div>"""

def classification_tag(cls):
    mapping = {
        "DUPLICATE": '<span class="tag-dup">DUPLICATE</span>',
        "NEAR_DUP": '<span class="tag-near">NEAR DUP</span>',
        "SIMILAR": '<span class="tag-sim">SIMILAR</span>',
        "NOT_RELATED": '<span class="tag-new">NEW</span>',
        "": '<span class="tag-new">NEW</span>',
    }
    return mapping.get(cls, cls)

# ==== CSS ====
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background-color: #0f1117; color: #e2e8f0; }
    .stTabs [data-baseweb="tab-list"] { background: #1a1f2e; border-radius: 8px; padding: 4px; gap: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 6px; color: #94a3b8; font-weight: 500; font-size: 13px; }
    .stTabs [aria-selected="true"] { background: #2563eb !important; color: white !important; }
    .metric-card { background: #1a1f2e; padding: 18px 20px; border-radius: 10px; border: 1px solid #2d3748; border-left: 3px solid; }
    .metric-num { font-size: 28px; font-weight: 600; font-family: 'IBM Plex Mono', monospace; margin-top: 4px; }
    .metric-label { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }
    .db-info-box { background: #1a1f2e; border: 1px solid #2d3748; border-radius: 10px; padding: 16px 20px; font-family: 'IBM Plex Mono', monospace; font-size: 13px; }
    .db-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #2d374840; color: #cbd5e1; }
    .db-row:last-child { border-bottom: none; }
    .db-key { color: #64748b; }
    .tag-dup { background: #7f1d1d; color: #fca5a5; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
    .tag-near { background: #78350f; color: #fcd34d; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
    .tag-sim { background: #1e3a5f; color: #93c5fd; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
    .tag-new { background: #14532d; color: #86efac; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
    div[data-testid="stButton"] > button { background: #2563eb; color: white; border: none; border-radius: 8px; font-weight: 600; font-size: 13px; }
    div[data-testid="stButton"] > button:hover { background: #1d4ed8; }
    .stDataFrame { background: #1a1f2e; border-radius: 10px; }
    .stProgress > div > div { background: #2563eb; }
    .reason-box { background: #111827; border: 1px solid #374151; border-radius: 8px; padding: 12px 14px; font-size: 12px; font-family: 'IBM Plex Mono', monospace; white-space: pre-wrap; color: #d1d5db; line-height: 1.7; }
    h1 { color: #f8fafc !important; font-weight: 600 !important; }
    h2, h3 { color: #e2e8f0 !important; font-weight: 500 !important; }
    label { color: #94a3b8 !important; font-size: 13px !important; }
    .stTextArea textarea, .stTextInput input { background: #1a1f2e !important; color: #e2e8f0 !important; border: 1px solid #374151 !important; border-radius: 8px !important; }
    .stAlert { border-radius: 8px; }
    .stExpander { background: #1a1f2e; border: 1px solid #2d3748; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ==== SIDEBAR ====
with st.sidebar:
    st.markdown("### ⚙️ Cấu hình")
    st.markdown(f"**AI Engine:** `{'vLLM' if USE_VLLM else 'Gemini'}`")
    if USE_VLLM:
        st.markdown(f"**vLLM:** `{VLLM_MODEL}`")
    else:
        st.markdown(f"**Gemini Key:** `...{GEMINI_API_KEY[-4:]}`" if GEMINI_API_KEY else "**Gemini:** ❌")
    st.divider()
    new_file = st.file_uploader("📂 Upload New List (Excel/CSV)", type=["xlsx", "csv"])
    jql_input = st.text_area("JQL Query (Master List)", value='project = "VF6" ORDER BY created DESC', height=80)
    st.divider()
    threshold = st.slider("Ngưỡng tương đồng (%)", 40, 90, 60, 5)
    top_k = st.slider("Top-K candidates", 3, 10, 5)
    run_btn = st.button("🚀 Chạy Phân Tích", type="primary", use_container_width=True)

# ==== MAIN ====
st.markdown("## 🔍 QCD v5 — vLLM + Qdrant")
tab_run, tab_db = st.tabs(["📊 Phân Tích", "🗄️ Quản lý Database"])

# DB TAB
with tab_db:
    st.markdown("### Trạng thái Vector Database")
    stats = get_db_stats()
    c1, c2, c3 = st.columns(3)
    c1.markdown(metric_card("Tickets trong DB", stats["count"], "#3b82f6"), unsafe_allow_html=True)
    c2.markdown(metric_card("Lần sync cuối", stats["last_synced"] or "—", "#8b5cf6"), unsafe_allow_html=True)
    c3.markdown(metric_card("Sample keys", len(stats["sample_keys"]), "#06b6d4"), unsafe_allow_html=True)

    st.markdown("#### 📥 Tải dữ liệu Jira")
    col_fetch, col_download = st.columns([3, 1])
    with col_fetch:
        if st.button("⬇️ Tải dữ liệu Jira", use_container_width=True):
            with st.spinner("Đang tải từ Jira..."):
                master_raw = fetch_jira_issues(jql_input)
            if master_raw.empty:
                st.error("Không lấy được dữ liệu")
            else:
                st.session_state["fetched_jira_data"] = master_raw
                st.success(f"✅ Đã tải {len(master_raw)} tickets.")
                st.rerun()

    if "fetched_jira_data" in st.session_state and not st.session_state["fetched_jira_data"].empty:
        st.dataframe(st.session_state["fetched_jira_data"], use_container_width=True)
        with col_download:
            buf = io.BytesIO()
            st.session_state["fetched_jira_data"].to_excel(buf, index=False)
            st.download_button("📥 Excel", buf.getvalue(), f"jira_{datetime.now():%Y%m%d_%H%M}.xlsx", use_container_width=True)

    st.markdown("#### Đồng bộ → Qdrant")
    col_sync, col_clear = st.columns([3, 1])
    with col_sync:
        if st.button("🔄 Đồng bộ (upsert)", use_container_width=True):
            data = st.session_state.get("fetched_jira_data")
            if data is None or data.empty:
                with st.spinner("Đang tải Jira..."):
                    data = fetch_jira_issues(jql_input)
            if not data.empty:
                prog = st.progress(0)
                st.info(f"Embedding {len(data)} tickets...")
                result = sync_to_qdrant(data, prog)
                prog.progress(1.0)
                st.success(f"✅ Upserted {result['upserted']} tickets (skip {result['skipped']}). DB: {get_db_stats()['count']} tickets.")
                st.rerun()
    with col_clear:
        if st.button("🗑️ Xoá DB", use_container_width=True):
            try:
                get_qdrant_client().delete_collection(QDRANT_COLLECTION)
                st.success("Đã xoá. Refresh trang.")
                st.rerun()
            except Exception as e:
                st.error(f"Lỗi: {e}")

    if stats["count"] > 0:
        with st.expander("👁️ Xem mẫu (10 records đầu)"):
            sample = get_qdrant_client().scroll(QDRANT_COLLECTION, limit=10, with_payload=True)[0]
            rows = [{"Key": p.payload.get("key",""), "Summary": (p.payload.get("summary_raw","") or "")[:80],
                     "Markets": p.payload.get("markets",""), "Synced At": (p.payload.get("synced_at","") or "")[:19]} for p in sample]
            st.dataframe(pd.DataFrame(rows))

# RUN TAB
with tab_run:
    if not run_btn or not new_file:
        st.info("👈 Upload file Excel và nhấn **Chạy Phân Tích**.")
        _db_count = get_db_stats()["count"]
        if _db_count == 0:
            st.warning("⚠️ Vector DB đang trống. Vào tab Quản lý Database để sync Jira trước.")
        else:
            st.success(f"✅ DB sẵn sàng với {_db_count} tickets.")
    else:
        with st.spinner("Đang đọc file..."):
            new_df = pd.read_excel(new_file) if new_file.name.endswith(".xlsx") else pd.read_csv(new_file)

        _db_count = get_db_stats()["count"]
        if _db_count == 0:
            st.error("⚠️ Vector DB trống. Vào tab Quản lý Database trước.")
        else:
            st.markdown(f"**File:** `{new_file.name}` — **{len(new_df)} tickets**")
            sum_col = next((c for c in new_df.columns if str(c).strip().upper() in ["SUMMARY","NEW_SUMMARY"]), "Summary")
            desc_col = next((c for c in new_df.columns if str(c).strip().upper() in ["DESCRIPTION","NEW_DESCRIPTION"]), "Description")
            market_col = next((c for c in new_df.columns if str(c).strip().upper() in ["MARKET","MARKETS","NEW_MARKET"]), None)

            new_items = []
            for i, row in new_df.iterrows():
                summary = str(row.get(sum_col,""))
                desc = str(row.get(desc_col,""))
                m_val = row.get(market_col) if market_col else None
                market = ""
                if m_val and not (isinstance(m_val, float) and pd.isna(m_val)) and str(m_val).strip().lower() not in ["nan","none",""]:
                    market = str(m_val).strip()
                if not market:
                    for _src in [summary, desc]:
                        mm = re.search(r'\[\s*(KZ|VN|EU|ME|UAE|AU|US|CAN|NA)\s*\]|\b(KZ|VN|EU|ME|UAE|AU|US|CAN|NA)\b', _src, re.IGNORECASE)
                        if mm: market = (mm.group(1) or mm.group(2)).upper(); break
                zones = extract_zones(desc)
                new_items.append({"idx": i, "summary": extract_core_summary(summary), "summary_raw": summary,
                                  "description_raw": desc, "observation": zones["observation"],
                                  "expected": zones["expected"], "procedure": zones["procedure"], "markets": market})

            matches, prog_bar, status_ph = [], st.progress(0), st.empty()
            total = len(new_items)
            qc = get_qdrant_client()

            for idx, item in enumerate(new_items):
                status_ph.text(f"⏳ [{idx+1}/{total}] {item['summary_raw'][:60]}...")
                query_text = build_vector_document(item["summary_raw"], {"observation": item["observation"], "expected": item["expected"], "procedure": item["procedure"]})
                item_market = item.get("markets","").strip().upper()

                try:
                    query_vec = embed_text(query_text)
                    search_filter = None
                    if item_market:
                        search_filter = Filter(must=[FieldCondition(key="markets", match=MatchValue(value=item_market))])
                    fetch_k = min(top_k * 2 + 2, _db_count)
                    if fetch_k == 0: continue
                    results = qc.search(QDRANT_COLLECTION, query_vector=query_vec, query_filter=search_filter, limit=fetch_k, with_payload=True)
                    if len(results) < 2 and item_market:
                        results = qc.search(QDRANT_COLLECTION, query_vector=query_vec, limit=fetch_k, with_payload=True)
                except Exception as e:
                    st.warning(f"⚠️ Search lỗi item {idx+1}: {e}"); prog_bar.progress((idx+1)/total); continue

                candidates = []
                for point in results:
                    if point.score >= SIMILARITY_THRESHOLD:
                        pl = point.payload
                        candidates.append({"jira_key": pl.get("key",""), "summary": pl.get("summary_raw",""),
                                           "actual": pl.get("observation",""), "expected": pl.get("expected",""),
                                           "procedure": pl.get("procedure",""), "markets": pl.get("markets",""),
                                           "description_full": pl.get("description_raw","")})
                        if len(candidates) >= top_k: break
                if not candidates: prog_bar.progress((idx+1)/total); continue

                new_ai = {"summary": item["summary"], "actual": item["observation"], "expected": item["expected"], "procedure": item["procedure"], "markets": item.get("markets","")}
                prompt = build_rerank_prompt(new_ai, candidates)
                try:
                    raw_text = call_llm_with_timeout(prompt)
                    ai_results = parse_ai_json(raw_text)
                    if not ai_results:
                        short_prompt = build_rerank_prompt(new_ai, candidates[:3])
                        raw_text = call_llm_with_timeout(short_prompt)
                        ai_results = parse_ai_json(raw_text)
                    if not ai_results: continue

                    cand_market_map = {c["jira_key"].strip().upper(): str(c.get("markets","")) for c in candidates}
                    ECU_KEYWORDS = ["EPS","BMS","BCM","MHU","IPC","MDU","FMCU","OBC","DCDC","EVCC","ETG","SRR","MRR","USS","WCBS","FCAM","ACU","MCU","DSCU","ELK","LKA","LDW","AEB","FCW","BSD","DOW","TSR","AHB","FPA"]
                    best = None
                    for r in ai_results:
                        raw_score = int(float(r.get("confidence_score") or 0))
                        r_key = str(r.get("jira_key","")).strip().upper()
                        jira_mkt = cand_market_map.get(r_key,"")
                        new_mkt = (item.get("markets") or "").strip().upper()
                        adj_score, notes = raw_score, ""
                        if new_mkt and jira_mkt and new_mkt != jira_mkt.strip().upper():
                            adj_score = max(0, adj_score - 20); notes += f" [Khác Market: {new_mkt}≠{jira_mkt}, -20đ]"
                        scope_match = r.get("scope_match", True)
                        if not scope_match and adj_score > 55: adj_score = 55; notes += " [Scope khác: cap 55]"
                        adj_cls = "DUPLICATE" if adj_score >= 88 else "NEAR_DUP" if adj_score >= 70 else "SIMILAR" if adj_score >= 50 else "NOT_RELATED"
                        if adj_score >= threshold:
                            if best is None or adj_score > best["score"]:
                                best = {"idx": idx, "jira_key": r.get("jira_key",""), "score": adj_score, "score_raw": raw_score,
                                        "classification": adj_cls, "scope_match": scope_match, "reason": f"{r.get('reason','')}{notes}",
                                        "jira_market": jira_mkt, "jira_summary": next((c.get("summary","") for c in candidates if c["jira_key"]==r.get("jira_key","")),""),
                                        "jira_description": next((c.get("description_full","") for c in candidates if c["jira_key"]==r.get("jira_key","")),"")}
                    if best: matches.append(best)
                except Exception as e:
                    pass
                prog_bar.progress((idx+1)/total)

            status_ph.success(f"✅ Phân tích xong {total} tickets!")

            # Build result
            match_by_idx = {m["idx"]: m for m in matches}
            final = []
            for item in new_items:
                m = match_by_idx.get(item["idx"])
                row = {"NEW_Summary": item["summary_raw"], "NEW_Market": item.get("markets",""), "NEW_Observation": item["description_raw"],
                       "Jira_Link": "", "Jira_Summary": "", "Jira_Market": "", "Jira_Description": "",
                       "Score": "", "Classification": "", "Lý do match": "Không phát hiện trùng lặp", "_score": 0}
                if m:
                    row.update({"Jira_Link": f"{JIRA_URL}/browse/{m['jira_key']}" if m['jira_key'] else "",
                                "Jira_Summary": m.get("jira_summary",""), "Jira_Market": m.get("jira_market",""),
                                "Jira_Description": m.get("jira_description",""), "Score": f"{m['score']}%",
                                "Classification": m["classification"], "Lý do match": m["reason"], "_score": m["score"]})
                final.append(row)

            result_df = pd.DataFrame(final).sort_values("_score", ascending=False).drop(columns=["_score"]).reset_index(drop=True)
            n_dup = len([m for m in matches if m["classification"]=="DUPLICATE"])
            n_near = len([m for m in matches if m["classification"]=="NEAR_DUP"])
            n_sim = len([m for m in matches if m["classification"]=="SIMILAR"])
            n_new = total - len(matches)

            c1,c2,c3,c4,c5 = st.columns(5)
            c1.markdown(metric_card("Tổng", total, "#475569"), unsafe_allow_html=True)
            c2.markdown(metric_card("DUPLICATE", n_dup, "#ef4444"), unsafe_allow_html=True)
            c3.markdown(metric_card("NEAR DUP", n_near, "#f59e0b"), unsafe_allow_html=True)
            c4.markdown(metric_card("SIMILAR", n_sim, "#3b82f6"), unsafe_allow_html=True)
            c5.markdown(metric_card("NEW", n_new, "#10b981"), unsafe_allow_html=True)

            for i, row in result_df.iterrows():
                cls = row["Classification"]
                tag = classification_tag(cls)
                with st.expander(f"{tag} [{row['Score'] or '—'}] {row['NEW_Summary'][:90]}", expanded=False):
                    cl, cr = st.columns(2)
                    with cl:
                        st.markdown("**🆕 NEW**")
                        st.markdown(f"**Summary:** {row['NEW_Summary']}")
                        if row["NEW_Observation"]: st.markdown(f"**Observation:** {row['NEW_Observation'][:500]}")
                    with cr:
                        if row["Jira_Summary"]:
                            st.markdown(f"**🔗 Match:** {row['Jira_Link']}")
                            st.markdown(f"**Score:** `{row['Score']}` | **{cls}**")
                    if row["Lý do match"]:
                        st.markdown(f'<div class="reason-box">{row["Lý do match"]}</div>', unsafe_allow_html=True)

            st.divider()
            buf = io.BytesIO()
            result_df.to_excel(buf, index=False)
            st.download_button("📥 Tải kết quả (.xlsx)", buf.getvalue(), f"ket_qua_v5_{datetime.now():%Y%m%d_%H%M}.xlsx")