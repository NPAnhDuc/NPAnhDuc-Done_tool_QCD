import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import os
import chromadb
import requests
import re
import io
import time
import hashlib
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from chromadb.utils import embedding_functions
from datetime import datetime
from dotenv import load_dotenv
import urllib3
import logging
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# Đảm bảo lấy đúng file .env cùng thư mục với script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH, override=True)

# #region agent log
DEBUG_LOG_PATH = os.path.join(BASE_DIR, "debug-d4063f.log")
def _debug_log(location, message, data=None, hypothesis_id=""):
    try:
        import json as _json
        entry = {"sessionId": "d4063f", "timestamp": int(time.time() * 1000),
                 "location": location, "message": message,
                 "data": data or {}, "hypothesisId": hypothesis_id, "runId": "pre-fix"}
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as _f:
            _f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass
# #endregion

# ====================== CONFIG ======================
st.set_page_config(page_title="QCD Tool – Vector Search", page_icon="🔍", layout="wide")

# ── Tunable constants ──
DISTANCE_THRESHOLD = 0.45
DESC_RAW_MAX_CHARS = 6000
AI_TIMEOUT_SECONDS = 30   # Giảm xuống 30s để phát hiện lỗi nhanh hơn

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background-color: #0f1117; color: #e2e8f0; }
    .stTabs [data-baseweb="tab-list"] { background: #1a1f2e; border-radius: 8px; padding: 4px; gap: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 6px; color: #94a3b8; font-weight: 500; font-size: 13px; }
    .stTabs [aria-selected="true"] { background: #2563eb !important; color: white !important; }
    .metric-card {
        background: #1a1f2e; padding: 18px 20px; border-radius: 10px;
        border: 1px solid #2d3748; border-left: 3px solid;
    }
    .metric-num { font-size: 28px; font-weight: 600; font-family: 'IBM Plex Mono', monospace; margin-top: 4px; }
    .metric-label { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }
    .db-info-box {
        background: #1a1f2e; border: 1px solid #2d3748; border-radius: 10px;
        padding: 16px 20px; font-family: 'IBM Plex Mono', monospace; font-size: 13px;
    }
    .db-row { display: flex; justify-content: space-between; padding: 6px 0;
              border-bottom: 1px solid #2d374840; color: #cbd5e1; }
    .db-row:last-child { border-bottom: none; }
    .db-key { color: #64748b; }
    .tag-dup  { background: #7f1d1d; color: #fca5a5; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
    .tag-near { background: #78350f; color: #fcd34d; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
    .tag-sim  { background: #1e3a5f; color: #93c5fd; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
    .tag-new  { background: #14532d; color: #86efac; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
    div[data-testid="stButton"] > button {
        background: #2563eb; color: white; border: none; border-radius: 8px;
        font-weight: 600; font-size: 13px;
    }
    div[data-testid="stButton"] > button:hover { background: #1d4ed8; }
    .stDataFrame { background: #1a1f2e; border-radius: 10px; }
    .stProgress > div > div { background: #2563eb; }
    .reason-box {
        background: #111827; border: 1px solid #374151; border-radius: 8px;
        padding: 12px 14px; font-size: 12px; font-family: 'IBM Plex Mono', monospace;
        white-space: pre-wrap; color: #d1d5db; line-height: 1.7;
    }
    h1 { color: #f8fafc !important; font-weight: 600 !important; }
    h2, h3 { color: #e2e8f0 !important; font-weight: 500 !important; }
    label, .stSelectbox label, .stTextArea label { color: #94a3b8 !important; font-size: 13px !important; }
    .stTextArea textarea, .stTextInput input {
        background: #1a1f2e !important; color: #e2e8f0 !important;
        border: 1px solid #374151 !important; border-radius: 8px !important;
    }
    .stAlert { border-radius: 8px; }
    .stExpander { background: #1a1f2e; border: 1px solid #2d3748; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ====================== ENV & API SETUP ======================
gemini_key   = os.getenv("GEMINI_API_KEY", "").strip()
jira_token   = os.getenv("JIRA_API_TOKEN", "").strip()
JIRA_URL     = os.getenv("JIRA_URL", "").rstrip("/")
JIRA_PROJECT = os.getenv("JIRA_PROJECT_KEY", "")

def _default_jql() -> str:
    """JIRA_PROJECT_KEY có thể là project key hoặc full JQL."""
    raw = (JIRA_PROJECT or "").strip()
    if not raw:
        return 'project = "VF6" ORDER BY created DESC'
    if "project" in raw.lower() or " and " in raw.lower():
        return raw if "order by" in raw.lower() else f"{raw} ORDER BY created DESC"
    return f'project = "{raw}" ORDER BY created DESC'

# #region agent log
_debug_log("appVTDB_v2.py:env", "env loaded", {
    "env_path": ENV_PATH, "key_prefix": gemini_key[:4] if gemini_key else "",
    "key_len": len(gemini_key), "api_mode": "google_direct",
    "jira_project_loaded": bool(JIRA_PROJECT), "jira_project_len": len(JIRA_PROJECT or ""),
}, "C")
# #endregion

if not gemini_key or not jira_token:
    st.error("❌ Thiếu GEMINI_API_KEY hoặc JIRA_API_TOKEN trong file .env")
    st.stop()

def clear_conflicting_auth():
    for var in ["GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_OAUTH_ACCESS_TOKEN",
                "GCLOUD_PROJECT", "GOOGLE_CLOUD_PROJECT", "CREDENTIALS", "CLOUDSDK_CONFIG"]:
        if var in os.environ:
            del os.environ[var]

clear_conflicting_auth()
os.environ["GOOGLE_API_KEY"] = gemini_key
os.environ["NO_GCE_CHECK"] = "true"
os.environ["GOOGLE_AUTH_SUPPRESS_CREDENTIALS_WARNINGS"] = "true"

# Gọi thẳng Google Gemini API — không qua proxy trung gian
genai.configure(api_key=gemini_key)
# #region agent log
_debug_log("appVTDB_v2.py:genai_configure", "google direct configured", {
    "key_prefix": gemini_key[:4] if gemini_key else "",
}, "F")
# #endregion

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# Model name 
LLM_MODEL_NAME = "gemini-2.5-flash"

llm_model = genai.GenerativeModel(
    LLM_MODEL_NAME,
    generation_config={
        "temperature": 0,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
    },
    safety_settings=safety_settings
)

def call_llm_with_timeout(prompt: str, timeout: int = AI_TIMEOUT_SECONDS) -> str:
    """
    Gọi Gemini với timeout cứng dùng threading.
    Tránh app bị đứng vô hạn khi proxy không phản hồi.
    Trả về raw text, hoặc raise TimeoutError / Exception.
    """
    import threading
    result = {"text": None, "error": None}

    def _call():
        try:
            resp = llm_model.generate_content(prompt)
            result["text"] = resp.text.strip()
        except Exception as e:
            result["error"] = e

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        raise TimeoutError(f"Gemini không phản hồi sau {timeout}s.")
    if result["error"]:
        raise result["error"]
    return result["text"]

# ====================== VECTOR DB SETUP ======================
CHROMA_COLLECTION_NAME = "jira_issues_v2"

class GeminiEmbeddingFunction(embedding_functions.EmbeddingFunction):
    MODELS_TO_TRY = [
        "models/gemini-embedding-001",
        "models/text-embedding-004",
        "models/gemini-embedding-2",
    ]
    _working_model: str = None
    _vector_dim: int = None

    @staticmethod
    def clean_jira_text(text: str) -> str:
        if not isinstance(text, str):
            text = str(text or "")
        text = re.sub(r'!\S+\.(png|jpg|jpeg|gif|bmp|svg)(\|[^!]*)?' + r'!', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\{[a-zA-Z][^}]{0,80}\}', '', text)
        text = re.sub(r'\*{1,3}', ' ', text)
        text = re.sub(r'\b_([^_]+)_\b', r'\1', text)
        text = re.sub(r'-{4,}', ' ', text)
        text = re.sub(r'(?m)^[#!@]+\s*', '', text)
        text = text.replace('\xa0', ' ').replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:7500]

    @classmethod
    def _embed_one(cls, model: str, text: str) -> list:
        # #region agent log
        _debug_log("appVTDB_v2.py:_embed_one", "embed attempt", {"model": model, "text_len": len(text)}, "A")
        # #endregion
        try:
            resp = genai.embed_content(model=model, content=text, task_type="retrieval_document")
        except Exception as e:
            # #region agent log
            _debug_log("appVTDB_v2.py:_embed_one", "embed failed", {
                "model": model, "error_type": type(e).__name__,
                "error_msg": str(e)[:300],
            }, "A")
            # #endregion
            raise
        if hasattr(resp, "embedding"):
            vec = resp.embedding
        elif isinstance(resp, dict):
            vec = resp.get("embedding") or resp.get("embeddings", [None])[0]
        else:
            vec = list(vars(resp).values())[0]
        if vec is not None and hasattr(vec, "values"):
            vec = vec.values
        if not vec:
            raise ValueError(f"embed_content trả vector rỗng. Model={model}, text[:80]={text[:80]!r}")
        return [float(x) for x in vec]

    @classmethod
    def _detect_model(cls):
        if cls._working_model:
            return
        last_exception = None
        for model in cls.MODELS_TO_TRY:
            try:
                vec = cls._embed_one(model, "VinFast QA test")
                cls._working_model = model
                cls._vector_dim    = len(vec)
                # #region agent log
                _debug_log("appVTDB_v2.py:_detect_model", "model detected", {
                    "model": model, "vector_dim": len(vec),
                }, "A")
                # #endregion
                return
            except Exception as e:
                last_exception = e
                err = str(e).lower()
                if any(x in err for x in ["401", "403", "invalid", "expired", "unauthenticated"]):
                    raise RuntimeError(
                        f"❌ GEMINI_API_KEY không hợp lệ hoặc hết hạn.\nChi tiết: {str(e)}\n"
                        "Kiểm tra GEMINI_API_KEY trong .env và khởi động lại Terminal."
                    ) from e
                continue
        raise RuntimeError(
            f"❌ Không có model embedding nào hoạt động.\nĐã thử: {cls.MODELS_TO_TRY}\n"
            f"Lỗi cuối: {str(last_exception)}"
        )

    def __call__(self, input_texts) -> list:
        self._detect_model()
        model = self._working_model
        results = []
        for raw_text in input_texts:
            cleaned = self.clean_jira_text(raw_text)
            if not cleaned:
                cleaned = "empty document"
            try:
                vec = self._embed_one(model, cleaned)
            except Exception as e:
                err = str(e).lower()
                if "401" in err or "unauthenticated" in err:
                    raise RuntimeError("❌ GEMINI_API_KEY không hợp lệ.") from e
                dim = self._vector_dim or 768
                vec = [0.0] * dim
            results.append(vec)
        return results

@st.cache_resource
def get_chroma_collection(api_key_hash: str):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    emb_fn = GeminiEmbeddingFunction()
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}
    )

def get_collection():
    key_hash = hashlib.md5(gemini_key.encode()).hexdigest()
    return get_chroma_collection(key_hash)

# ====================== JIRA ======================
def create_jira_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({
        "Authorization": f"Bearer {jira_token}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    })
    return session

jira_session = create_jira_session()

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
            st.error(f"Jira API lỗi {resp.status_code}: {resp.text[:200]}")
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
        market_raw = (
            f.get("customfield_14506") or f.get("customfield_12101") or
            f.get("customfield_markets") or f.get("customfield_market") or ""
        )
        if isinstance(market_raw, dict):
            market_raw = market_raw.get("value", "")
        elif isinstance(market_raw, list):
            market_raw = ", ".join([m.get("value", m) if isinstance(m, dict) else str(m) for m in market_raw])
        market_raw = str(market_raw or "")

        if not market_raw:
            search_text = f"{f.get('summary', '')} {f.get('description', '') or ''}"
            m_match = re.search(r'\[\s*(KZ|VN|EU|ME|UAE|AU|US|CAN|NA)\s*\]|\b(KZ|VN|EU|ME|UAE|AU|US|CAN|NA)\b', search_text, re.IGNORECASE)
            if m_match:
                market_raw = (m_match.group(1) or m_match.group(2)).upper()

        rows.append({
            "Key":         issue.get("key", ""),
            "Summary":     f.get("summary", ""),
            "Description": f.get("description", "") or "",
            "Status":      f.get("status", {}).get("name", ""),
            "Markets":     market_raw,
        })
    return pd.DataFrame(rows)

# ====================== TEXT EXTRACTION ======================
def extract_zones(raw_desc: str) -> dict:
    text = raw_desc or ""
    OBS_MARKERS  = ["observation:", "actual observation:", "actual result:", "actual results:",
                    "iv. actual", "4. actual result", "2.4. actual", "actual:", "4. actual", "[actual]"]
    EXP_MARKERS  = ["expected result:", "expected results:", "expect result:", "expectation:",
                    "iii. expect", "3. expect", "2.3. expected", "test expected", "expected:"]
    PROC_MARKERS = ["test procedure:", "test procedure :", "step/test procedure:", "steps to reproduce",
                    "procedure:", "test step", "2.2. test step", "how to reproduce:", "test steps:",
                    "ii. test condition", "step:", "steps:", "[step]", "[test procedure]"]
    END_MARKERS  = ["recovery:", "recover:", "frequency:", "occurrences:", "attachment:",
                    "remark:", "reference:", "2.5", "2.6", "v. frequency", "vi. recover"]
    lower = text.lower()

    def find_last(markers):
        best_pos, best_len = -1, 0
        for m in markers:
            idx = lower.rfind(m)
            if idx > best_pos:
                best_pos, best_len = idx, len(m)
        return best_pos, best_len

    def find_first(markers, after=0):
        best_pos, best_len = len(text) + 1, 0
        for m in markers:
            idx = lower.find(m, after)
            if idx != -1 and idx < best_pos:
                best_pos, best_len = idx, len(m)
        return (best_pos if best_pos <= len(text) else -1), best_len

    def extract_section(start_pos, start_len, stop_markers):
        if start_pos == -1:
            return ""
        content_start = start_pos + start_len
        end_pos = len(text)
        for sm in stop_markers:
            idx = lower.find(sm, content_start)
            if idx != -1 and idx < end_pos:
                end_pos = idx
        return text[content_start:end_pos].strip()

    obs_pos, obs_len = find_last(OBS_MARKERS)
    if obs_pos != -1:
        observation = extract_section(obs_pos, obs_len, EXP_MARKERS + END_MARKERS)
    else:
        lines = text.split('\n')
        error_lines = [l for l in lines if any(k in l.lower() for k in ['fail', 'error', 'wrong', 'not', 'issue', 'bug', 'bị'])]
        observation = " ".join(error_lines) if error_lines else text[:1500]

    exp_pos,  exp_len  = find_first(EXP_MARKERS)
    proc_pos, proc_len = find_first(PROC_MARKERS)
    expected  = extract_section(exp_pos,  exp_len,  OBS_MARKERS + PROC_MARKERS + END_MARKERS)
    procedure = extract_section(proc_pos, proc_len, EXP_MARKERS + OBS_MARKERS  + END_MARKERS)
    return {"observation": observation, "expected": expected, "procedure": procedure}

def extract_core_summary(summary: str) -> str:
    if not isinstance(summary, str):
        return ""
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
        if s == prev:
            break
    if s.count('|') >= 2:
        parts = [p.strip() for p in s.split('|')]
        s = parts[-1] if parts[-1] else parts[-2]
    return s or summary

def build_vector_document(summary: str, zones: dict) -> str:
    parts = []
    clean_sum = extract_core_summary(summary)
    obs = zones.get("observation", "").strip()
    exp = zones.get("expected", "").strip()
    if obs:
        parts.append(f"Fault: {obs[:500]}")
        parts.append(f"Actual: {obs[:300]}")
    if exp:
        parts.append(f"Expected: {exp[:250]}")
    if clean_sum:
        parts.append(f"Summary: {clean_sum}")
    return " | ".join(parts)

# ====================== DB SYNC ======================
def sync_jira_to_vector_db(df: pd.DataFrame, progress_placeholder=None) -> dict:
    col = get_collection()
    ids, documents, metadatas = [], [], []
    total = len(df)
    skipped = 0

    for i, row in df.iterrows():
        key = str(row.get("Key", f"row-{i}")).strip()
        if not key:
            skipped += 1
            continue
        summary = str(row.get("Summary", ""))
        desc    = str(row.get("Description", ""))
        status  = str(row.get("Status", ""))
        market  = str(row.get("Markets", ""))
        zones   = extract_zones(desc)
        doc     = build_vector_document(summary, zones)
        if not doc.strip():
            skipped += 1
            continue
        ids.append(key)
        documents.append(doc)
        metadatas.append({
            "key":             key,
            "summary_raw":     summary,
            "observation":     zones["observation"],
            "expected":        zones["expected"],
            "procedure":       zones["procedure"],
            "description_raw": desc[:DESC_RAW_MAX_CHARS],
            "status":          status,
            "markets":         market,
            "synced_at":       datetime.now().isoformat()
        })
        if progress_placeholder and i % 50 == 0:
            progress_placeholder.progress(min((i + 1) / total, 1.0))

    if ids:
        batch_size = 25
        for b in range(0, len(ids), batch_size):
            try:
                col.upsert(
                    ids=ids[b:b+batch_size],
                    documents=documents[b:b+batch_size],
                    metadatas=metadatas[b:b+batch_size]
                )
                time.sleep(0.5)
            except Exception as e:
                if "401" in str(e) or "unauthenticated" in str(e).lower():
                    raise RuntimeError("❌ Lỗi xác thực: API Key bị từ chối giữa chừng.") from e
                raise e

    return {"total": total, "upserted": len(ids), "skipped": skipped}

def list_available_embedding_models() -> list:
    try:
        return [m.name for m in genai.list_models() if "embedContent" in m.supported_generation_methods]
    except Exception as e:
        return [f"Lỗi khi list models: {e}"]

def get_db_stats() -> dict:
    col = get_collection()
    count = col.count()
    stats = {"count": count, "last_synced": "—", "sample_keys": []}
    if count > 0:
        sample = col.get(limit=3, include=["metadatas"])
        metas  = sample.get("metadatas", [])
        if metas:
            dates = [m.get("synced_at", "") for m in metas if m.get("synced_at")]
            if dates:
                stats["last_synced"] = max(dates)[:19].replace("T", " ")
            stats["sample_keys"] = [m.get("key", "") for m in metas[:3]]
    return stats

# ====================== AI RERANKING PROMPT ======================
def slim_for_ai(item: dict, max_field: int = 600) -> dict:
    """Rút gọn field trước khi gửi AI — tránh prompt/output quá dài bị cắt JSON."""
    out = {}
    for k, v in item.items():
        if k == "description_full":
            continue
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
Before scoring, compare the "expected" fields of NEW TICKET vs CANDIDATE.
  SAME SCOPE: both tickets expect the same system behavior/outcome
  DIFFERENT SCOPE: tickets have fundamentally different purposes
  SCOPE RULE: If DIFFERENT SCOPE, cap final score at 55 regardless of Actual similarity.

PHASE 2 - SIMILARITY SCORING (Actual Result - primary driver):
The "actual" field is the main determinant of the score.

Keyword scoring per match:
  +3 pts — SPECIFIC: exact DTC code (P0xxx, U0xxx), exact signal value (0x7F), exact fault state (limphome, VehFailGrade_ERR)
  +2 pts — component + symptom PAIR: "MHU freeze", "DMS false alarm", "HWA not activate", "ACC disengage", "FOTA fail"
  +1 pt  — single generic term: VCU, BMS, DTC, MHU, warning, display

  ⚠️ Generic terms alone CANNOT push score above 35. Score above 50 requires at least one +2 or +3 match.

Score floors (SAME SCOPE only):
  actual >= 85% match AND same component AND same fault mode  -->  final >= 88  (DUPLICATE)
  actual >= 70% match AND same component                      -->  final >= 72  (NEAR_DUP)
  actual >= 55% match (specific keywords)                     -->  final >= 55  (SIMILAR)
  actual < 40%  OR only generic terms                         -->  final <= 30  (NOT_RELATED)

  CRITICAL: DIFFERENT SCOPE --> score MUST be <= 55.
  CRITICAL: DIFFERENT fault behaviors --> score MUST be < 50 regardless of shared component names.

NEVER penalize for: different FRS/TPV version, markets, model (ECO/PLUS/FL), VIN, date.

CLASSIFICATION:
  85-100 --> DUPLICATE | 70-84 --> NEAR_DUP | 50-69 --> SIMILAR | <50 --> NOT_RELATED

REASON FORMAT (all 4 sections required):
[Expected Result]: SAME SCOPE / DIFFERENT SCOPE -- brief explanation
[Actual Result]: <new_actual ~10w> approx <master_actual ~10w>
[Keywords matched]: <comma-separated matches>
[Conclusion]: <1 sentence: classification + key similarity/difference>

NEW TICKET:
{json.dumps(new_slim, ensure_ascii=False)}

TOP CANDIDATES FROM JIRA:
{json.dumps(cand_slim, ensure_ascii=False)}

IMPORTANT: Return ONLY a valid JSON array. No markdown, no explanation, no code block.
Keep each "reason" under 120 words. One object per candidate, sorted by confidence_score descending.
[
  {{
    "jira_key": "<key>",
    "is_duplicate": true/false,
    "confidence_score": <0-100>,
    "classification": "DUPLICATE" | "NEAR_DUP" | "SIMILAR" | "NOT_RELATED",
    "scope_match": true/false,
    "reason": "<structured reason with 4 sections above>"
  }}
]"""

def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text).strip()
    return text

def _salvage_json_array(text: str) -> list:
    """Trích các object JSON hoàn chỉnh từ array bị cắt giữa chừng."""
    start = text.find("[")
    if start < 0:
        return []
    decoder = json.JSONDecoder()
    pos = start + 1
    results = []
    while pos < len(text):
        while pos < len(text) and text[pos] in " \t\n\r,":
            pos += 1
        if pos >= len(text) or text[pos] == "]":
            break
        try:
            obj, end = decoder.raw_decode(text, pos)
            results.append(obj)
            pos = end
        except json.JSONDecodeError:
            break
    return results

def parse_ai_json(raw_text: str) -> list:
    """
    Parse JSON từ response AI — xử lý markdown fence, text thừa, hoặc JSON bị cắt.
    """
    if not raw_text or not raw_text.strip():
        raise json.JSONDecodeError("Response rỗng", raw_text or "", 0)

    text = _strip_json_fences(raw_text)

    for candidate in (text, _strip_json_fences(raw_text)):
        try:
            result = json.loads(candidate)
            return result if isinstance(result, list) else [result]
        except json.JSONDecodeError:
            pass

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            return result if isinstance(result, list) else [result]
        except json.JSONDecodeError:
            pass

    salvaged = _salvage_json_array(text)
    if salvaged:
        return salvaged

    raise json.JSONDecodeError("Không tìm thấy JSON array hợp lệ", text, 0)

# ====================== HELPER UI ======================
def metric_card(label, value, color):
    return f"""<div class="metric-card" style="border-left-color:{color}">
        <div class="metric-label">{label}</div>
        <div class="metric-num" style="color:{color}">{value}</div>
    </div>"""

def classification_tag(cls):
    mapping = {
        "DUPLICATE":   '<span class="tag-dup">DUPLICATE</span>',
        "NEAR_DUP":    '<span class="tag-near">NEAR DUP</span>',
        "SIMILAR":     '<span class="tag-sim">SIMILAR</span>',
        "NOT_RELATED": '<span class="tag-new">NEW</span>',
        "":            '<span class="tag-new">NEW</span>',
    }
    return mapping.get(cls, cls)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.markdown("### ⚙️ Cấu hình")
    with st.expander("🔑 Kiểm tra API Key hiện tại"):
        st.caption(f"File: `{ENV_PATH}` | API: Google trực tiếp")
        if gemini_key:
            st.code(f"Key ends with: ...{gemini_key[-4:]}")
    st.divider()
    new_file  = st.file_uploader("📂 Upload New List (Excel/CSV)", type=["xlsx", "csv"])
    jql_input = st.text_area(
        "JQL Query (Master List từ Jira)",
        value=_default_jql(),
        height=80
    )
    st.divider()
    threshold = st.slider("Ngưỡng tương đồng (%)", 40, 90, 60, 5,
                          help="Chỉ hiển thị kết quả có score ≥ ngưỡng này")
    top_k     = st.slider("Top-K candidates (Vector Search)", 3, 10, 5,
                          help="Số lượng candidates lấy từ Vector DB trước khi AI rerank")
    run_btn   = st.button("🚀 Chạy Phân Tích", type="primary", use_container_width=True)

# ====================== MAIN TABS ======================
st.markdown("## 🔍 QCD Tool — Phân tích Tương đồng (Vector DB)")
tab_run, tab_db = st.tabs(["📊 Phân Tích", "🗄️ Quản lý Database"])

# ─────────── TAB 2: DB MANAGEMENT ───────────
with tab_db:
    st.markdown("### Trạng thái Vector Database")
    stats = get_db_stats()
    c1, c2, c3 = st.columns(3)
    c1.markdown(metric_card("Tickets trong DB", stats["count"], "#3b82f6"), unsafe_allow_html=True)
    c2.markdown(metric_card("Lần sync cuối", stats["last_synced"] or "—", "#8b5cf6"), unsafe_allow_html=True)
    c3.markdown(metric_card("Sample keys", len(stats["sample_keys"]), "#06b6d4"), unsafe_allow_html=True)

    if stats["sample_keys"]:
        st.markdown(f"""<div class="db-info-box">
<div class="db-row"><span class="db-key">Collection</span><span>{CHROMA_COLLECTION_NAME}</span></div>
<div class="db-row"><span class="db-key">Storage path</span><span>./chroma_db/</span></div>
<div class="db-row"><span class="db-key">LLM model</span><span>{LLM_MODEL_NAME}</span></div>
<div class="db-row"><span class="db-key">Embedding model</span><span>{GeminiEmbeddingFunction._working_model or "auto-detect (chưa dùng)"}</span></div>
<div class="db-row"><span class="db-key">Distance metric</span><span>Cosine similarity</span></div>
<div class="db-row"><span class="db-key">Sample IDs</span><span>{" | ".join(stats["sample_keys"])}</span></div>
</div>""", unsafe_allow_html=True)

    with st.expander("🔎 Kiểm tra Embedding Models khả dụng"):
        if st.button("Liệt kê models"):
            with st.spinner("Đang query Gemini API..."):
                available = list_available_embedding_models()
            if available:
                st.success("Models hỗ trợ embedContent:")
                for m in available:
                    st.code(m)
            else:
                st.error("Không lấy được danh sách. Kiểm tra GEMINI_API_KEY.")

    st.markdown("#### 📥 Tải dữ liệu Jira")
    st.info("Tải tickets từ Jira theo JQL → hiển thị + tải về Excel → sync vào Vector DB.")

    col_fetch, col_download = st.columns([3, 1])
    with col_fetch:
        if st.button("⬇️ Tải dữ liệu Jira (từ JQL)", use_container_width=True, key="fetch_jira_data_btn"):
            with st.spinner("Đang tải từ Jira..."):
                master_raw = fetch_jira_issues(jql_input)
            if master_raw.empty:
                st.error("Không lấy được dữ liệu. Kiểm tra JQL và token.")
            else:
                st.session_state["fetched_jira_data"] = master_raw
                st.success(f"✅ Đã tải {len(master_raw)} tickets.")
                st.rerun()

    if "fetched_jira_data" in st.session_state and not st.session_state["fetched_jira_data"].empty:
        st.markdown("##### Dữ liệu Jira đã tải:")
        st.dataframe(st.session_state["fetched_jira_data"], use_container_width=True)
        with col_download:
            buffer = io.BytesIO()
            st.session_state["fetched_jira_data"].to_excel(buffer, index=False)
            st.download_button(
                "📥 Tải về Excel", buffer.getvalue(),
                f"jira_data_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True, key="download_fetched_jira_data_btn"
            )

    st.markdown("#### Đồng bộ dữ liệu Jira → Vector DB")
    st.info("Mỗi ticket tách thành 4 vùng → embedding → ChromaDB. ID = Jira Key, upsert an toàn.")

    col_sync, col_clear = st.columns([3, 1])
    with col_sync:
        if st.button("🔄 Đồng bộ → DB (upsert)", use_container_width=True, key="sync_to_db_btn"):
            data_to_sync = st.session_state.get("fetched_jira_data", None)
            if data_to_sync is None or data_to_sync.empty:
                with st.spinner("Chưa có dữ liệu, đang tải từ Jira..."):
                    data_to_sync = fetch_jira_issues(jql_input)
            if data_to_sync.empty:
                st.error("Không lấy được dữ liệu. Kiểm tra JQL và token.")
            else:
                prog = st.progress(0)
                st.info(f"Đang tạo embeddings cho {len(data_to_sync)} tickets...")
                result = sync_jira_to_vector_db(data_to_sync, prog)
                prog.progress(1.0)
                st.success(f"✅ Upserted **{result['upserted']}** tickets (bỏ qua {result['skipped']}). DB: **{get_collection().count()}** tickets.")
                st.rerun()

    with col_clear:
        if st.button("🗑️ Xoá DB", use_container_width=True):
            try:
                chromadb.PersistentClient(path=CHROMA_PATH).delete_collection(CHROMA_COLLECTION_NAME)
                st.cache_resource.clear()
                GeminiEmbeddingFunction._working_model = None
                GeminiEmbeddingFunction._vector_dim    = None
                st.success("Đã xoá. Refresh trang.")
                st.rerun()
            except Exception as e:
                st.error(f"Lỗi xoá DB: {e}")

    if stats["count"] > 0:
        with st.expander("👁️ Xem mẫu dữ liệu trong DB (10 records đầu)"):
            sample_data = get_collection().get(limit=10, include=["metadatas", "documents"])
            rows = []
            for i, meta in enumerate(sample_data.get("metadatas", [])):
                rows.append({
                    "Key":         meta.get("key", ""),
                    "Summary":     meta.get("summary_raw", "")[:80],
                    "Markets":     meta.get("markets", ""),
                    "Observation": meta.get("observation", "")[:100],
                    "Expected":    meta.get("expected", "")[:60],
                    "Synced At":   meta.get("synced_at", "")[:19]
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ─────────── TAB 1: ANALYSIS ───────────
with tab_run:
    if not run_btn or not new_file:
        st.info("👈 Upload file Excel và nhấn **Chạy Phân Tích** từ sidebar.")
        _db_count = get_collection().count()
        if _db_count == 0:
            st.warning("⚠️ Vector DB đang **trống**. Hãy vào tab **Quản lý Database** để đồng bộ Jira trước.")
        else:
            st.success(f"✅ Vector DB sẵn sàng với **{_db_count}** tickets.")
    else:
        with st.spinner("Đang đọc file..."):
            new_df = pd.read_excel(new_file) if new_file.name.endswith(".xlsx") else pd.read_csv(new_file)

        _col = get_collection()
        if _col.count() == 0:
            st.error("⚠️ Vector DB đang trống. Vào tab 'Quản lý Database' để sync Jira trước.")
        else:
            st.markdown(f"**File:** `{new_file.name}` — **{len(new_df)} tickets** cần kiểm tra")

            # ── Prepare new items ──
            sum_col    = next((c for c in new_df.columns if str(c).strip().upper() in ["SUMMARY", "NEW_SUMMARY"]), "Summary")
            desc_col   = next((c for c in new_df.columns if str(c).strip().upper() in ["DESCRIPTION", "NEW_DESCRIPTION"]), "Description")
            market_col = next((c for c in new_df.columns if str(c).strip().upper() in ["MARKET", "MARKETS", "NEW_MARKET"]), None)

            new_items = []
            for i, row in new_df.iterrows():
                summary = str(row.get(sum_col, ""))
                desc    = str(row.get(desc_col, ""))
                m_val   = row.get(market_col) if market_col else None
                if m_val is None or (isinstance(m_val, float) and pd.isna(m_val)) \
                        or str(m_val).strip().lower() in ["nan", "none", ""]:
                    market = ""
                else:
                    market = str(m_val).strip()
                if not market:
                    for _src in [summary, desc]:
                        m_match = re.search(r'\[\s*(KZ|VN|EU|ME|UAE|AU|US|CAN|NA)\s*\]|\b(KZ|VN|EU|ME|UAE|AU|US|CAN|NA)\b', _src, re.IGNORECASE)
                        if m_match:
                            market = (m_match.group(1) or m_match.group(2)).upper()
                            break
                zones = extract_zones(desc)
                new_items.append({
                    "idx":             i,
                    "summary":         extract_core_summary(summary),
                    "summary_raw":     summary,
                    "description_raw": desc,
                    "observation":     zones["observation"],
                    "expected":        zones["expected"],
                    "procedure":       zones["procedure"],
                    "markets":         market,
                })

            # ── Run analysis ──
            matches   = []
            prog_bar  = st.progress(0)
            status_ph = st.empty()
            total     = len(new_items)

            for idx, item in enumerate(new_items):
                status_ph.text(f"⏳ [{idx+1}/{total}] Đang phân tích: {item['summary_raw'][:60]}...")

                # Vector Search
                query_text = build_vector_document(item["summary_raw"], {
                    "observation": item["observation"],
                    "expected":    item["expected"],
                    "procedure":   item["procedure"]
                })
                item_market = item.get("markets", "").strip().upper()

                try:
                    db_count = _col.count()
                    fetch_k  = min(top_k * 2 + 2, db_count)
                    if fetch_k == 0:
                        prog_bar.progress((idx + 1) / total)
                        continue

                    results = None
                    if item_market:
                        try:
                            results = _col.query(
                                query_texts=[query_text], n_results=fetch_k,
                                where={"markets": {"$eq": item_market}}
                            )
                            if len(results["ids"][0]) < 2:
                                results = None
                        except Exception:
                            results = None

                    if results is None:
                        results = _col.query(query_texts=[query_text], n_results=fetch_k)

                except Exception as e:
                    st.warning(f"⚠️ Vector search lỗi item {idx+1}: {e}")
                    prog_bar.progress((idx + 1) / total)
                    continue

                # Filter by distance
                candidates = []
                distances  = results.get("distances", [[]])[0]
                for i in range(len(results["ids"][0])):
                    dist = distances[i] if i < len(distances) else 1.0
                    if dist > DISTANCE_THRESHOLD:
                        continue
                    m = results["metadatas"][0][i]
                    candidates.append({
                        "jira_key":         m.get("key", ""),
                        "summary":          m.get("summary_raw", ""),
                        "actual":           m.get("observation", ""),
                        "expected":         m.get("expected", ""),
                        "procedure":        m.get("procedure", ""),
                        "markets":          m.get("markets", ""),
                        "description_full": m.get("description_raw", ""),
                    })
                    if len(candidates) >= top_k:
                        break

                if not candidates:
                    prog_bar.progress((idx + 1) / total)
                    continue

                # AI Reranking
                new_item_for_ai = {
                    "summary":   item["summary"],
                    "actual":    item["observation"],
                    "expected":  item["expected"],
                    "procedure": item["procedure"],
                    "markets":   item.get("markets", ""),
                }
                prompt = build_rerank_prompt(new_item_for_ai, candidates)

                try:
                    raw_text = ""
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            status_ph.text(f"⏳ [{idx+1}/{total}] Đang gọi AI... (attempt {attempt+1}/{max_retries})")
                            raw_text = call_llm_with_timeout(prompt, timeout=AI_TIMEOUT_SECONDS)
                            break
                        except TimeoutError as e:
                            if attempt < max_retries - 1:
                                status_ph.warning(f"⏱️ Timeout item {idx+1}, thử lại ({attempt+2}/{max_retries})...")
                                time.sleep(3)
                            else:
                                raise
                        except Exception as e:
                            err_msg = str(e).lower()
                            if ("429" in err_msg or "quota" in err_msg) and attempt < max_retries - 1:
                                wait_sec = (attempt + 1) * 5 # Giảm thời gian chờ quota
                                status_ph.warning(f"⚠️ Quota 429 item {idx+1}. Chờ {wait_sec}s...")
                                time.sleep(wait_sec)
                            else:
                                raise

                    # Parse JSON — thử trực tiếp, retry với prompt rút gọn nếu fail
                    try:
                        ai_results = parse_ai_json(raw_text)
                    except json.JSONDecodeError:
                        status_ph.warning(f"⚠️ JSON lỗi item {idx+1}, thử lại với 3 candidates...")
                        short_prompt = build_rerank_prompt(new_item_for_ai, candidates[:3])
                        raw_text2 = call_llm_with_timeout(short_prompt, timeout=AI_TIMEOUT_SECONDS)
                        ai_results = parse_ai_json(raw_text2)  # raise nếu vẫn lỗi

                    candidate_market_map = {
                        c.get("jira_key", "").strip().upper(): str(c.get("markets", ""))
                        for c in candidates
                    }

                    ECU_KEYWORDS = ["EPS", "BMS", "BCM", "MHU", "IPC", "MDU", "FMCU", "OBC",
                                    "DCDC", "EVCC", "ETG", "SRR", "MRR", "USS", "WCBS",
                                    "FCAM", "ACU", "MCU", "DSCU", "ELK", "LKA", "LDW",
                                    "AEB", "FCW", "BSD", "DOW", "TSR", "AHB", "FPA"]

                    best = None
                    for r in ai_results:
                        raw_score = int(float(r.get("confidence_score") or 0))
                        r_key     = str(r.get("jira_key", "")).strip().upper()
                        jira_mkt  = candidate_market_map.get(r_key, "")
                        new_mkt   = (item.get("markets") or "").strip().upper()

                        # Market penalty
                        if new_mkt and jira_mkt and new_mkt != jira_mkt.strip().upper():
                            adj_score = max(0, raw_score - 20)
                            mkt_note  = f" [Khác Market: {new_mkt} ≠ {jira_mkt}, -20đ]"
                        else:
                            adj_score = raw_score
                            mkt_note  = ""

                        cand_info      = next((c for c in candidates if c["jira_key"] == r_key), {})
                        jira_full_desc = cand_info.get("description_full", "") or \
                                         f"Actual: {cand_info.get('actual','')}\nExpected: {cand_info.get('expected','')}\nSteps: {cand_info.get('procedure','')}"

                        # Scope gate
                        scope_match = r.get("scope_match", True)
                        scope_note  = ""
                        if not scope_match and adj_score > 55:
                            adj_score  = 55
                            scope_note = " [Scope khác: score cap 55]"

                        # Classify
                        if   adj_score >= 88: adj_cls = "DUPLICATE"
                        elif adj_score >= 70: adj_cls = "NEAR_DUP"
                        elif adj_score >= 50: adj_cls = "SIMILAR"
                        else:                 adj_cls = "NOT_RELATED"

                        if not scope_match and adj_cls == "DUPLICATE":
                            adj_cls     = "NEAR_DUP"
                            scope_note += " [Override DUPLICATE→NEAR_DUP]"

                        # Market penalty classification note
                        if mkt_note and raw_score != adj_score:
                            orig_cls = ("DUPLICATE" if raw_score >= 88 else "NEAR_DUP" if raw_score >= 70
                                        else "SIMILAR" if raw_score >= 50 else "NOT_RELATED")
                            if orig_cls != adj_cls:
                                mkt_note += f" [{orig_cls}→{adj_cls}]"

                        # ECU penalty
                        new_ecus  = {k for k in ECU_KEYWORDS if k in item.get("summary", "").upper()}
                        cand_ecus = {k for k in ECU_KEYWORDS if k in cand_info.get("summary", "").upper()}
                        if new_ecus and cand_ecus and new_ecus.isdisjoint(cand_ecus):
                            adj_score = max(0, adj_score - 10)
                            mkt_note += f" [Khác ECU: {'/'.join(sorted(new_ecus))}≠{'/'.join(sorted(cand_ecus))}, -10đ]"
                            if   adj_score >= 88: adj_cls = "DUPLICATE"
                            elif adj_score >= 70: adj_cls = "NEAR_DUP"
                            elif adj_score >= 50: adj_cls = "SIMILAR"
                            else:                 adj_cls = "NOT_RELATED"

                        if adj_score >= threshold:
                            if best is None or adj_score > best["score"]:
                                best = {
                                    "idx":              idx,
                                    "jira_key":         str(r.get("jira_key", "")),
                                    "score":            adj_score,
                                    "score_raw":        raw_score,
                                    "classification":   adj_cls,
                                    "scope_match":      scope_match,
                                    "reason":           f"{r.get('reason', '')}{mkt_note}{scope_note}",
                                    "jira_market":      jira_mkt,
                                    "jira_summary":     cand_info.get("summary", ""),
                                    "jira_description": jira_full_desc,
                                }

                    if best:
                        matches.append(best)

                except TimeoutError as e:
                    st.warning(f"⏱️ Timeout item {idx+1} sau {AI_TIMEOUT_SECONDS}s — bỏ qua, tiếp tục.")
                except json.JSONDecodeError as e:
                    st.warning(f"⚠️ AI rerank item {idx+1}: JSON lỗi — {str(e)[:80]}\nRaw: {raw_text[:100]}")
                except Exception as e:
                    st.warning(f"⚠️ AI rerank item {idx+1}: {str(e)[:120]}")

                prog_bar.progress((idx + 1) / total)

            status_ph.success(f"✅ Phân tích xong {total} tickets!")

            # ── Build result DataFrame ──
            matches_by_idx = {m["idx"]: m for m in matches}
            final_rows = []
            for item in new_items:
                m   = matches_by_idx.get(item["idx"])
                row = {
                    "NEW_Summary":      item["summary_raw"],
                    "NEW_Market":       item.get("markets", ""),
                    "NEW_Observation":  item["description_raw"],
                    "Jira_Link":        "",
                    "Jira_Summary":     "",
                    "Jira_Market":      "",
                    "Jira_Description": "",
                    "Score":            "",
                    "Score_Raw":        "",
                    "Classification":   "",
                    "Lý do match":      "Không phát hiện trùng lặp. Đủ điều kiện tạo mới.",
                    "Note":             "",
                    "_score_val":       0,
                }
                if m:
                    adj_s = m["score"]
                    raw_s = m.get("score_raw", adj_s)
                    row.update({
                        "Jira_Link":        f"{JIRA_URL}/browse/{m['jira_key']}" if m["jira_key"] else "",
                        "Jira_Summary":     m.get("jira_summary", ""),
                        "Jira_Market":      m.get("jira_market", ""),
                        "Jira_Description": m.get("jira_description", ""),
                        "Score":            f"{adj_s}%",
                        "Score_Raw":        f"{raw_s}%" if raw_s != adj_s else "",
                        "Classification":   m["classification"],
                        "Lý do match":      m["reason"],
                        "_score_val":       adj_s,
                    })
                final_rows.append(row)

            result_df = (
                pd.DataFrame(final_rows)
                .sort_values("_score_val", ascending=False)
                .drop(columns=["_score_val"])
                .reset_index(drop=True)
            )

            # Merge Group
            jira_to_rows: dict = {}
            for i, row in result_df.iterrows():
                jkey = row.get("Jira_Link", "")
                if jkey:
                    jira_to_rows.setdefault(jkey, []).append(i)

            merge_group_col = [""] * len(result_df)
            group_id = 1
            for jkey, idxs in jira_to_rows.items():
                if len(idxs) >= 2:
                    label = f"Group-{group_id:02d} ({len(idxs)} tickets)"
                    for i in idxs:
                        merge_group_col[i] = label
                    group_id += 1
            result_df.insert(result_df.columns.get_loc("Note"), "Merge_Group", merge_group_col)

            # Metrics
            n_dup  = len([m for m in matches if m["classification"] == "DUPLICATE"])
            n_near = len([m for m in matches if m["classification"] == "NEAR_DUP"])
            n_sim  = len([m for m in matches if m["classification"] == "SIMILAR"])
            n_new  = total - len(matches)

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.markdown(metric_card("Tổng kiểm tra", total, "#475569"), unsafe_allow_html=True)
            c2.markdown(metric_card("DUPLICATE",     n_dup,  "#ef4444"), unsafe_allow_html=True)
            c3.markdown(metric_card("NEAR DUP",      n_near, "#f59e0b"), unsafe_allow_html=True)
            c4.markdown(metric_card("SIMILAR",       n_sim,  "#3b82f6"), unsafe_allow_html=True)
            c5.markdown(metric_card("NEW (sạch)",    n_new,  "#10b981"), unsafe_allow_html=True)

            chart_data = pd.DataFrame({"Số lượng": [n_dup, n_near, n_sim, n_new]},
                                      index=["DUPLICATE", "NEAR DUP", "SIMILAR", "NEW (Sạch)"])
            with st.expander("📊 Biểu đồ phân bổ kết quả", expanded=True):
                st.bar_chart(chart_data, color="#2563eb", horizontal=True)

            st.divider()
            st.markdown("#### Kết quả chi tiết")
            for i, row in result_df.iterrows():
                cls   = row["Classification"]
                score = row["Score"]
                tag_html = classification_tag(cls)
                with st.expander(f"{tag_html} &nbsp; [{score or '—'}] &nbsp; {row['NEW_Summary'][:90]}", expanded=False):
                    col_left, col_right = st.columns([1, 1])
                    with col_left:
                        st.markdown("**🆕 NEW Ticket**")
                        st.markdown(f"**Summary:** {row['NEW_Summary']}")
                        if row.get("NEW_Market"):
                            st.markdown(f"**Market:** `{row['NEW_Market']}`")
                        if row["NEW_Observation"]:
                            st.markdown(f"**Observation:** {row['NEW_Observation'][:1000]}")
                    with col_right:
                        if row["Jira_Summary"]:
                            st.markdown(f"**🔗 JIRA Match:** {row['Jira_Link']}")
                            if row.get("Jira_Market"):
                                st.markdown(f"**Market (Jira):** `{row['Jira_Market']}`")
                            score_info = f"`{score}`"
                            if row.get("Score_Raw"):
                                score_info += f" *(raw: {row['Score_Raw']})*"
                            st.markdown(f"**Score:** {score_info} &nbsp; **Classification:** `{cls}`")
                            if row.get("Merge_Group"):
                                st.markdown(f"**🔀 Merge Group:** `{row['Merge_Group']}`")
                    if row["Lý do match"]:
                        st.markdown("**📋 Lý do match:**")
                        st.markdown(f'<div class="reason-box">{row["Lý do match"]}</div>', unsafe_allow_html=True)

            st.divider()
            buf = io.BytesIO()
            export_cols = ["NEW_Summary", "NEW_Market", "NEW_Observation", "Jira_Link", "Jira_Summary",
                           "Jira_Market", "Jira_Description", "Score", "Score_Raw", "Classification",
                           "Lý do match", "Merge_Group", "Note"]
            result_df[export_cols].to_excel(buf, index=False)
            st.download_button(
                "📥 Tải kết quả (.xlsx)", buf.getvalue(),
                f"ket_qua_vectordb_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )