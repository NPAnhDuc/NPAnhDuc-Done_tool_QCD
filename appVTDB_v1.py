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

# Đảm bảo lấy đúng file .env cùng thư mục với script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH, override=True)

# ====================== CONFIG ======================
st.set_page_config(page_title="QCD Tool – Vector Search", page_icon="🔍", layout="wide")

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
    .stSlider [data-baseweb="slider"] { }
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
gemini_key  = os.getenv("GEMINI_API_KEY", "").strip()
jira_token  = os.getenv("JIRA_API_TOKEN", "").strip()
JIRA_URL    = os.getenv("JIRA_URL", "").rstrip("/")
JIRA_PROJECT = os.getenv("JIRA_PROJECT_KEY", "")

if not gemini_key or not jira_token:
    st.error("❌ Thiếu GEMINI_API_KEY hoặc JIRA_API_TOKEN trong file .env")
    st.stop()

def clear_conflicting_auth():
    """Xoá các biến môi trường gây xung đột xác thực Google Cloud SDK."""
    for var in [
        "GOOGLE_APPLICATION_CREDENTIALS", 
        "GOOGLE_OAUTH_ACCESS_TOKEN", 
        "GCLOUD_PROJECT", 
        "GOOGLE_CLOUD_PROJECT",
        "CREDENTIALS",
        "CLOUDSDK_CONFIG"
    ]:
        if var in os.environ:
            del os.environ[var]

clear_conflicting_auth()

os.environ["GOOGLE_API_KEY"] = gemini_key
os.environ["NO_GCE_CHECK"] = "true"
os.environ["GOOGLE_AUTH_SUPPRESS_CREDENTIALS_WARNINGS"] = "true"

genai.configure(api_key=gemini_key)

# Cấu hình safety settings để tránh lỗi 400 khi gặp log lỗi hoặc từ ngữ kỹ thuật
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# Hàm tìm model phù hợp nhất cho generateContent
def get_best_generation_model() -> str:
    try:
        available_models = [
            m.name for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]
        
        # Ưu tiên gemini-1.5-flash, sau đó là gemini-1.5-pro, rồi các bản experimental
        for preferred_name in ["gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-1.5-pro", "gemini-1.5-pro-latest", "gemini-2.0-flash-exp", "gemini-2.0-pro-exp"]:
            if f"models/{preferred_name}" in available_models:
                return f"models/{preferred_name}"
            if preferred_name in available_models: # Một số SDK trả về không có tiền tố "models/"
                return preferred_name
        
        if available_models:
            return available_models[0] # Lấy model đầu tiên nếu không tìm thấy ưu tiên
            
    except Exception as e:
        st.error(f"❌ Lỗi khi liệt kê các model Gemini khả dụng: {e}. Vui lòng kiểm tra API Key và kết nối mạng.")
        st.stop()
    
    st.error("❌ Không tìm thấy model Gemini nào hỗ trợ generateContent với API Key của bạn.")
    st.stop()
    return "" # Should not reach here

llm_model = genai.GenerativeModel(
    get_best_generation_model(), # Tự động tìm model phù hợp nhất
    generation_config={"temperature": 0, "response_mime_type": "application/json"},
    safety_settings=safety_settings
)

# ====================== VECTOR DB SETUP ======================
CHROMA_COLLECTION_NAME = "jira_issues_v2"   # ← đổi tên ở đây khi nâng version DB schema
# ChromaDB lưu dữ liệu tại ./chroma_db/ (persistent trên ổ đĩa)
# Mỗi ticket Jira được lưu với:
#   id        → jira_key (VF6LHD-12345) — tránh trùng khi upsert
#   document  → "Summary: <clean> | Actual: <obs> | Expected: <exp>" — văn bản để tạo embedding
#   metadata  → {key, summary_raw, actual, expected, procedure, synced_at} — dùng để AI rerank

class GeminiEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """
    Embedding function dùng Gemini API cho ChromaDB.

    Root cause của lỗi cũ:
      - genai.embed_content(content=LIST) với task_type trả về response.embedding (singular)
        chứ không phải response.embeddings (list) → parse sai → empty / crash.
      - Fix: gọi TỪNG TEXT MỘT qua embed_content(content=STRING) rồi gom lại.
      - Text Jira chứa \xa0, !image!, * markup → clean trước khi gửi API.
    """

    # Models theo thứ tự ưu tiên (GA 2026)
    # Lưu ý: text-embedding-004 hiện là bản ổn định nhất
    MODELS_TO_TRY = [
        "models/text-embedding-004",     
        "models/gemini-embedding-001",   
        "models/gemini-embedding-2"
    ]

    # Cache model đang hoạt động — chỉ detect 1 lần, tránh thử lại mỗi batch
    _working_model: str = None
    _vector_dim: int = None

    # ── Bước 1: clean Jira wiki markup ──────────────────────────────────────
    @staticmethod
    def clean_jira_text(text: str) -> str:
        if not isinstance(text, str):
            text = str(text or "")
        # Xoá Jira image attachment: !filename.png|width=256,height=192!
        text = re.sub(r'!\S+\.(png|jpg|jpeg|gif|bmp|svg)(\|[^!]*)?' + r'!', '', text, flags=re.IGNORECASE)
        # Xoá Jira macro: {color:#xxx}, {panel}, {noformat:...}
        text = re.sub(r'\{[a-zA-Z][^}]{0,80}\}', '', text)
        # Xoá wiki bold/italic: * ** _
        text = re.sub(r'\*{1,3}', ' ', text)
        text = re.sub(r'\b_([^_]+)_\b', r'\1', text)
        # Xoá horizontal rule ----
        text = re.sub(r'-{4,}', ' ', text)
        # Xoá Jira list markers: # ## !! @@
        text = re.sub(r'(?m)^[#!@]+\s*', '', text)
        # Non-breaking space và Windows line endings → space
        text = text.replace('\xa0', ' ').replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Giới hạn độ dài — Gemini embedding-001 giới hạn ~2048 tokens (~8000 chars)
        return text[:7500]

    # ── Bước 2: gọi API cho 1 string, trả về list[float] ────────────────────
    @classmethod
    def _embed_one(cls, model: str, text: str) -> list:
        """Gọi embed_content với content=STRING (không phải list)."""
        resp = genai.embed_content(
            model=model,
            content=text,              # ← STRING, không phải list → trả .embedding đơn
            task_type="retrieval_document"
        )
        # SDK mới: resp là EmbedContentResponse có thuộc tính .embedding (list[float])
        # SDK cũ / dict: resp["embedding"]
        if hasattr(resp, "embedding"):
            vec = resp.embedding
        elif isinstance(resp, dict):
            vec = resp.get("embedding") or resp.get("embeddings", [None])[0]
        else:
            # Fallback cuối
            vec = list(vars(resp).values())[0]

        # Nếu vec vẫn là object (ContentEmbedding), lấy .values
        if vec is not None and hasattr(vec, "values"):
            vec = vec.values

        if not vec:
            raise ValueError(f"embed_content trả vector rỗng. Model={model}, text[:80]={text[:80]!r}")

        return [float(x) for x in vec]

    # ── Bước 3: detect model hoạt động ──────────────────────────────────────
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
                return
            except Exception as e:
                last_exception = e
                err = str(e).lower()
                # Lỗi auth dứt khoát (401/403) → raise ngay, không thử model tiếp
                if any(x in err for x in ["401", "403", "invalid", "expired", "unauthenticated"]):
                    raise RuntimeError(
                        "❌ GEMINI_API_KEY không hợp lệ, hết hạn hoặc không có quyền truy cập.\n"
                        f"Chi tiết từ Google: {str(e)}\n"
                        "Vui lòng kiểm tra lại file .env, lưu lại và KHỞI ĐỘNG LẠI Terminal."
                    ) from e
                # 400 (bad request) / 404 (model deprecated/not found) → thử model tiếp
                continue
        raise RuntimeError(
            f"❌ Không có model embedding nào hoạt động.\n"
            f"Đã thử: {cls.MODELS_TO_TRY}\n"
            f"Lỗi cuối cùng: {str(last_exception)}"
        )

    # ── Bước 4: entry point ChromaDB gọi ────────────────────────────────────
    def __call__(self, input_texts) -> list:
        """
        ChromaDB gọi hàm này với input_texts: list[str].
        Trả về list[list[float]] — 1 vector cho mỗi text.
        """
        self._detect_model()
        model = self._working_model

        results = []
        for raw_text in input_texts:
            cleaned = self.clean_jira_text(raw_text)
            # Đảm bảo không embed chuỗi rỗng
            if not cleaned:
                cleaned = "empty document"
            try:
                vec = self._embed_one(model, cleaned)
            except Exception as e:
                err = str(e).lower()
                if "401" in err or "unauthenticated" in err:
                    raise RuntimeError("❌ GEMINI_API_KEY không hợp lệ.") from e
                # Text bị từ chối (quá dài, ký tự lạ) → zero vector
                # Ticket vẫn được lưu vào DB, chỉ không match tốt bằng
                dim = self._vector_dim or 768
                vec = [0.0] * dim
            results.append(vec)

        return results

@st.cache_resource
def get_chroma_collection(api_key_hash: str):
    """Khởi tạo collection, phụ thuộc vào hash của API Key để tự động refresh khi đổi Key."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    emb_fn = GeminiEmbeddingFunction()
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}
    )

def get_collection():
    """Luôn lấy collection qua hàm này — tránh stale reference sau khi xoá/tạo lại DB."""
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
        # customfield_14506 là ID chính xác của trường Markets dựa trên metadata bạn cung cấp
        params = {
            "jql": jql, "startAt": start_at, "maxResults": 100,
            "fields": "key,summary,description,status,issuetype,customfield_14506,customfield_12101,customfield_market"
        }
        resp = jira_session.get(f"{JIRA_URL}/rest/api/2/search", params=params, verify=False)
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
        f = issue.get("fields", {}) # 'f' contains the fields from Jira
        # Thử lấy dữ liệu Markets từ các ID phổ biến theo thứ tự ưu tiên
        market_raw = (
            f.get("customfield_14506") or 
            f.get("customfield_12101") or 
            f.get("customfield_markets") or 
            f.get("customfield_market") or ""
        )
        
        if isinstance(market_raw, dict):
            market_raw = market_raw.get("value", "")
        elif isinstance(market_raw, list):
            market_raw = ", ".join([m.get("value", m) if isinstance(m, dict) else str(m) for m in market_raw])
        market_raw = str(market_raw or "")

        # Fallback mạnh hơn: quét cả summary và description nếu custom field trống
        if not market_raw:
            search_text = f"{f.get('summary', '')} {f.get('description', '') or ''}" # Fallback to extract market from text
            # Tìm tag trong ngoặc [EU] hoặc đứng độc lập có ranh giới từ
            m_match = re.search(r'\[\s*(KZ|VN|EU|ME|UAE|AU|US|CAN|NA)\s*\]|\b(KZ|VN|EU|ME|UAE|AU|US|CAN|NA)\b', search_text, re.IGNORECASE)
            if m_match:
                market_raw = (m_match.group(1) or m_match.group(2)).upper()

        rows.append({
            "Key":         issue.get("key", ""),
            "Summary":     f.get("summary", ""),
            "Description": f.get("description", "") or "",
            "Status":      f.get("status", {}).get("name", ""), # Output column name is "Markets"
            "Markets":     market_raw, 
        })
    return pd.DataFrame(rows)

# ====================== TEXT EXTRACTION ======================
# extract_zones: tách Description thành 4 vùng riêng biệt
# Đây là bước QUAN TRỌNG NHẤT — chất lượng vector phụ thuộc vào đây

def extract_zones(raw_desc: str) -> dict:
    """
    Tách Description thành 4 vùng:
      observation  — Actual Result / Observation (ưu tiên cao nhất)
      expected     — Expected Result
      procedure    — Test Procedure / Steps
    Logic: tìm marker cuối cùng cho observation (rfind), marker đầu tiên cho các vùng còn lại.
    """
    text = raw_desc or ""

    OBS_MARKERS  = ["observation:", "actual observation:", "actual result:",
                    "actual results:", "iv. actual", "4. actual result",
                    "2.4. actual", "actual:", "4. actual", "[actual]"]
    EXP_MARKERS  = ["expected result:", "expected results:", "expect result:",
                    "expectation:", "iii. expect", "3. expect",
                    "2.3. expected", "test expected", "expected:"]
    PROC_MARKERS = ["test procedure:", "test procedure :", "step/test procedure:",
                    "steps to reproduce", "procedure:", "test step",
                    "2.2. test step", "how to reproduce:", "test steps:",
                    "ii. test condition", "step:", "steps:", "[step]",
                    "[test procedure]"]
    END_MARKERS  = ["recovery:", "recover:", "frequency:", "occurrences:",
                    "attachment:", "remark:", "reference:", "2.5", "2.6",
                    "v. frequency", "vi. recover"]

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

    # Observation — lấy marker CUỐI CÙNG (actual result thực sự)
    obs_pos, obs_len = find_last(OBS_MARKERS)
    if obs_pos != -1:
        observation = extract_section(obs_pos, obs_len, EXP_MARKERS + END_MARKERS)
    else:
        # Thay vì lấy đuôi, hãy quét các dòng chứa từ khóa mang tính hành vi lỗi
        lines = text.split('\n')
        error_lines = [l for l in lines if any(k in l.lower() for k in ['fail', 'error', 'wrong', 'not', 'issue', 'bug', 'bị'])]
        if error_lines:
            observation = " ".join(error_lines)
        else:
            observation = text # Lấy toàn bộ text nếu không phân vùng được

    # Expected, Procedure — lấy marker đầu tiên
    exp_pos,  exp_len  = find_first(EXP_MARKERS)
    proc_pos, proc_len = find_first(PROC_MARKERS)

    expected  = extract_section(exp_pos,  exp_len,  OBS_MARKERS + PROC_MARKERS + END_MARKERS)
    procedure = extract_section(proc_pos, proc_len, EXP_MARKERS + OBS_MARKERS  + END_MARKERS)

    return {
        "observation": observation,
        "expected":    expected,
        "procedure":   procedure,
    }

def extract_core_summary(summary: str) -> str:
    """Strip tất cả tag kỹ thuật VinFast, chỉ giữ mô tả lỗi thực sự."""
    if not isinstance(summary, str):
        return ""
    s = summary
    # Strip nhiều lớp tag lồng nhau
    for _ in range(8):
        prev = s
        s = re.sub(r'^\s*\[[^\]]*\]\s*', '', s).strip()          # [tag]
        s = re.sub(r'^\s*VF6\w*[-_]?\s*[|\-]?\s*', '', s, flags=re.IGNORECASE).strip()
        s = re.sub(r'^\s*VF\d\w*[-_]?\s*[|\-]?\s*', '', s, flags=re.IGNORECASE).strip()
        s = re.sub(r'^\s*EE[_\s]\w+[-_]?\s*[|\-]?\s*', '', s, flags=re.IGNORECASE).strip()
        s = re.sub(r'^\s*EEC-\w+[-_]?\s*[|\-]?\s*', '', s, flags=re.IGNORECASE).strip()
        s = re.sub(r'^\s*(ePT|FPT|ODX|CTF|AUTO)[-_]?\w*\s*[|\-]?\s*', '', s, flags=re.IGNORECASE).strip()
        s = re.sub(r'^\s*FRS\s*\S+\s*[|\-]?\s*', '', s, flags=re.IGNORECASE).strip()
        if s == prev:
            break
    # Pipe-delimited: "VF6-EU | PLUS | FRS... | <lỗi thực sự>"
    if s.count('|') >= 2:
        parts = [p.strip() for p in s.split('|')]
        s = parts[-1] if parts[-1] else parts[-2]
    return s or summary

def build_vector_document(summary: str, zones: dict) -> str:
    """
    Tạo văn bản đầu vào cho embedding.
    Dùng Actual Result (trọng số cao nhất) + Expected Result (scope gate) + Summary.

    Lý do thiết kế:
      - Actual Result: repeat 2 lần → tăng ảnh hưởng trong embedding space,
        đây là yếu tố đo độ tương đồng lỗi thực tế.
      - Expected Result: đưa vào 1 lần → giúp vector search loại bỏ các ticket
        cùng Actual nhưng khác mục đích/scope (e.g., "display wrong" vs "system crash").
      - Procedure/Steps: KHÔNG embed → thường là boilerplate, gây nhiễu cosine.
    """
    parts = []
    clean_sum = extract_core_summary(summary)
    obs = zones.get("observation", "").strip()
    exp = zones.get("expected", "").strip()

    # Actual Result — trọng số cao nhất, repeat có chủ đích
    if obs:
        parts.append(f"Fault: {obs[:500]}")
        parts.append(f"Actual: {obs[:300]}")   # repeat để tăng ảnh hưởng

    # Expected Result — scope gate: giúp phân biệt ticket cùng lỗi nhưng khác mục đích
    if exp:
        parts.append(f"Expected: {exp[:250]}")

    if clean_sum:
        parts.append(f"Summary: {clean_sum}")

    return " | ".join(parts)

# ====================== DB SYNC ======================
def sync_jira_to_vector_db(df: pd.DataFrame, progress_placeholder=None) -> dict:
    """
    Upsert Jira issues vào ChromaDB.
    ID = Jira Key (VF6LHD-xxxxx) → safe to re-run, chỉ update ticket đã tồn tại.
    Metadata lưu đầy đủ: key, summary_raw, observation, expected, procedure, synced_at.
    """
    col = get_collection()
    ids, documents, metadatas = [], [], []
    total = len(df)
    skipped = 0

    for i, row in df.iterrows():
        key  = str(row.get("Key", f"row-{i}")).strip()
        if not key:
            skipped += 1
            continue

        summary = str(row.get("Summary", ""))
        desc    = str(row.get("Description", ""))
        status  = str(row.get("Status", ""))
        market  = str(row.get("Markets", ""))
        zones   = extract_zones(desc)
        doc     = build_vector_document(summary, zones)

        # Bỏ qua ticket không có nội dung — ChromaDB không nhận empty string
        if not doc.strip():
            skipped += 1
            continue

        ids.append(key)
        documents.append(doc)
        metadatas.append({
            "key":         key,
            "summary_raw": summary,
            "observation": zones["observation"],
            "expected":    zones["expected"],
            "procedure":   zones["procedure"],
            "description_raw": desc, # Lưu full description để xuất file
            "status":      status,
            "markets":     market,
            "synced_at":   datetime.now().isoformat()
        })

        if progress_placeholder and i % 50 == 0:
            progress_placeholder.progress(min((i + 1) / total, 1.0))

    if ids:
        # Nhỏ batch (25) → mỗi lần chỉ embed 25 text, tránh rate limit Gemini
        batch_size = 25
        for b in range(0, len(ids), batch_size):
            try:
                col.upsert(
                    ids=ids[b:b+batch_size],
                    documents=documents[b:b+batch_size],
                    metadatas=metadatas[b:b+batch_size]
                )
                time.sleep(0.5)  # Thêm delay nhỏ để tránh làm nghẽn API (lỗi 429/400)
            except Exception as e:
                if "401" in str(e) or "unauthenticated" in str(e).lower():
                    raise RuntimeError("❌ Lỗi xác thực: API Key đã bị từ chối giữa chừng.") from e
                raise e

    return {"total": total, "upserted": len(ids), "skipped": skipped}

def list_available_embedding_models() -> list:
    """Liệt kê tất cả embedding models đang khả dụng trong API key hiện tại."""
    try:
        return [
            m.name for m in genai.list_models()
            if "embedContent" in m.supported_generation_methods
        ]
    except Exception as e:
        return [f"Lỗi khi list models: {e}"]

def get_db_stats() -> dict:
    """Trả về thông tin trạng thái DB hiện tại."""
    col = get_collection()
    count = col.count()
    stats = {"count": count, "last_synced": "—", "sample_keys": []}
    if count > 0:
        sample = col.get(limit=3, include=["metadatas"])
        metas = sample.get("metadatas", [])
        if metas:
            dates = [m.get("synced_at", "") for m in metas if m.get("synced_at")]
            if dates:
                stats["last_synced"] = max(dates)[:19].replace("T", " ")
            stats["sample_keys"] = [m.get("key", "") for m in metas[:3]]
    return stats

# ====================== AI RERANKING PROMPT ======================
def build_rerank_prompt(new_item: dict, candidates: list) -> str:
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
  SAME SCOPE: proceed to Phase 2 scoring normally.

PHASE 2 - SIMILARITY SCORING (Actual Result - primary driver):
The "actual" field is the main determinant of the score.

Keyword scoring per match:
  +3 pts — SPECIFIC fault behavior only: exact DTC code (P0xxx, U0xxx), exact signal value (0x7F, 0x1),
           exact fault state (limphome, VehFailGrade_ERR, Walk-away lock, Tow mode, Star button, IN-PROGRESS)
  +2 pts — component + symptom PAIR: "MHU freeze", "DMS false alarm", "HWA not activate",
           "ACC disengage", "FOTA fail", "BMS SOC mismatch", "DDAW missed", "HWA unavailable"
  +1 pt  — single generic term: VCU, BMS, DTC, MHU, warning, display, drive, connect, ON, OFF, speed

  ⚠️ CRITICAL: Generic terms alone (+1 pt each) CANNOT push score above 35.
               Score above 50 requires at least one +2 or +3 match in the actual field.

Score floors — ONLY apply when SAME SCOPE (Phase 1 passed) AND actual content describes the SAME fault behavior:
  actual >= 85% match AND exact same component AND same fault mode  -->  final >= 88  (DUPLICATE)
  actual >= 70% match AND same component                            -->  final >= 72  (NEAR_DUP)
  actual >= 55% match (specific keywords, NOT generic terms only)   -->  final >= 55  (SIMILAR)
  actual < 40%  OR only generic terms match (VCU/DTC/warning)       -->  final <= 30  (NOT_RELATED)

  CRITICAL: If DIFFERENT SCOPE (Phase 1 failed) --> final score MUST be <= 55, SIMILAR or lower.
  CRITICAL: If actual fields describe DIFFERENT fault behaviors --> score MUST be < 50 regardless of shared component names.

NEVER penalize for: different FRS/TPV version, markets (KZ/VN/EU/ME), model (ECO/PLUS/FL), VIN, date.
Note: market difference is handled separately outside AI scoring.

CLASSIFICATION:
  85-100 --> DUPLICATE
  70-84  --> NEAR_DUP
  50-69  --> SIMILAR
  < 50   --> NOT_RELATED (still return it)

REASON FORMAT (mandatory, all 4 sections required):
[Expected Result]: SAME SCOPE / DIFFERENT SCOPE -- brief explanation of what each ticket expects
[Actual Result]: <new_actual ~10w> approx <master_actual ~10w>
[Keywords matched]: <comma-separated exact+synonym matches>
[Conclusion]: <1 sentence: classification + key similarity/difference, mention scope cap if applied>

NEW TICKET:
{json.dumps(new_item, ensure_ascii=False)}

TOP CANDIDATES FROM JIRA:
{json.dumps(candidates, ensure_ascii=False)}

Return ONLY a JSON array, one object per candidate, sorted by confidence_score descending:
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

# ====================== HELPER UI ======================
def metric_card(label, value, color):
    return f"""<div class="metric-card" style="border-left-color:{color}">
        <div class="metric-label">{label}</div>
        <div class="metric-num" style="color:{color}">{value}</div>
    </div>"""

def classification_tag(cls):
    mapping = {
        "DUPLICATE": '<span class="tag-dup">DUPLICATE</span>',
        "NEAR_DUP":  '<span class="tag-near">NEAR DUP</span>',
        "SIMILAR":   '<span class="tag-sim">SIMILAR</span>',
        "NOT_RELATED": '<span class="tag-new">NEW</span>',
        "":          '<span class="tag-new">NEW</span>',
    }
    return mapping.get(cls, cls)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.markdown("### ⚙️ Cấu hình")
    # --- PHẦN DEBUG KEY ---
    with st.expander("🔑 Kiểm tra API Key hiện tại"):
        st.caption(f"File: `{ENV_PATH}`")
        if gemini_key:
            st.code(f"Key ends with: ...{gemini_key[-4:]}")
    st.divider()
    new_file  = st.file_uploader("📂 Upload New List (Excel/CSV)", type=["xlsx", "csv"])
    jql_input = st.text_area(
        "JQL Query (Master List từ Jira)",
        value=f'project = "{JIRA_PROJECT}" ORDER BY created DESC',
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
<div class="db-row"><span class="db-key">Embedding model</span><span>gemini-embedding-001 (Gemini)</span></div>
<div class="db-row"><span class="db-key">Distance metric</span><span>Cosine similarity</span></div>
<div class="db-row"><span class="db-key">Sample IDs</span><span>{" | ".join(stats["sample_keys"])}</span></div>
</div>""", unsafe_allow_html=True)

    # Kiểm tra embedding models khả dụng
    with st.expander("🔎 Kiểm tra Embedding Models khả dụng (click nếu lỗi 404)"):
        if st.button("Liệt kê models"):
            with st.spinner("Đang query Gemini API..."):
                available = list_available_embedding_models()
            if available:
                st.success("Models hỗ trợ embedContent trong API key của bạn:")
                for m in available:
                    st.code(m)
                st.info("Nếu lỗi 404, copy tên model ở trên và cập nhật MODELS_TO_TRY trong code.")
            else:
                st.error("Không lấy được danh sách. Kiểm tra GEMINI_API_KEY.")

    st.markdown("#### 📥 Tải dữ liệu Jira")
    st.info("""**Cách hoạt động:**
- Tải các tickets từ Jira dựa trên JQL Query bạn cung cấp.
- Dữ liệu sẽ được hiển thị và có thể tải về dưới dạng Excel.
- Dữ liệu này sau đó có thể được dùng để đồng bộ vào Vector DB.
""")

    col_fetch, col_download = st.columns([3, 1])
    with col_fetch:
        if st.button("⬇️ Tải dữ liệu Jira (từ JQL)", use_container_width=True, key="fetch_jira_data_btn"):
            with st.spinner("Đang tải dữ liệu từ Jira..."):
                master_raw = fetch_jira_issues(jql_input)
            if master_raw.empty:
                st.error("Không lấy được dữ liệu từ Jira. Kiểm tra JQL và token.")
            else:
                st.session_state["fetched_jira_data"] = master_raw
                st.success(f"✅ Đã tải {len(master_raw)} tickets từ Jira.")
                st.rerun() # Rerun to display the data and download button

    if "fetched_jira_data" in st.session_state and not st.session_state["fetched_jira_data"].empty:
        st.markdown("##### Dữ liệu Jira đã tải:")
        st.dataframe(st.session_state["fetched_jira_data"], use_container_width=True)
        with col_download:
            buffer = io.BytesIO()
            st.session_state["fetched_jira_data"].to_excel(buffer, index=False)
            st.download_button(
                "📥 Tải về Excel",
                buffer.getvalue(),
                f"jira_data_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="download_fetched_jira_data_btn"
            )

    st.markdown("#### Đồng bộ dữ liệu Jira → Vector DB")
    st.info("""**Cách hoạt động:**
- Mỗi ticket Jira được tách thành 4 vùng: **Observation**, **Expected**, **Procedure**, **Summary**
- Tạo embedding từ văn bản kết hợp 4 vùng → lưu vào ChromaDB
- **ID = Jira Key** (VF6LHD-xxxxx) → chạy lại chỉ cập nhật, không tạo trùng
- Lưu persistent tại `./chroma_db/` — dữ liệu giữ nguyên khi restart app""")

    col_sync, col_clear = st.columns([3, 1])
    with col_sync:
        if st.button("🔄 Đồng bộ dữ liệu đã tải → DB (upsert)", use_container_width=True, key="sync_to_db_btn"):
            # Dùng data đã tải trong session_state nếu có, tránh fetch lại Jira
            data_to_sync = st.session_state.get("fetched_jira_data", None)
            if data_to_sync is None or data_to_sync.empty:
                with st.spinner("Chưa có dữ liệu, đang tải từ Jira..."):
                    data_to_sync = fetch_jira_issues(jql_input)
            if data_to_sync.empty:
                st.error("Không lấy được dữ liệu từ Jira. Kiểm tra JQL và token.")
            else:
                prog = st.progress(0)
                st.info(f"Đang tạo embeddings và upsert {len(data_to_sync)} tickets...")
                result = sync_jira_to_vector_db(data_to_sync, prog)
                prog.progress(1.0)
                st.success(f"✅ Upserted **{result['upserted']}** tickets (bỏ qua {result['skipped']} thiếu key). DB hiện có **{get_collection().count()}** tickets.")
                st.rerun()

    with col_clear:
        if st.button("🗑️ Xoá DB", use_container_width=True):
            try:
                chroma_client_local = chromadb.PersistentClient(path=CHROMA_PATH)
                chroma_client_local.delete_collection(CHROMA_COLLECTION_NAME)
                # Reset cache Streamlit + reset working model cache
                st.cache_resource.clear()
                GeminiEmbeddingFunction._working_model = None
                GeminiEmbeddingFunction._vector_dim    = None
                st.success("Đã xoá. Refresh lại trang.")
                st.rerun()
            except Exception as e:
                st.error(f"Lỗi xoá DB: {e}")

    # Preview DB entries
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
        st.stop()

    # ── Load new file ──
    with st.spinner("Đang đọc file..."):
        new_df = pd.read_excel(new_file) if new_file.name.endswith(".xlsx") else pd.read_csv(new_file)

    _col = get_collection()
    if _col.count() == 0:
        st.error("⚠️ Vector DB đang trống. Vào tab 'Quản lý Database' để sync Jira trước.")
        st.stop()

    st.markdown(f"**File:** `{new_file.name}` — **{len(new_df)} tickets** cần kiểm tra")

    # ── Prepare new items ──
    new_items = []
    # Tìm cột linh hoạt hơn: không phân biệt hoa thường và chấp nhận "Market" hoặc "Markets"
    sum_col = next((c for c in new_df.columns if str(c).strip().upper() in ["SUMMARY", "NEW_SUMMARY"]), "Summary")
    desc_col = next((c for c in new_df.columns if str(c).strip().upper() in ["DESCRIPTION", "NEW_DESCRIPTION"]), "Description")
    market_col = next((c for c in new_df.columns if str(c).strip().upper() in ["MARKET", "MARKETS", "NEW_MARKET"]), None)

    for i, row in new_df.iterrows():
        summary = str(row.get(sum_col, ""))
        desc    = str(row.get(desc_col, ""))
        
        # Lấy giá trị market từ Excel và lọc bỏ các giá trị rỗng/NaN
        if market_col and market_col in new_df.columns:
            m_val = row.get(market_col)
        else:
            m_val = None
        if m_val is None or (isinstance(m_val, float) and pd.isna(m_val)) \
                or str(m_val).strip().lower() in ["nan", "none", ""]:
            market = ""
        else:
            market = str(m_val).strip()

        # Fallback đồng bộ với logic Jira
        if not market:
            for _src in [summary, desc]:
                m_match = re.search(r'\[\s*(KZ|VN|EU|ME|UAE|AU|US|CAN|NA)\s*\]|\b(KZ|VN|EU|ME|UAE|AU|US|CAN|NA)\b', _src, re.IGNORECASE)
                if m_match:
                    market = (m_match.group(1) or m_match.group(2)).upper()
                    break

        zones   = extract_zones(desc)
        new_items.append({
            "idx":         i,
            "summary":     extract_core_summary(summary),
            "summary_raw": summary,
            "description_raw": desc,
            "observation": zones["observation"],
            "expected":    zones["expected"],
            "procedure":   zones["procedure"],
            "markets":     market,
        })

    # ── Run analysis ──
    matches   = []
    prog_bar  = st.progress(0)
    status_ph = st.empty()
    total     = len(new_items)

    for idx, item in enumerate(new_items):
        status_ph.text(f"⏳ Tiến độ: {idx+1}/{total} tickets...")

        # Step 2: Vector Search — tìm Top-K candidates
        query_text = build_vector_document(item["summary_raw"], {
            "observation": item["observation"],
            "expected":    item["expected"],
            "procedure":   item["procedure"]
        })

        # Ngưỡng cosine distance — ChromaDB metric="cosine" trả distance (0=identical, 2=opposite)
        # distance ≤ 0.45 ≈ similarity ≥ 0.55; candidates xa hơn là nhiễu
        DISTANCE_THRESHOLD = 0.45
        item_market = item.get("markets", "").strip().upper()

        try:
            # Ưu tiên filter cùng market để giảm nhiễu cross-market
            fetch_k = top_k + 3   # lấy thêm buffer để bù sau khi filter distance
            results = None
            if item_market:
                try:
                    results = _col.query(
                        query_texts=[query_text],
                        n_results=fetch_k,
                        where={"markets": {"$eq": item_market}}
                    )
                    # Nếu quá ít kết quả cùng market → fallback không filter
                    if len(results["ids"][0]) < 2:
                        results = None
                except Exception:
                    results = None

            if results is None:
                results = _col.query(query_texts=[query_text], n_results=fetch_k)

        except Exception as e:
            st.warning(f"Vector search lỗi item {idx}: {e}")
            prog_bar.progress((idx + 1) / total)
            continue

        # Lọc candidates theo cosine distance threshold
        candidates = []
        distances  = results.get("distances", [[]])[0]
        for i in range(len(results["ids"][0])):
            dist = distances[i] if i < len(distances) else 1.0
            if dist > DISTANCE_THRESHOLD:
                continue   # bỏ candidates quá xa, là nhiễu
            m = results["metadatas"][0][i]
            candidates.append({
                "jira_key":  m.get("key", ""),
                "summary":   m.get("summary_raw", ""),
                "actual":    m.get("observation", ""),
                "expected":  m.get("expected", ""),
                "procedure": m.get("procedure", ""),
                "markets":   m.get("markets", ""),
                "description_full": m.get("description_raw", ""),
            })
            if len(candidates) >= top_k:
                break   # đủ top_k candidates sạch

        if not candidates:
            prog_bar.progress((idx + 1) / total)
            continue

        # Step 3: AI Reranking
        new_item_for_ai = {
            "summary":   item["summary"],
            "actual":    item["observation"],
            "expected":  item["expected"],
            "procedure": item["procedure"],
            "markets":   item.get("markets", ""),
        }
        prompt = build_rerank_prompt(new_item_for_ai, candidates)

        try:
            # Step 3: AI Reranking (Xử lý retry nếu gặp lỗi Quota 429)
            raw_text = ""
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    resp = llm_model.generate_content(prompt)
                    raw_text = resp.text.strip()
                    break
                except Exception as e:
                    err_msg = str(e).lower()
                    if ("429" in err_msg or "quota" in err_msg) and attempt < max_retries - 1:
                        wait_sec = (attempt + 1) * 12 # Đợi 12s, 24s...
                        status_ph.warning(f"⚠️ Hết Quota (429) tại dòng {idx+1}. Đang đợi {wait_sec}s để thử lại lần {attempt+1}...")
                        time.sleep(wait_sec)
                    else:
                        raise e

            # response_mime_type="application/json" đã ép Gemini trả JSON sạch.
            # Fallback strip markdown phòng các bản preview vẫn thêm fence.
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                raw_text = re.sub(r"\s*```$", "", raw_text).strip()

            ai_results = json.loads(raw_text)
            if not isinstance(ai_results, list):
                ai_results = [ai_results]

            # Build lookup map: jira_key → market
            candidate_market_map = {
                c.get("jira_key", "").strip().upper(): str(c.get("markets", ""))
                for c in candidates
            }

            # Lấy match có score cao nhất vượt threshold
            best = None
            for r in ai_results:
                raw_score  = int(r.get("confidence_score", 0))
                r_key      = str(r.get("jira_key", "")).strip().upper()
                jira_mkt   = candidate_market_map.get(r_key, "")
                new_mkt    = (item.get("markets") or "").strip().upper()

                # ── Market penalty ──
                # Nếu cả hai đều có market VÀ khác nhau → trừ 20 điểm
                if new_mkt and jira_mkt and new_mkt != jira_mkt.strip().upper():
                    penalty   = 20
                    adj_score = max(0, raw_score - penalty)
                    mkt_note  = f" [Lưu ý: Không cùng Market: {new_mkt} ≠ {jira_mkt}, trừ {penalty} điểm]"
                else:
                    adj_score = raw_score
                    mkt_note  = ""

                # Lấy dữ liệu full từ candidate record
                cand_info = next((c for c in candidates if c["jira_key"] == r_key), {})
                jira_full_desc = cand_info.get("description_full", "")
                if not jira_full_desc: # Fallback nếu DB cũ chưa có description_raw
                    jira_full_desc = f"Actual: {cand_info.get('actual','')}\nExpected: {cand_info.get('expected','')}\nSteps: {cand_info.get('procedure','')}"

                # ── Scope gate ──
                # Nếu AI báo scope khác nhau → cap score ≤ 55 VÀ không được ra DUPLICATE/NEAR_DUP
                scope_match = r.get("scope_match", True)  # default True nếu AI cũ không trả field này
                scope_note  = ""
                if not scope_match and adj_score > 55:
                    adj_score  = 55
                    scope_note = " [Scope khác nhau: Expected Result không cùng mục đích, score bị cap 55]"

                # ── Re-classify sau tất cả adjustments ──
                if   adj_score >= 88: adj_cls = "DUPLICATE"
                elif adj_score >= 70: adj_cls = "NEAR_DUP"
                elif adj_score >= 50: adj_cls = "SIMILAR"
                else:                 adj_cls = "NOT_RELATED"

                # ── Fix 1: Consistency guard ──
                # scope_match=False → không được là DUPLICATE (AI đôi khi inconsistent)
                if not scope_match and adj_cls == "DUPLICATE":
                    adj_cls    = "NEAR_DUP"
                    scope_note += " [Override: scope khác nhau, hạ từ DUPLICATE → NEAR_DUP]"

                # ── Fix 2: Market penalty sync vào reason conclusion ──
                # Nếu market penalty đã hạ classification, rewrite conclusion trong reason
                if mkt_note and raw_score != adj_score:
                    ai_reason   = r.get("reason", "")
                    penalty_cls = adj_cls  # classification sau penalty
                    orig_cls    = ("DUPLICATE" if raw_score >= 88
                                   else "NEAR_DUP" if raw_score >= 70
                                   else "SIMILAR" if raw_score >= 50
                                   else "NOT_RELATED")
                    if orig_cls != penalty_cls:
                        # Thêm dòng override vào cuối reason để tránh misleading conclusion
                        mkt_note += f" [Classification điều chỉnh: {orig_cls} → {penalty_cls} do khác market]"

                # ── Fix 3: ECU/component penalty ──
                # Nếu cùng fault pattern nhưng khác ECU component → trừ 10 điểm
                # Detect bằng cách check summary có chứa ECU keyword khác nhau không
                new_summary_upper  = item.get("summary", "").upper()
                cand_summary_upper = cand_info.get("summary", "").upper()
                ECU_KEYWORDS = ["EPS", "BMS", "BCM", "MHU", "IPC", "MDU", "FMCU", "OBC",
                                "DCDC", "EVCC", "ETG", "SRR", "MRR", "USS", "WCBS",
                                "FCAM", "ACU", "MCU", "DSCU", "ELK", "LKA", "LDW",
                                "AEB", "FCW", "BSD", "DOW", "TSR", "AHB", "FPA"]
                new_ecus  = {k for k in ECU_KEYWORDS if k in new_summary_upper}
                cand_ecus = {k for k in ECU_KEYWORDS if k in cand_summary_upper}
                # Chỉ áp penalty khi cả hai đều có ECU keyword rõ ràng VÀ không overlap
                if new_ecus and cand_ecus and new_ecus.isdisjoint(cand_ecus):
                    ecu_penalty = 10
                    adj_score   = max(0, adj_score - ecu_penalty)
                    ecu_note    = f" [Khác ECU/component: {'/'.join(sorted(new_ecus))} ≠ {'/'.join(sorted(cand_ecus))}, trừ {ecu_penalty} điểm]"
                    # Re-classify lại sau ECU penalty
                    if   adj_score >= 88: adj_cls = "DUPLICATE"
                    elif adj_score >= 70: adj_cls = "NEAR_DUP"
                    elif adj_score >= 50: adj_cls = "SIMILAR"
                    else:                 adj_cls = "NOT_RELATED"
                    mkt_note += ecu_note  # append vào notes chung để hiển thị

                if adj_score >= threshold:
                    if best is None or adj_score > best["score"]:
                        best = {
                            "idx":            idx,
                            "jira_key":       str(r.get("jira_key", "")),
                            "score":          adj_score,
                            "score_raw":      raw_score,
                            "classification": adj_cls,
                            "scope_match":    scope_match,
                            "reason":         f"{r.get('reason', '')}{mkt_note}{scope_note}",
                            "jira_market":    jira_mkt,
                            "jira_summary":   cand_info.get("summary", ""),
                            "jira_description": jira_full_desc,
                        }

            if best:
                matches.append(best)

        except json.JSONDecodeError as e:
            st.warning(f"⚠️ AI rerank item {idx}: JSON parse lỗi — {str(e)[:100]}\nRaw: {raw_text[:120]}")
        except Exception as e:
            st.warning(f"⚠️ AI rerank item {idx}: {str(e)[:120]}")

        prog_bar.progress((idx + 1) / total)

    status_ph.success(f"✅ Phân tích xong {total} tickets!")

    # ── Build result DataFrame ──
    matches_by_idx = {m["idx"]: m for m in matches}
    final_rows = []

    for item in new_items:
        m = matches_by_idx.get(item["idx"])
        row = {
            "NEW_Summary":    item["summary_raw"],
            "NEW_Market":     item.get("markets", ""),
            "NEW_Observation":item["description_raw"],
            "Jira_Link":      "",
            "Jira_Summary":   "",
            "Jira_Market":    "",
            "Jira_Description": "",
            "Score":          "",
            "Score_Raw":      "",
            "Classification": "",
            "Lý do match":    "Không phát hiện trùng lặp. Đủ điều kiện tạo mới.",
            "Note":           "",
            "_score_val":     0,
        }
        if m:
            jira_key = m["jira_key"]
            raw_s = m.get("score_raw", m["score"])
            adj_s = m["score"]
            score_display = f"{adj_s}%"
            score_raw_display = f"{raw_s}%" if raw_s != adj_s else ""
            row.update({
                "Jira_Link":      f"{JIRA_URL}/browse/{jira_key}" if jira_key else "",
                "Jira_Summary":   m.get("jira_summary", ""),
                "Jira_Market":    m.get("jira_market", ""),
                "Jira_Description": m.get("jira_description", ""),
                "Score":          score_display,
                "Score_Raw":      score_raw_display,
                "Classification": m["classification"],
                "Lý do match":    m["reason"],
                "_score_val":     adj_s,
            })
        final_rows.append(row)

    result_df = (
        pd.DataFrame(final_rows)
        .sort_values("_score_val", ascending=False)
        .drop(columns=["_score_val"])
        .reset_index(drop=True)
    )

    # ── Suggested Merge Group ──
    # Nhóm các ticket có cùng Jira match → gợi ý QA engineer xem xét merge
    # Đặc biệt hữu ích cho các bug "factory reset settings" hay "FOTA fail ECU khác nhau"
    jira_to_rows: dict = {}
    for i, row in result_df.iterrows():
        jkey = row.get("Jira_Link", "")
        if jkey:
            jira_to_rows.setdefault(jkey, []).append(i)

    merge_group_col = [""] * len(result_df)
    group_id = 1
    for jkey, idxs in jira_to_rows.items():
        if len(idxs) >= 2:   # chỉ đánh nhóm khi có ≥2 ticket cùng match 1 Jira key
            label = f"Group-{group_id:02d} ({len(idxs)} tickets)"
            for i in idxs:
                merge_group_col[i] = label
            group_id += 1
    result_df.insert(result_df.columns.get_loc("Note"), "Merge_Group", merge_group_col)

    # ── Metrics ──
    n_dup    = len([m for m in matches if m["classification"] == "DUPLICATE"])
    n_near   = len([m for m in matches if m["classification"] == "NEAR_DUP"])
    n_sim    = len([m for m in matches if m["classification"] == "SIMILAR"])
    n_new    = total - len(matches)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(metric_card("Tổng kiểm tra", total, "#475569"), unsafe_allow_html=True)
    c2.markdown(metric_card("DUPLICATE",  n_dup,  "#ef4444"), unsafe_allow_html=True)
    c3.markdown(metric_card("NEAR DUP",   n_near, "#f59e0b"), unsafe_allow_html=True)
    c4.markdown(metric_card("SIMILAR",    n_sim,  "#3b82f6"), unsafe_allow_html=True)
    c5.markdown(metric_card("NEW (sạch)", n_new,  "#10b981"), unsafe_allow_html=True)

    # ── Visualization Chart ──
    st.write("")
    chart_data = pd.DataFrame({
        "Số lượng": [n_dup, n_near, n_sim, n_new]
    }, index=["DUPLICATE", "NEAR DUP", "SIMILAR", "NEW (Sạch)"])
    
    with st.expander("📊 Biểu đồ phân bổ kết quả", expanded=True):
        st.bar_chart(chart_data, color="#2563eb", horizontal=True)

    # ── Table ──
    st.divider()
    st.markdown("#### Kết quả chi tiết")

    # Hiển thị từng dòng với reason được format đẹp
    for i, row in result_df.iterrows():
        cls   = row["Classification"]
        score = row["Score"]
        summary_display = row["NEW_Summary"][:90]

        tag_html = classification_tag(cls)
        with st.expander(f"{tag_html} &nbsp; [{score or '—'}] &nbsp; {summary_display}", expanded=False):
            col_left, col_right = st.columns([1, 1])
            with col_left:
                st.markdown("**🆕 NEW Ticket**")
                st.markdown(f"**Summary:** {row['NEW_Summary']}")
                if row.get("NEW_Market"):
                    st.markdown(f"**Market:** `{row['NEW_Market']}`")
                if row["NEW_Observation"]:
                    st.markdown(f"**Observation:** {row['NEW_Observation'][:1000]}") # Chỉ giới hạn hiển thị UI, Excel vẫn full
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
                        st.markdown(f"**🔀 Merge Group:** `{row['Merge_Group']}`  \n*(Nhiều ticket cùng match Jira key này — xem xét merge)*")
            if row["Lý do match"]:
                st.markdown("**📋 Lý do match:**")
                st.markdown(f'<div class="reason-box">{row["Lý do match"]}</div>', unsafe_allow_html=True)



    # ── Download ──
    st.divider()
    buf = io.BytesIO()
    # Chỉ lấy các cột user yêu cầu cho file Excel
    export_cols = ["NEW_Summary", "NEW_Market", "NEW_Observation", "Jira_Link", "Jira_Summary",
                   "Jira_Market", "Jira_Description", "Score", "Score_Raw", "Classification",
                   "Lý do match", "Merge_Group", "Note"]
    result_df[export_cols].to_excel(buf, index=False)
    st.download_button(
        "📥 Tải kết quả (.xlsx)",
        buf.getvalue(),
        f"ket_qua_vectordb_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )