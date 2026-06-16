"""
Streamlit App đơn giản - Dùng Gemini API (free) thay vì Ollama/vLLM
Không cần Docker, không cần Ollama, chạy được ngay!

Cách dùng:
1. Lấy Gemini API key tại: https://aistudio.google.com/apikey
2. Thêm vào file .env: GEMINI_API_KEY=your_key_here
3. Chạy: streamlit run app_streamlit_simple.py
"""

import streamlit as st
import json
import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import urllib3
import re

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load .env
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH, override=True)

# ====================== CONFIG ======================
st.set_page_config(page_title="QCD Tool - Đơn giản (Gemini)", page_icon="🔍", layout="wide")

JIRA_URL = os.getenv("JIRA_URL", "").rstrip("/")
JIRA_TOKEN = os.getenv("JIRA_API_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

if not GEMINI_API_KEY:
    st.error("❌ Thiếu GEMINI_API_KEY trong file .env. Lấy tại: https://aistudio.google.com/apikey")
    st.stop()

# ====================== GEMINI ======================
try:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_MODEL = "models/gemini-2.0-flash-lite"  # Nhanh, free
except ImportError:
    st.error("❌ Cần cài: pip install google-generativeai")
    st.stop()

def call_gemini(prompt: str) -> str:
    """Gọi Gemini API - free, không cần GPU."""
    model = genai.GenerativeModel(GEMINI_MODEL)
    resp = model.generate_content(prompt)
    return resp.text.strip()

def call_gemini_embedding(text: str) -> list[float]:
    """Tạo embedding vector từ Gemini (miễn phí)."""
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text
    )
    return result["embedding"]

# ====================== SESSION ======================
@st.cache_resource
def get_jira_session():
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {JIRA_TOKEN}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    })
    return session

# ====================== UI ======================
st.title("🔍 QCD Tool - Phân tích Ticket (Gemini AI)")

st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #e2e8f0; }
    .stTextArea textarea, .stTextInput input {
        background: #1a1f2e !important; color: #e2e8f0 !important;
        border: 1px solid #374151 !important; border-radius: 8px !important;
    }
    div[data-testid="stButton"] > button {
        background: #2563eb; color: white; border: none; border-radius: 8px; font-weight: 600;
    }
    .result-box {
        background: #1a1f2e; border: 1px solid #374151; border-radius: 8px;
        padding: 16px; margin: 8px 0;
    }
    .dup { border-left: 4px solid #ef4444; }
    .near { border-left: 4px solid #f59e0b; }
    .sim { border-left: 4px solid #3b82f6; }
    .new { border-left: 4px solid #22c55e; }
</style>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊 Phân tích Ticket", "🔎 Tra cứu Jira"])

# ─── TAB 2: TRA CỨU JIRA ───
with tab2:
    st.subheader("Tra cứu Jira Ticket")
    
    if not JIRA_TOKEN:
        st.warning("⚠️ Thiếu JIRA_API_TOKEN - chỉ dùng được AI phân tích text")
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            jql = st.text_area("JQL Query", "project = VF6 ORDER BY created DESC", height=80)
        with col2:
            st.write("")
            st.write("")
            if st.button("🔍 Tìm", use_container_width=True):
                with st.spinner("Đang tìm trên Jira..."):
                    try:
                        session = get_jira_session()
                        resp = session.get(
                            f"{JIRA_URL}/rest/api/2/search",
                            params={"jql": jql, "maxResults": 20, "fields": "key,summary,description,status"},
                            verify=False, timeout=15
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            st.success(f"✅ Tìm thấy {data.get('total', 0)} tickets")
                            rows = []
                            for issue in data.get("issues", []):
                                f = issue.get("fields", {})
                                rows.append({
                                    "Key": issue.get("key"),
                                    "Summary": f.get("summary", ""),
                                    "Status": f.get("status", {}).get("name", ""),
                                })
                            st.dataframe(pd.DataFrame(rows), use_container_width=True)
                            st.session_state["jira_results"] = data.get("issues", [])
                        else:
                            st.error(f"Lỗi {resp.status_code}: Token không hợp lệ hoặc hết hạn")
                    except Exception as e:
                        st.error(f"Lỗi kết nối Jira: {e}")

# ─── TAB 1: PHÂN TÍCH ───
with tab1:
    st.subheader("Phân tích tương đồng Ticket bằng Gemini AI")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("**Ticket mới (cần kiểm tra)**")
        new_summary = st.text_area("Summary", "Hệ thống cảnh báo ACC không hoạt động", height=80, key="new_sum")
        new_desc = st.text_area("Mô tả lỗi (Actual)", "Khi bật ACC trên đường cao tốc, xe không tăng tốc và báo lỗi trên màn hình", height=120, key="new_desc")
        new_expected = st.text_area("Kỳ vọng (Expected)", "ACC phải hoạt động, giữ khoảng cách với xe phía trước", height=80, key="new_exp")
    
    with col_right:
        st.markdown("**Ticket Master (từ Jira / nhập tay)**")
        master_key = st.text_input("Jira Key (nếu có)", "VF6-12345")
        master_summary = st.text_area("Summary", "ACC không bật được trên cao tốc", height=80, key="master_sum")
        master_desc = st.text_area("Mô tả lỗi (Actual)", "Khi kích hoạt ACC ở tốc độ >60km/h, hệ thống báo lỗi và không hoạt động", height=120, key="master_desc")
        master_expected = st.text_area("Kỳ vọng (Expected)", "ACC hoạt động bình thường ở mọi tốc độ", height=80, key="master_exp")
    
    if st.button("🚀 Phân tích tương đồng", type="primary", use_container_width=True):
        if not new_summary or not master_desc:
            st.warning("Vui lòng nhập ít nhất Summary mới và Mô tả lỗi Master")
        else:
            prompt = f"""You are a Senior Automotive QA expert for VinFast electric vehicles.
Compare the NEW TICKET against the MASTER TICKET from Jira database.

NEW TICKET:
- Summary: {new_summary}
- Actual Result (mô tả lỗi): {new_desc}
- Expected Result: {new_expected}

MASTER TICKET ({master_key}):
- Summary: {master_summary}
- Actual Result (mô tả lỗi): {master_desc}
- Expected Result: {master_expected}

PHASE 1 - SCOPE CHECK (Expected Result):
Compare the "expected" fields of both tickets:
- SAME SCOPE: both tickets expect the same system behavior/outcome
- DIFFERENT SCOPE: tickets have fundamentally different purposes

PHASE 2 - SIMILARITY SCORING (Actual Result - primary driver):
Score 0-100 based on keyword matching:
- +3 pts: SPECIFIC DTC code (P0xxx, U0xxx), exact signal value (0x7F), exact fault state
- +2 pts: component + symptom PAIR ("MHU freeze", "ACC not activate", "DMS false alarm")
- +1 pt: single generic term (VCU, BMS, DTC, warning, display)

CLASSIFICATION:
- 85-100: DUPLICATE (cùng lỗi)
- 70-84: NEAR_DUP (rất giống, cùng component)
- 50-69: SIMILAR (cùng nhóm lỗi)
- <50: NOT_RELATED (không liên quan)

Return ONLY valid JSON (no markdown, no code block):
{{
    "scope_match": true/false,
    "confidence_score": <0-100>,
    "classification": "DUPLICATE" | "NEAR_DUP" | "SIMILAR" | "NOT_RELATED",
    "reason": "<1-2 câu giải thích ngắn gọn bằng tiếng Việt>"
}}"""
            
            try:
                with st.spinner("⏳ Gemini AI đang phân tích..."):
                    result = call_gemini(prompt)
                
                # Parse JSON từ response
                result = result.strip()
                if result.startswith("```"):
                    result = result.replace("```json", "").replace("```", "").strip()
                
                data = json.loads(result)
                score = data.get("confidence_score", 0)
                cls = data.get("classification", "NOT_RELATED")
                scope = data.get("scope_match", False)
                reason = data.get("reason", "")
                
                # Hiển thị kết quả
                cls_color = {"DUPLICATE": "#ef4444", "NEAR_DUP": "#f59e0b", 
                             "SIMILAR": "#3b82f6", "NOT_RELATED": "#22c55e"}
                cls_class = {"DUPLICATE": "dup", "NEAR_DUP": "near", 
                             "SIMILAR": "sim", "NOT_RELATED": "new"}
                
                st.markdown("### 📊 Kết quả phân tích")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Điểm tương đồng", f"{score}/100")
                c2.metric("Phân loại", cls)
                c3.metric("Cùng Scope", "✅ Có" if scope else "❌ Không")
                
                st.markdown(f"""
                <div class="result-box {cls_class.get(cls, 'new')}">
                    <strong>Giải thích:</strong><br>
                    {reason}
                </div>
                """, unsafe_allow_html=True)
                
                # Gợi ý
                if cls == "DUPLICATE":
                    st.error(f"🔴 **Kết luận: Ticket này có vẻ TRÙNG ({score}%)** - Nên đánh dấu duplicate")
                elif cls == "NEAR_DUP":
                    st.warning(f"🟠 **Kết luận: Ticket RẤT GIỐNG ({score}%)** - Cần kiểm tra thêm")
                elif cls == "SIMILAR":
                    st.info(f"🔵 **Kết luận: Ticket TƯƠNG TỰ ({score}%)** - Cùng nhóm lỗi")
                else:
                    st.success(f"🟢 **Kết luận: Ticket MỚI ({score}%)** - Không trùng với ticket đã có")
                
            except json.JSONDecodeError:
                st.error(f"Lỗi parse JSON từ Gemini. Raw response:\n{result}")
            except Exception as e:
                st.error(f"Lỗi: {e}")