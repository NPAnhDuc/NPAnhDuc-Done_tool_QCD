# Hướng dẫn Self-host Ollama ở nhà, dùng từ công ty

## 🎯 Mục đích
- Cài **Ollama** + **Qwen2.5** trên máy tính ở nhà (có GPU)
- Host lên internet qua **Cloudflare Tunnel** (miễn phí, bảo mật)
- Máy công ty gọi API qua tunnel để dùng AI

---

## 📋 Yêu cầu

### Ở nhà (cần có)
| Thứ | Yêu cầu tối thiểu | Khuyên dùng |
|-----|-------------------|-------------|
| **GPU** | 8GB VRAM | 16GB+ VRAM (RTX 4060 Ti trở lên) |
| **RAM** | 16GB | 32GB |
| **OS** | Windows 11 / Linux | Linux (Ubuntu 22.04) |
| **Internet** | 5Mbps upload | 20Mbps+ upload |
| **Router** | Hỗ trợ UPnP hoặc port forwarding | Có thể cấu hình NAT |

---

## 📦 Bước 1: Cài đặt Ollama

### Trên Windows
```bash
# Cách 1: Download installer
# Vào https://ollama.com/download → tải OllamaSetup.exe → cài đặt

# Cách 2: Dùng winget (nếu có)
winget install Ollama.Ollama
```

### Trên Linux (Ubuntu/Debian)
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Kiểm tra cài đặt thành công
```bash
ollama --version
# Kết quả: ollama version 0.5.0 (hoặc mới hơn)
```

---

## 🤖 Bước 2: Pull model AI

```bash
# Model chính (LLM - sinh văn bản, phân tích ticket)
# Dung lượng ~9GB
ollama pull qwen2.5:14b

# Model embedding (tạo vector cho tìm kiếm)
# Dung lượng ~0.5GB
ollama pull nomic-embed-text

# Kiểm tra model đã pull
ollama list
# Kết quả:
# qwen2.5:14b        ...    9.0 GB
# nomic-embed-text   ...    0.5 GB
```

### Test thử model hoạt động
```bash
# Test LLM
ollama run qwen2.5:14b "Hello, are you working?"
# → Model trả lời

# Test embedding (gọi API)
curl http://localhost:11434/api/embeddings -d '{
  "model": "nomic-embed-text",
  "prompt": "ACC fault error"
}'
# → Trả về vector 768 số
```

---

## 🔌 Bước 3: Cho phép gọi API từ xa

Mặc định Ollama **chỉ chạy local** (127.0.0.1). Cần cho phép gọi từ mạng ngoài.

### Trên Windows:
1. Click chuột phải **Ollama icon** ở system tray (gần đồng hồ)
2ọn **Settings**
3. Bật **"Allow connections from any origin"**
4. Restart Ollama

### Hoặc dùng lệnh:
```bash
# Set environment variable để Ollama listen trên tất cả IP
# Trên Windows (PowerShell Admin):
$env:OLLAMA_HOST = "0.0.0.0"
[Environment]::SetEnvironmentVariable("OLLAMA_HOST", "0.0.0.0", "Machine")
# Restart máy hoặc restart Ollama service
```

### Kiểm tra:
```bash
# Lúc này Ollama listen ở port 11434 trên tất cả IP
curl http://localhost:11434/api/tags
# → List models
```

---

## 🌐 Bước 4: Mở kết nối an toàn ra internet

### Cách A: Cloudflare Tunnel (MIỄN PHÍ, KHUYẾN KHÍCH)

Cloudflare Tunnel là cách **an toàn nhất** vì:
- ✅ Không cần mở port trên router
- ✅ Mã hóa toàn bộ đường truyền (TLS)
- ✅ Có xác thực
- ✅ Bảo vệ khỏi DDoS, scan
- ✅ Miễn phí

#### Cài cloudflared:
```bash
# Trên Windows: Download từ
# https://github.com/cloudflare/cloudflared/releases
# → cloudflared-windows-amd64.exe → đổi tên thành cloudflared.exe
# → để ở C:\cloudflared\cloudflared.exe

# Trên Linux:
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared
chmod +x /usr/local/bin/cloudflared
```

#### Chạy Tunnel:
```bash
# Mở 1 terminal riêng và chạy:
cloudflared tunnel --url http://localhost:11434

# Kết quả:
# 2025/01/01 10:00:00 INF + https://random-name.trycloudflare.com
# → Copy URL này (dạng https://xxx.trycloudflare.com)
```

**Giữ terminal này chạy**, khi nào tắt là mất kết nối.

### Cách B: Dùng VPN (nếu có)
Nếu anh đã có VPN (WireGuard/OpenVPN):
- Cài VPN server ở nhà
- Máy công ty kết nối VPN vào
- Gọi API qua IP VPN: `http://192.168.x.x:11434`

---

## 🖥️ Bước 5: Cấu hình project ở công ty

### Sửa file `.env` trên máy công ty:
```env
# THAY URL này bằng URL từ Cloudflare Tunnel ở Bước 4
OLLAMA_BASE_URL=https://random-name.trycloudflare.com
OLLAMA_LLM_MODEL=qwen2.5:14b
OLLAMA_EMBED_MODEL=nomic-embed-text

# Jira config giữ nguyên
JIRA_URL=https://tms.vinfast.vn
JIRA_API_TOKEN=your_token
```

### Chạy app:
```bash
# Không cần Docker, chạy trực tiếp
streamlit run appVTDB_v3_local.py
```

---

## 🧪 Bước 6: Kiểm tra kết nối

### Test từ máy công ty:
```bash
# Gọi thử API Ollama qua tunnel
curl https://random-name.trycloudflare.com/api/tags

# Nếu trả về danh sách model → thành công!
# Nếu lỗi → kiểm tra tunnel còn chạy không
```

### Test Embedding từ xa:
```bash
curl https://random-name.trycloudflare.com/api/embeddings -d '{
  "model": "nomic-embed-text",
  "prompt": "test"
}'
```

---

## ⚠️ Lưu ý quan trọng

### 1. Bảo mật
```yaml
⚠️ KHÔNG bao giờ:
  - Share URL tunnel công khai
  - Để mật khẩu/token trong code
  - Dùng mà không có xác thực thêm

✅ NÊN:
  - Tắt tunnel khi không dùng (Ctrl+C)
  - Thay đổi URL tunnel mỗi lần chạy
  - Giới hạn IP nếu có thể
```

### 2. Hiệu suất
```yaml
⏱️ Latency dự kiến:
  - Mạng nhà xài ổn: 50-100ms (dùng thoải mái)
  - Mạng nhà yếu: 200-500ms (hơi chậm)

💡 Mẹo tăng tốc:
  - Dùng model nhỏ hơn: qwen2.5:7b (4GB) thay vì 14b
  - Bật GPU acceleration trong Ollama
  - Dùng dây mạng thay vì WiFi ở nhà
```

### 3. Chi phí
```
💰 Chi phí hàng tháng:
  - Cloudflare Tunnel: 0đ (miễn phí)
  - Điện máy chạy 24/7: ~200.000đ - 400.000đ
  - Model AI: 0đ (open source)

💵 Tổng: ~200.000đ - 400.000đ/tháng
```

---

## 🔧 Troubleshooting

### Lỗi: "Connection refused"
```
Nguyên nhân: Ollama chưa chạy hoặc chưa cho phép từ xa
Cách fix:
  1. Kiểm tra Ollama đang chạy: ollama --version
  2. Kiểm tra OLLAMA_HOST đã set chưa
  3. Restart Ollama
```

### Lỗi: "Tunnel not found"
```
Nguyên nhân: Cloudflare tunnel bị tắt
Cách fix:
  - Mở lại terminal cloudflared
  - Lấy URL mới
  - Cập nhật .env
```

### Lỗi: "Timeout"
```
Nguyên nhân: Mạng nhà yếu hoặc đường truyền chậm
Cách fix:
  - Kiểm tra upload speed ở nhà
  - Thử dùng model nhỏ hơn
  - Tăng timeout trong code
```

### Lỗi: "Model not found"
```
Nguyên nhân: Chưa pull model trên máy nhà
Cách fix:
  - ollama pull qwen2.5:14b
  - ollama pull nomic-embed-text
  - ollama list (để kiểm tra)
```

---

## 📝 Tóm tắt nhanh

```bash
# ===== Ở NHÀ =====

# 1. Cài Ollama
winget install Ollama.Ollama

# 2. Pull model
ollama pull qwen2.5:14b
ollama pull nomic-embed-text

# 3. Cho phép từ xa (set env: OLLAMA_HOST=0.0.0.0)
# Restart Ollama

# 4. Chạy Cloudflare Tunnel
cloudflared tunnel --url http://localhost:11434
# → Copy URL (https://xxx.trycloudflare.com)


# ===== Ở CÔNG TY =====

# 5. Sửa .env
OLLAMA_BASE_URL=https://xxx.trycloudflare.com

# 6. Chạy app
streamlit run appVTDB_v3_local.py
```

---
*Happy coding!*