# So sánh các giải pháp AI cho QCD Tool

## 🎯 Mục tiêu
Chạy AI phân tích ticket Jira (tìm duplicate, so sánh lỗi) mà **không bị firewall công ty chặn**, **đảm bảo bảo mật dữ liệu**.

---

## 1. 🏠 Giải pháp A: Self-host Ollama ở nhà, gọi từ công ty

```
🏠 NHÀ ANH (GPU)                  Internet              🏢 CÔNG TY
┌───────────────┐              ┌─────────────────────────────────┐
│ Ollama Server  │──VPN/Tunnel──│ Streamlit App                  │
│ qwen2.5:14b    │   mã hóa     │ + Jira + Qdrant               │
│ nomic-embed-text│              │ (gọi API qua tunnel)          │
│ Port:11434     │              │                                 │
└───────────────┘              └─────────────────────────────────┘
```

### ✅ Điểm mạnh

| Ưu điểm | Mô tả |
|----------|-------|
| **🔒 Bảo mật tuyệt đối** | Dữ liệu Jira không qua server trung gian nào |
| **💯 Làm chủ hoàn toàn** | Model AI, dữ liệu đều do anh quản lý |
| **🔄 Dùng không giới hạn** | Không giới hạn request, không tốn phí API |
| **⚡ Tốc độ ổn định** | Chạy trên máy nhà riêng, không phụ thuộc bên thứ 3 |
| **🎨 Tùy chỉnh model** | Có thể swap model AI bất kỳ lúc nào |
| **🚫 Không sợ chặn** | Chỉ cần internet ra được, không cần vào các domain lạ |

### ❌ Điểm yếu

| Nhược điểm | Mô tả |
|------------|-------|
| **💰 Cần GPU mạnh** | qwen2.5:14b cần ~16GB VRAM, card rời tầm 15-20 triệu |
| **🔧 Setup phức tạp** | Cần cài Ollama, pull model, cấu hình VPN/tunnel |
| **🌐 Cần kỹ thuật mạng** | Phải biết mở port, dùng Cloudflare Tunnel hoặc VPN |
| **📡 Phụ thuộc internet** | Nếu mạng nhà yếu thì công ty không dùng được |
| **⚡ Upload speed cần thiết** | Cần ít nhất 5Mbps upload để gọi API mượt |
| **💡 Tốn điện 24/7** | Máy nhà phải bật liên tục (nếu muốn dùng mọi lúc) |

---

## 2. ☁️ Giải pháp B: Dùng Gemini API (Google - miễn phí)

```
🏢 CÔNG TY
┌─────────────────────────────┐     ┌──────────────┐
│ Streamlit App               │────▶│ Google Gemini │
│ + Jira + Qdrant            │     │ API (Cloud)  │
│ (gọi Gemini API trực tiếp)  │     │ (free tier)  │
└─────────────────────────────┘     └──────────────┘
```

### ✅ Điểm mạnh

| Ưu điểm | Mô tả |
|----------|-------|
| **🆓 Miễn phí** | Gemini free tier: 60 requests/phút, đủ dùng |
| **⚡ Cài đặt cực nhanh** | Chỉ cần lấy API key, không cần cài gì thêm |
| **💻 Không cần GPU** | Google xử lý hết, máy công ty chạy nhẹ |
| **🌐 Không cần cấu hình mạng** | Chỉ cần máy có internet là chạy được |
| **🔧 Zero maintenance** | Không cần bảo trì, update, restart |
| **🚀 Model mạnh** | Gemini 2.0 Flash ngang ngửa GPT-4o |
| **📈 Scale tự động** | Nếu cần nhiều hơn thì trả tiền, không lo quá tải |

### ❌ Điểm yếu

| Nhược điểm | Mô tả |
|------------|-------|
| **🔓 Bảo mật kém hơn** | Dữ liệu Jira đi qua Google server |
| **📊 Giới hạn free tier** | 60 req/phút, 1500 req/ngày |
| **🌐 Phụ thuộc Google** | Nếu Google chặn VN hoặc lỗi service thì không dùng được |
| **🔌 Cần internet** | Mất mạng là app không chạy |
| **📝 Không tùy chỉnh được** | Dùng đúng model Google, không swap được |
| **💰 Tốn phí nếu dùng nhiều** | Trên 1500 req/ngày thì mất tiền |

---

## 3. 🏢 Giải pháp C: vLLM host của công ty (nếu có sẵn)

```
🏢 SERVER CÔNG TY (GPU)              🏢 MÁY ANH
┌───────────────┐              ┌──────────────────┐
│ vLLM Server   │──Mạng nội bộ──│ Streamlit App   │
│ (GPU mạnh)    │   nhanh,      │ + Jira + Qdrant │
│ qwen2.5:14b   │   bảo mật     │                  │
└───────────────┘              └──────────────────┘
```

### ✅ Điểm mạnh
| Ưu điểm | Mô tả |
|----------|-------|
| Tốc độ cực nhanh | Mạng nội bộ, latency thấp |
| Bảo mật tuyệt đối | Dữ liệu không ra khỏi công ty |
| Không cần GPU cá nhân | Dùng chung server GPU của công ty |
| Không lo internet | Mạng nội bộ, không sợ chặn |

### ❌ Điểm yếu
| Nhược điểm | Mô tả |
|------------|-------|
| Cần có sẵn vLLM host | Phải IT công ty set up |
| Chia sẻ tài nguyên | Có thể bị tranh GPU với người khác |
| Phụ thuộc IT | Phải nhờ IT cài đặt, cấu hình |

---

## 📊 So sánh tổng quan

| Tiêu chí | A: Ollama ở nhà | B: Gemini API | C: vLLM công ty |
|----------|:---------------:|:-------------:|:---------------:|
| **Chi phí ban đầu** | 15-20tr (GPU) | 0đ | 0đ |
| **Chi phí vận hành** | Tiền điện ~200k/tháng | 0đ (free) | 0đ |
| **Bảo mật** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Dễ cài đặt** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| **Tốc độ** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Không bị chặn** | ✅ | ✅ | ✅ |
| **Không giới hạn** | ✅ | ❌ (free tier) | ✅ |

---

## 🎯 Kết luận cho anh

| Nếu anh... | Chọn giải pháp |
|-----------|---------------|
| **Có máy GPU mạnh ở nhà + rành kỹ thuật** | **A: Ollama self-host** ✅ |
| **Không có GPU, muốn chạy ngay lập tức** | **B: Gemini API** ✅ |
| **Công ty có sẵn vLLM server** | **C: vLLM host** ✅ |

### Lời khuyên của em
Anh nên **bắt đầu với Gemini API (B)** trước - mất 5 phút lấy key là chạy được ngay.

Sau đó nếu có thời gian và GPU ở nhà, hãy chuyển sang **Ollama self-host (A)** để bảo mật hơn.

---
*Happy to help!*