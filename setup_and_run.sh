#!/bin/bash
# ===================================================================
# VinFast QCD Tool — Local Stack Startup Script
# ===================================================================
# Script khởi động toàn bộ local stack:
#   Qdrant (vector DB) + Ollama (local AI) + Streamlit App
#
# Usage:
#   chmod +x setup_and_run.sh
#   ./setup_and_run.sh
#
# Yêu cầu:
#   - Docker & Docker Compose
#   - NVIDIA GPU + NVIDIA Container Toolkit (cho Ollama GPU)
# ===================================================================

set -e

echo "=============================================="
echo "  VinFast QCD Tool - Local Stack Setup"
echo "=============================================="
echo ""

# ──────────────────────────────────────────────
# 1. Kiểm tra prerequisites
# ──────────────────────────────────────────────
echo "🔍 Kiểm tra Docker..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker chưa được cài đặt. Hãy cài Docker trước."
    exit 1
fi
echo "   ✅ Docker OK"

if ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose chưa được cài đặt."
    exit 1
fi
echo "   ✅ Docker Compose OK"

# Kiểm tra NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "   ✅ NVIDIA GPU detected"
else
    echo "   ⚠️ Không tìm thấy nvidia-smi. Ollama sẽ chạy CPU-only (chậm hơn)."
fi

# ──────────────────────────────────────────────
# 2. Kiểm tra file .env
# ──────────────────────────────────────────────
echo ""
echo "🔍 Kiểm tra file .env..."
if [ ! -f ".env" ]; then
    echo "⚠️  File .env chưa tồn tại. Đang copy từ .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "   ✅ Đã tạo .env từ .env.example"
        echo "   ✏️  Hãy chỉnh sửa JIRA_API_TOKEN trong .env trước khi chạy."
        echo "   ✏️  Nếu chạy local (không Docker), set OLLAMA_BASE_URL=http://localhost:11434"
    else
        echo "❌ Không tìm thấy .env.example. Tạo file .env thủ công."
        exit 1
    fi
else
    echo "   ✅ File .env tồn tại"
fi

# ──────────────────────────────────────────────
# 3. Pull images & start services
# ──────────────────────────────────────────────
echo ""
echo "🐳 Pull Docker images..."
docker compose pull qdrant ollama

echo ""
echo "🚀 Khởi động Qdrant và Ollama..."
docker compose up -d qdrant ollama

# ──────────────────────────────────────────────
# 4. Chờ services ready
# ──────────────────────────────────────────────
echo ""
echo "⏳ Chờ Ollama và Qdrant khởi động..."

# Chờ Qdrant
echo "   - Qdrant..."
for i in $(seq 1 30); do
    if curl -s -f "http://localhost:6333/health" > /dev/null 2>&1; then
        echo "     ✅ Qdrant ready"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "     ❌ Qdrant không khởi động được. Kiểm tra logs: docker compose logs qdrant"
    fi
    sleep 2
done

# Chờ Ollama
echo "   - Ollama..."
for i in $(seq 1 30); do
    if curl -s -f "http://localhost:11434/api/tags" > /dev/null 2>&1; then
        echo "     ✅ Ollama ready"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "     ❌ Ollama không khởi động được. Kiểm tra logs: docker compose logs ollama"
    fi
    sleep 2
done

# ──────────────────────────────────────────────
# 5. Pull models vào Ollama
# ──────────────────────────────────────────────
echo ""
echo "📦 Pull models vào Ollama..."

echo "   - Pulling nomic-embed-text..."
docker compose exec -T ollama ollama pull nomic-embed-text 2>&1 | tail -1
echo "     ✅ nomic-embed-text"

echo "   - Pulling qwen2.5:14b (có thể mất vài phút)..."
docker compose exec -T ollama ollama pull qwen2.5:14b 2>&1 | tail -1
echo "     ✅ qwen2.5:14b"

# ──────────────────────────────────────────────
# 6. Build & start Streamlit App
# ──────────────────────────────────────────────
echo ""
echo "🏗️  Build Streamlit App..."
docker compose build app

echo ""
echo "🚀 Khởi động Streamlit App..."
docker compose up -d app

# ──────────────────────────────────────────────
# 7. Verify
# ──────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  ✅ ALL SERVICES STARTED SUCCESSFULLY"
echo "=============================================="
echo ""
echo "  🔗 Streamlit:  http://localhost:8501"
echo "  🔗 Qdrant UI:  http://localhost:6333/dashboard"
echo "  🔗 Ollama API: http://localhost:11434"
echo ""
echo "📋 Kiểm tra trạng thái:"
echo "   docker compose ps"
echo ""
echo "📋 Xem logs:"
echo "   docker compose logs -f app"
echo "   docker compose logs -f ollama"
echo ""
echo "📋 Dừng tất cả:"
echo "   docker compose down"
echo ""
echo "=============================================="