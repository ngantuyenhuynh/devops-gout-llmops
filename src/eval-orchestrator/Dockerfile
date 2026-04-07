FROM python:3.10-slim
WORKDIR /app

# 1. Cài PyTorch CPU siêu nhẹ
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Cài các thư viện còn lại (Đã sửa lại đường dẫn COPY)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy toàn bộ code vào trong (Đã sửa lại đường dẫn COPY)
COPY . ./

# Lệnh chạy FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
