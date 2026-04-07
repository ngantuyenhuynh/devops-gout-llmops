FROM python:3.9-slim
WORKDIR /app

# 1. Cài đặt bản PyTorch siêu nhẹ dành cho CPU trước
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Sau đó mới cài các thư viện còn lại
COPY src/data-ingestion/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy code và dữ liệu
COPY src/data-ingestion/ ./
COPY data/ ./data/

CMD ["python", "ingest.py"]
