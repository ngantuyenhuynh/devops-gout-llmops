FROM python:3.11-slim
WORKDIR /app

# Copy file requirements và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- 2 DÒNG SINH TỬ ĐÂY RỒI ---
COPY data/ ./data/
COPY src/ ./src/

# Copy file chạy chính ra ngoài
COPY src/evaluation/main.py .

# Lệnh chạy dành cho Job chấm điểm
CMD ["python", "main.py"]
