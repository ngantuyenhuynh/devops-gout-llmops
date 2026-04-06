FROM python:3.9-slim
WORKDIR /app

# Copy file requirements và cài đặt
COPY src/data-ingestion/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn VÀ thư mục data vào bên trong container
COPY src/data-ingestion/ ./
COPY data/ ./data/

CMD ["python", "ingest.py"]
