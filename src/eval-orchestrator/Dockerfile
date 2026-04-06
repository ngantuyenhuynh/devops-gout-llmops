# Dùng ảnh Python siêu nhẹ
FROM python:3.10-slim

WORKDIR /app

# Cài đặt FastAPI và Uvicorn
RUN pip install fastapi uvicorn pydantic requests

# Copy mã nguồn vào container
COPY main.py .

# Mở cổng 8000
EXPOSE 8000

# Lệnh khởi chạy server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
