FROM python:3.11-slim

WORKDIR /app

# Cài đặt các thư viện cần thiết
RUN pip install --no-cache-dir streamlit requests

# Copy code vào container
COPY app.py .

# Streamlit mặc định chạy cổng 8501
EXPOSE 8501

# Lệnh khởi chạy
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
