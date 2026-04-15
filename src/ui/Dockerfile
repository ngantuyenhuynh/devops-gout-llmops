FROM python:3.11-slim
WORKDIR /app

# Cài đặt trực tiếp Streamlit và các thư viện cần thiết (ví dụ: requests)
# Nếu app.py cần thêm thư viện khác (như pandas, qdrant-client...), hãy thêm vào dòng này
RUN pip install --no-cache-dir streamlit requests

# Copy file mã nguồn giao diện vào container
COPY app.py .

# Mở port mặc định của Streamlit
EXPOSE 8501

# Lệnh khởi chạy giao diện
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
