FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY src/evaluation-job/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY src/evaluation-job/main.py /app/main.py
COPY data /app/data

CMD ["python", "/app/main.py"]
