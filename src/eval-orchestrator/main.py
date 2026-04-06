from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI(title="Gout-LLM EvalOps Orchestrator")

class EvalRequest(BaseModel):
    question: str
    context: str = ""

# Nhờ K8s Service Discovery, chúng ta chỉ cần gọi tên service thay vì địa chỉ IP
OLLAMA_URL = "http://ollama-service:11434/api/generate"

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Eval-Orchestrator is running"}

@app.post("/evaluate")
def trigger_evaluation(req: EvalRequest):
    # Tạo gói tin gửi sang Ollama
    payload = {
        "model": "qwen2:1.5b",
        "prompt": f"Bạn là một trợ lý y tế chuyên về bệnh Gút. Hãy trả lời câu hỏi sau: {req.question}",
        "stream": False
    }

    try:
        # Bắn request sang Inference-Worker
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        
        return {
            "question": req.question,
            "answer": result.get("response", ""),
            "model_used": "qwen2:1.5b",
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}
