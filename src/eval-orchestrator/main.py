from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI(title="Gout-LLM RAG Orchestrator")

# Cấu hình địa chỉ nội bộ K8s
OLLAMA_URL = "http://ollama-service:11434/api/generate"
QDRANT_URL = "http://qdrant-service:6333/collections/gout_knowledge_base/points/search"

class EvalRequest(BaseModel):
    question: str

@app.post("/evaluate")
def r_a_g_evaluation(req: EvalRequest):
    # 1. Tìm kiếm kiến thức y khoa liên quan trong Qdrant
    # (Đoạn này giả lập logic search đơn giản, thực tế sẽ dùng langchain qdrant client)
    context = "Dựa trên Quyết định 361/QĐ-BYT: Bệnh nhân gút cần hạn chế hải sản, uống nhiều nước khoáng kiềm." 
    
    # 2. Xây dựng Prompt "ép" AI trả lời theo tài liệu
    prompt = f"""Bạn là chuyên gia y khoa về bệnh Gút. 
    Dựa vào tài liệu chuẩn sau đây: {context}
    Hãy trả lời câu hỏi: {req.question}
    Lưu ý: Nếu tài liệu không có thông tin, hãy nói 'Tôi không rõ', không được bịa đặt."""

    # 3. Gửi sang Ollama
    payload = {
        "model": "qwen2:1.5b",
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        return {
            "answer": response.json().get("response"),
            "source_referenced": "QĐ 361/BYT & Gold Dataset",
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e)}
