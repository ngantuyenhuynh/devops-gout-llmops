from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Gout-LLM EvalOps Orchestrator")

class EvalRequest(BaseModel):
    question: str
    context: str = "" # Dùng cho RAG sau này

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Eval-Orchestrator is running on GKE"}

@app.post("/evaluate")
def trigger_evaluation(req: EvalRequest):
    # TODO: Thêm logic gọi Vistral/PhoGPT và GPT-4 Judge tại đây
    return {
        "message": "Đã nhận câu hỏi y khoa",
        "question_received": req.question,
        "status": "pending"
    }
