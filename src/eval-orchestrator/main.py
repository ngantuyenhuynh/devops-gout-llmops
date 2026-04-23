from fastapi import FastAPI
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import requests

from prometheus_fastapi_instrumentator import Instrumentator
from langfuse.decorators import observe, langfuse_context

app = FastAPI(title="Gout-LLM RAG API")

# Kích hoạt Prometheus Metrics /metrics
Instrumentator().instrument(app).expose(app)

# Cấu hình đường dẫn nội bộ K8s
QDRANT_URL = "http://qdrant-service:6333"
OLLAMA_URL = "http://ollama-service:11434/api/generate"
COLLECTION_NAME = "gout_knowledge_base"

print("Đang tải model Embedding...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
client = QdrantClient(url=QDRANT_URL)
vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)

# 1. CẬP NHẬT Ở ĐÂY: Thêm model_name vào Request
class QuestionRequest(BaseModel):
    question: str
    model_name: str | None = None

# Langfuse theo dõi toàn bộ luồng
@observe(name="Gout_RAG_Pipeline")
@app.post("/ask")
def ask_gout_bot(req: QuestionRequest):
    # 2. CẬP NHẬT Ở ĐÂY: Gắn thêm tên model vào Langfuse để lên Dashboard dễ lọc
    langfuse_context.update_current_trace(
        input=req.question,
        metadata={"model_used": req.model_name} 
    )

    try:
        docs = vector_store.similarity_search(req.question, k=3)
        context = "\n---\n".join([doc.page_content for doc in docs])
        sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))
    except Exception as e:
        return {"error": f"Lỗi khi kết nối Qdrant: {str(e)}"}

    prompt = f"""Bạn là một bác sĩ chuyên khoa về bệnh Gút (Gout). 
Nhiệm vụ của bạn là trả lời câu hỏi của bệnh nhân DỰA VÀO TÀI LIỆU Y KHOA được cung cấp dưới đây.
Tuyệt đối không được bịa đặt. Nếu tài liệu không có thông tin, hãy trả lời: "Dựa theo phác đồ hiện tại, tôi không tìm thấy thông tin về vấn đề này."

TÀI LIỆU Y KHOA:
{context}

CÂU HỎI: {req.question}
TRẢ LỜI:"""

    # 3. CẬP NHẬT Ở ĐÂY: Truyền biến model_name vào payload của Ollama
    payload = {
        "model": req.model_name or "qwen2:1.5b",
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
        response_data = response.json()

        if response_data.get("error"):
            return {"error": f"Ollama error: {response_data['error']}"}

        answer = str(response_data.get("response", "")).strip()
        if not answer:
            return {"error": f"Ollama returned no response: {response_data}"}
        
        # Lưu kết quả trả về vào Langfuse Trace
        langfuse_context.update_current_trace(output=answer)
        
        return {
            "question": req.question,
            "answer": answer,
            "sources": sources,
            "context_used": context 
        }
    except Exception as e:
        return {"error": f"Lỗi khi gọi Ollama: {str(e)}"}
