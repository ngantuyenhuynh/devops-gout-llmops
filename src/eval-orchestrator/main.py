from fastapi import FastAPI
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import requests

app = FastAPI(title="Gout-LLM RAG API")

# Cấu hình đường dẫn nội bộ K8s
QDRANT_URL = "http://qdrant-service:6333"
OLLAMA_URL = "http://ollama-service:11434/api/generate"
COLLECTION_NAME = "gout_knowledge_base"

# 1. Khởi tạo mô hình Embedding (Bắt buộc phải giống hệt lúc nạp dữ liệu)
print("Đang tải model Embedding...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
client = QdrantClient(url=QDRANT_URL)
vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_gout_bot(req: QuestionRequest):
    # 2. RETRIEVAL: Lục lọi Vector DB để tìm top 3 đoạn tài liệu sát nghĩa nhất
    try:
        docs = vector_store.similarity_search(req.question, k=3)
        context = "\n---\n".join([doc.page_content for doc in docs])
        # Lấy tên file nguồn (PDF hoặc JSON) để làm bằng chứng
        sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))
    except Exception as e:
        return {"error": f"Lỗi khi kết nối Qdrant: {str(e)}"}

    # 3. AUGMENTED: Gắn tài liệu vào Prompt để "ép" AI trả lời chuẩn
    prompt = f"""Bạn là một bác sĩ chuyên khoa về bệnh Gút (Gout). 
Nhiệm vụ của bạn là trả lời câu hỏi của bệnh nhân DỰA VÀO TÀI LIỆU Y KHOA được cung cấp dưới đây.
Tuyệt đối không được bịa đặt. Nếu tài liệu không có thông tin, hãy trả lời: "Dựa theo phác đồ hiện tại, tôi không tìm thấy thông tin về vấn đề này."

TÀI LIỆU Y KHOA:
{context}

CÂU HỎI: {req.question}
TRẢ LỜI:"""

    # 4. GENERATION: Gửi sang Ollama (Đảm bảo bạn nhập đúng tên model đang chạy trong Ollama nhé)
    payload = {
        "model": "qwen2:1.5b",  # HOẶC vistral, phogpt... tùy model bạn đã pull
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response_data = response.json()
        
        return {
            "question": req.question,
            "answer": response_data.get("response", "Lỗi sinh text từ LLM"),
            "sources": sources,
            "context_used": context # In ra để debug xem nó tìm đúng bài không
        }
    except Exception as e:
        return {"error": f"Lỗi khi gọi Ollama: {str(e)}"}
