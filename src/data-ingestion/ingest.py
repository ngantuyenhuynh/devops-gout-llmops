import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

# Cấu hình đường dẫn nội bộ K8s
DATA_FOLDER = "./data/"
QDRANT_URL = "http://qdrant-service:6333"
COLLECTION_NAME = "gout_knowledge_base"

def ingest_all():
    all_documents = []
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    print("--- 🚀 Bắt đầu quét kho dữ liệu Gout ---")
    
    for file in os.listdir(DATA_FOLDER):
        file_path = os.path.join(DATA_FOLDER, file)
        
        # 1. Xử lý PDF (Guideline 1 & 2)
        if file.endswith(".pdf"):
            print(f"📄 Đang đọc PDF: {file}")
            loader = PyPDFLoader(file_path)
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            all_documents.extend(loader.load_and_split(splitter))

        # 2. Xử lý JSON (Test Cases)
        elif file.endswith(".json"):
            print(f"📊 Đang đọc JSON: {file}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    content = f"Q: {item.get('cau_hoi')}\nA: {item.get('ground_truth')}"
                    all_documents.append(Document(page_content=content, metadata={"source": file, "type": "qa"}))

        # 3. Xử lý JSONL (KB Chunks & Test Cases JSONL)
        elif file.endswith(".jsonl"):
            print(f"📦 Đang đọc JSONL: {file}")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip() # Dọn dẹp khoảng trắng và dấu xuống dòng thừa
                    if not line:        # Bỏ qua ngay nếu là dòng trống
                        continue
                    
                    try:
                        item = json.loads(line)
                        # Tùy biến theo cấu trúc JSONL của bạn
                        text = item.get("content") or item.get("text") or str(item)
                        all_documents.append(Document(page_content=text, metadata={"source": file}))
                    except json.JSONDecodeError as e:
                        # Báo lỗi nhẹ nhàng và chạy tiếp dòng khác, không làm sập cả chương trình
                        print(f"⚠️ Bỏ qua 1 dòng lỗi trong {file}: {e}")

    # 4. Đẩy lên Qdrant
    if all_documents:
        print(f"✅ Tổng cộng có {len(all_documents)} đoạn tri thức. Đang nạp vào Qdrant...")
        QdrantVectorStore.from_documents(
            all_documents, embeddings, url=QDRANT_URL, 
            collection_name=COLLECTION_NAME, force_recreate=True
        )
        print("--- 🏁 Hoàn tất nạp dữ liệu! ---")

if __name__ == "__main__":
    ingest_all()
