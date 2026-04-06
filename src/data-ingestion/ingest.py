import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

# Cấu hình
DATA_FOLDER = "../../data/"
QDRANT_URL = "http://qdrant-service:6333"
COLLECTION_NAME = "gout_hybrid_knowledge"

def ingest_data():
    all_documents = []
    
    # 1. Khởi tạo mô hình Embedding (Dành cho tiếng Việt)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    print("--- Bắt đầu quét dữ liệu Gout (JSON & PDF) ---")
    
    for file in os.listdir(DATA_FOLDER):
        file_path = os.path.join(DATA_FOLDER, file)
        
        # XỬ LÝ FILE PDF
        if file.endswith(".pdf"):
            print(f"Đang đọc PDF: {file}")
            loader = PyPDFLoader(file_path)
            # Chia nhỏ PDF ngay khi đọc
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
            pdf_docs = loader.load_and_split(text_splitter)
            all_documents.extend(pdf_docs)

        # XỬ LÝ FILE JSON (Dữ liệu đã chunk sẵn)
        elif file.endswith(".json"):
            print(f"Đang đọc JSON chunk: {file}")
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                # Giả sử JSON là list các dict: [{"content": "..."}, ...]
                for item in json_data:
                    content = item.get("content") or item.get("text") or str(item)
                    # Giữ lại metadata nếu có để sau này AI trích dẫn nguồn
                    doc = Document(page_content=content, metadata={"source": file})
                    all_documents.append(doc)

    # 2. Đẩy toàn bộ vào Qdrant
    if all_documents:
        print(f"Tổng cộng có {len(all_documents)} đoạn dữ liệu. Đang nạp vào Qdrant...")
        QdrantVectorStore.from_documents(
            all_documents,
            embeddings,
            url=QDRANT_URL,
            collection_name=COLLECTION_NAME,
            force_recreate=True
        )
        print("--- Hoàn tất! Vector DB đã sẵn sàng ---")
    else:
        print("Không tìm thấy dữ liệu hợp lệ trong thư mục data!")

if __name__ == "__main__":
    ingest_data()
