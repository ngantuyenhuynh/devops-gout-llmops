# devops-gout-llmops

1. Thư mục src/data-ingestion/ (Bộ Nạp Dữ Liệu - Khởi tạo "Trí nhớ")
Thư mục này chịu trách nhiệm cho chữ R (Retrieval) trong hệ thống RAG, cụ thể là bước chuẩn bị dữ liệu (Data Preparation).

ingest.py: Đây là kịch bản cốt lõi để biến tài liệu thô thành cơ sở dữ liệu véc-tơ. Các nhiệm vụ chính của file này:

Đọc file (Loaders): Dùng pypdf để đọc sách hướng dẫn y khoa và json để đọc các test case. Nó bao gồm cả đoạn code "mặc áo giáp" (try...except) để tự động bỏ qua các dòng bị lỗi cú pháp trong file JSONL mà không làm sập hệ thống.

Cắt nhỏ (Splitting): Chia văn bản dài thành các đoạn nhỏ (chunks) để AI dễ tiêu hóa hơn.

Nhúng (Embedding): Gọi mô hình paraphrase-multilingual... của HuggingFace để biến các đoạn chữ thành các dãy số (véc-tơ).

Lưu trữ (Storage): Kết nối qua cổng 6333 và bơm toàn bộ các véc-tơ này vào bộ nhớ của Qdrant (gout_knowledge_base).

Dockerfile: Bản thiết kế để đóng gói file ingest.py thành một máy ảo mini (Container). Điểm nhấn kỹ thuật ở đây là lệnh tải PyTorch CPU (--index-url https://download.pytorch.org/whl/cpu) giúp giảm dung lượng Image từ 8GB xuống còn hơn 1GB, tăng tốc độ triển khai.

requirements.txt: Danh sách "đồ nghề" (thư viện) bắt buộc phải cài đặt để code Python chạy được (langchain, qdrant-client, sentence-transformers...).

2. Thư mục src/eval-orchestrator/ (Bộ Điều Phối - Trái tim của hệ thống RAG)
Đây là API Backend giao tiếp trực tiếp với người dùng và liên kết các thành phần lại với nhau.

main.py: Chứa toàn bộ luồng logic RAG hoàn chỉnh được viết bằng FastAPI.

API Endpoint (/ask): Mở cửa nhận câu hỏi từ bệnh nhân/người dùng.

Retrieval (Truy xuất): Dùng lại mô hình Embedding của HuggingFace để biến câu hỏi thành véc-tơ, sau đó chạy sang Qdrant tìm 3 đoạn tài liệu y khoa sát nghĩa nhất. Nó cũng trích xuất tên file (sources) để làm bằng chứng chống "ảo giác" (hallucination).

Augmented (Tăng cường): Lấy 3 đoạn tài liệu tìm được, ghép chung với câu hỏi của người dùng và nhét vào một cái Prompt (khuôn mẫu lệnh) quy định vai trò "Bác sĩ chuyên khoa Gút".

Generation (Sinh văn bản): Bắn cái Prompt đã được nhồi nhét tài liệu đó sang máy chủ Ollama (chạy model qwen2:1.5b) để xin câu trả lời và trả về kết quả cuối cùng dưới dạng JSON.

Dockerfile: Tương tự như bộ nạp, file này cũng dùng kỹ thuật "ép cân" PyTorch CPU để đóng gói FastAPI. Lệnh cuối cùng là khởi chạy máy chủ web Uvicorn ở cổng 8000.

requirements.txt: Chứa các thư viện xây dựng API và gọi mô hình (fastapi, uvicorn, requests,...).

3. Thư mục k8s/ (Cấu hình Kubernetes - Quản lý Hạ tầng)
Khu vực này chứa các file .yaml đóng vai trò như "bản vẽ thi công" để Kubernetes biết cách vận hành các container.

k8s/data-ingestion/job.yaml: Khai báo một đối tượng loại Job. Đặc điểm của Job là chạy xong việc (nạp xong dữ liệu) thì nó báo Completed và nằm im, giải phóng tài nguyên CPU/RAM cho hệ thống. Nó không chạy liên tục.

k8s/eval-orchestrator/deployment.yaml: Khai báo đối tượng Deployment. Trái ngược với Job, Deployment yêu cầu ứng dụng (API FastAPI) phải chạy liên tục 24/7. File này chứa lệnh imagePullPolicy: Always để đảm bảo K8s luôn kéo code mới nhất, và chỉ định ứng dụng chạy ở cổng 8000.

k8s/eval-orchestrator/service.yaml: Khai báo đối tượng Service (Kiểu ClusterIP). Đây chính là "quầy lễ tân" nội bộ, cấp cho eval-orchestrator một địa chỉ IP tĩnh và định tuyến mọi luồng dữ liệu (traffic) vào đúng cái Pod FastAPI cổng 8000.

4. Thư mục .github/workflows/ (CI/CD Pipeline)
ci.yaml: Kịch bản tự động hóa GitHub Actions.

Khi bạn đẩy code lên nhánh main, hệ thống sẽ kích hoạt.

Nó đăng nhập vào Google Cloud bằng Service Account, cd vào đúng các thư mục src/, đọc các file Dockerfile đã được tối ưu để build ra các Docker Image.

Cuối cùng, nó push các Image đó lên Google Artifact Registry một cách hoàn toàn tự động.

Toàn bộ các file này liên kết chặt chẽ với nhau: Code Python (src/) định nghĩa "nó làm gì", Docker đóng gói "nó cần gì để chạy", Kubernetes (k8s/) quy định "nó sống ở đâu, vận hành thế nào", và CI/CD đảm bảo quy trình "từ lúc viết code đến lúc lên mây" không cần con người can thiệp.
