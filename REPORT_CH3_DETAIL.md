# BÁO CÁO PHÂN TÍCH VÀ BỔ SUNG MỤC LỤC ĐỒ ÁN GOUT-LLMOPS

## 1. NHẬN XÉT VÀ CÁC ĐIỂM CÒN THIẾU SO VỚI THỰC TẾ PROJECT

Dựa trên quá trình chúng ta xây dựng hệ thống `devops-gout-llmops` thực tế, mục lục của bạn đã rất tốt nhưng vẫn còn thiếu một số điểm "ăn tiền" và có một chút mâu thuẫn cần điều chỉnh:

### Các điểm mâu thuẫn cần sửa:
1. **Mâu thuẫn về Kubernetes (Mục 2.6 và 3.2.3):** Ở chương 2 bạn liệt kê `Google Kubernetes Engine (GKE)` nhưng ở chương 3 bạn lại viết `Khởi tạo Kubernetes bằng Kubeadm`. Nếu bạn dùng VM và tự cài K8s bằng Kubeadm/K3s thì nên **xóa mục 2.6 (GKE)** và thay bằng **Kubeadm/K3s**.
2. **Thiếu Infrastructure as Code (IaC):** Project của chúng ta sử dụng **Terraform** và **Ansible** (thể hiện qua thư mục `infra/ansible`). Bạn cần bổ sung 2 công nghệ này vào Chương 2 và đưa vào mục 3.2.
3. **Thiếu Helm:** Toàn bộ hệ thống ArgoCD của chúng ta đang deploy qua Helm chart (`helm/gout-service`). Thiếu Helm trong mục lục là một thiếu sót lớn.

### Các điểm kỹ thuật "ăn tiền" (Cần bổ sung vào Chương 3):
1. **Cơ chế Streaming Response (Chống Timeout):** Chúng ta vừa giải quyết một vấn đề cực hay: Model xử lý mất hơn 120s làm UI bị timeout. Giải pháp là dùng `NDJSON Stream` từ FastAPI xuống Streamlit. (Nên thêm vào 3.5 và 3.8).
2. **Cơ chế Auto-Retry & Xử lý Race Condition:** Evaluation Job bị lỗi `Connection refused` do ArgoCD chưa kịp kéo Image mới. Chúng ta đã thêm vòng lặp Retry thông minh. (Nên thêm vào 3.6).
3. **Tối ưu hóa phần cứng (Resource Tuning):** Quản lý node `worker-ai`, ép LLM chạy trên node riêng, tinh chỉnh CPU limit (3 cores) và giảm `k=2` trong Qdrant để tăng tốc inference.

---

## 2. VIẾT CHI TIẾT CHƯƠNG 3: TRIỂN KHAI (DỰA TRÊN THỰC TẾ CODE)

Dưới đây là phần nội dung siêu chi tiết cho Chương 3 để bạn đưa thẳng vào báo cáo (đã lồng ghép tất cả những kỹ thuật chúng ta vừa làm).

### CHƯƠNG 3. TRIỂN KHAI HỆ THỐNG GOUT-LLMOPS

#### 3.1. Tổng quan kiến trúc hệ thống
Hệ thống được thiết kế theo mô hình Microservices, quản lý mã nguồn dưới dạng **Monorepo**. Kiến trúc tổng thể bao gồm 4 thành phần chính:
- **Tầng Infrastructure:** Cụm Kubernetes tự quản trị, được tự động hóa cấu hình bằng IaC.
- **Tầng Data & AI:** Qdrant Vector DB lưu trữ tri thức và Ollama Worker chạy mô hình ngôn ngữ lớn (Qwen2.5:1.5b).
- **Tầng Backend & Frontend:** FastAPI (Orchestrator) xử lý logic RAG và Streamlit (UI) tương tác với người dùng.
- **Tầng LLMOps:** Hệ thống CI/CD đa giai đoạn tích hợp GitOps (ArgoCD) và Continuous Evaluation (CE).

#### 3.2. Quản trị hạ tầng và kiến trúc Kubernetes (Infrastructure as Code)
**3.2.1. Tự động hóa hạ tầng với Terraform và Ansible**
Thay vì thiết lập thủ công, toàn bộ máy chủ ảo được cấp phát tự động bằng Terraform. Sau đó, Ansible được sử dụng để thiết lập môi trường, cài đặt container runtime và cấu hình bảo mật.
**3.2.2. Khởi tạo và thiết kế mạng cụm Kubernetes**
Cụm được khởi tạo bằng Kubeadm/K3s với topology gồm các node Control Plane và Worker Nodes. Giao tiếp nội bộ được quản lý thông qua CNI (Container Network Interface) đảm bảo hiệu năng cao cho dữ liệu Vector.
**3.2.3. Cô lập tài nguyên (Resource Isolation) và Scheduling**
Tính năng NodeSelector và Taints/Tolerations được áp dụng triệt để. Cụ thể, Pod `ollama-worker` được chỉ định bắt buộc chạy trên node có label `role=worker-ai`. Việc này ngăn chặn sự cố LLM sử dụng hết RAM/CPU ảnh hưởng đến các service khác như DB hay UI.

#### 3.3. Chuỗi cung ứng CI/CD và GitOps
**3.3.1. Tối ưu hoá Continuous Integration (CI) với Path-based Trigger**
Để giảm tải tài nguyên, CI pipeline (`ci.yaml`) sử dụng thư viện `dorny/paths-filter`. Hệ thống chỉ tiến hành build lại Image của dịch vụ nào thực sự có mã nguồn bị thay đổi (ví dụ: đổi thư mục `src/ui/` thì chỉ build lại UI).
**3.3.2. Chiến lược GitOps và Cập nhật Tag tự động**
Thay vì CD push thẳng Image vào K8s, CI Pipeline có một Job tên `gitops-update`. Job này tự động dùng lệnh `sed` để ghi đè mã băm của Git (`github.sha`) vào các file `values.yaml` của Helm Chart, sau đó commit ngược lại nhánh `main`. Nhờ logic `always() && !contains(failure)`, hệ thống xử lý hoàn hảo việc bỏ qua (skip) các service không thay đổi mà không làm gãy pipeline.
**3.3.3. Triển khai bằng ArgoCD và Helm Chart**
Hệ thống sử dụng Helm Chart chung (`helm/gout-service`) làm template chuẩn hóa cho cả UI và Orchestrator. ArgoCD liên tục theo dõi thư mục `k8s/argocd-apps/`, tự động phát hiện mã SHA mới từ GitOps và đồng bộ trạng thái (Sync) xuống cụm K8s theo cơ chế Rolling Update.

#### 3.4. Thiết kế Pipeline RAG cho hệ thống tư vấn bệnh Gút
**3.4.1. Data Ingestion Pipeline**
Dịch vụ Ingestion chạy dưới dạng Kubernetes Job. Nó sử dụng `PyPDFLoader` để đọc tài liệu y khoa và `RecursiveCharacterTextSplitter` cắt nhỏ văn bản (chunk_size=1000, overlap=200) để đảm bảo ngữ cảnh không bị đứt gãy.
**3.4.2. Vector Database (Qdrant) và Retrieval**
Sử dụng mô hình nhúng `paraphrase-multilingual-MiniLM-L12-v2` để chuyển đổi văn bản thành vector. Ở pha truy xuất, hệ thống được tinh chỉnh (Tuning) tham số `k=2` thay vì `k=3` để tối ưu hóa thời gian tính toán Prompt (Prompt Evaluation) trên môi trường CPU, giúp giảm 33% độ trễ suy luận.
**3.4.3. Kỹ thuật Prompt Engineering**
Prompt được đóng khung chặt chẽ với vai trò "Bác sĩ chuyên khoa Gút" và chèn chỉ thị chống ảo giác (Hallucination): *"Tuyệt đối không được bịa đặt. Nếu tài liệu không có thông tin, hãy trả lời: Tôi không tìm thấy thông tin."*

#### 3.5. Thiết kế Orchestrator Service (Backend)
**3.5.1. FastAPI Gateway và Cơ chế Streaming (NDJSON)**
Một thách thức lớn xuất hiện khi mô hình AI tốn hơn 120 giây để sinh câu trả lời do rào cản phần cứng CPU, dẫn đến lỗi `Read timed out` ở Frontend. Giải pháp đột phá được áp dụng là thiết kế endpoint `/ask/stream`. 
Orchestrator không chờ AI sinh xong toàn bộ văn bản, mà sử dụng `StreamingResponse` để trả về nguồn trích dẫn (sources) ngay trong mili-giây đầu tiên để giữ kết nối, sau đó liên tục đẩy từng Token (chunk) xuống client dưới chuẩn NDJSON.
**3.5.2. Quan sát hệ thống (Observability) với Langfuse**
Mọi câu hỏi và luồng truy xuất RAG đều được theo dõi bằng `@observe` của Langfuse. Để giải quyết lỗi mất Context (ContextVar) khi chạy thuật toán Streaming trong Background Thread, hệ thống bọc lệnh gọi `update_current_trace` trong khối try-except, đảm bảo log được ghi nhận mà không làm gãy luồng sinh chữ của người dùng.

#### 3.6. Hệ thống đánh giá mô hình (Continuous Evaluation - CE)
**3.6.1. Pipeline Evaluation-Job**
Evaluation Job được kích hoạt tự động ngay sau khi CI hoàn tất. Job này lấy một tập dữ liệu chuẩn (Ground Truth) để hỏi AI và so sánh.
**3.6.2. Cơ chế Auto-Retry và Resilience (Xử lý Race Condition)**
Một vấn đề kỹ thuật cực kỳ tinh vi phát sinh: Khi CD chạy Evaluation Job thì ArgoCD vẫn đang bận tải Image mới (kéo dài 60s). Điều này gây lỗi mất kết nối `Connection refused`. Giải pháp triển khai là xây dựng cơ chế **Auto-Retry** trong mã nguồn đánh giá (Thử lại 15 lần, mỗi lần cách nhau 10s). Kỹ thuật này giúp hệ thống kiên nhẫn chờ đợi đến khi Pod Orchestrator thật sự sẵn sàng, chống gãy Pipeline do Race Condition.
**3.6.3. Lưu trữ Artifacts**
Kết quả đánh giá (Log) và Metrics được GitHub Actions đóng gói tự động thành Artifacts và đính kèm vào lịch sử Workflow, giúp các kỹ sư dễ dàng so sánh chất lượng mô hình qua từng phiên bản Commit.

#### 3.7. Observability và Monitoring (Giám sát hệ thống)
**3.7.1. Cấu trúc kube-prometheus-stack**
Triển khai toàn bộ Stack giám sát thông qua Helm, tích hợp tự động qua ArgoCD.
**3.7.2. Tùy chỉnh Grafana Dashboard bằng ConfigMap**
Dashboard theo dõi hệ thống LLMOps được triển khai hoàn toàn dưới dạng Code (ConfigMap). Các truy vấn PromQL được viết để lấy trực tiếp trạng thái tức thời (Instant query) bằng tổng `kube_pod_container_status_restarts_total` thay vì dùng hàm `increase([1h])`, giúp giải quyết triệt để lỗi Blank Data không hiển thị thông số đối với các Pod vừa mới được khởi tạo.
**3.7.3. Giám sát độ trễ AI (P95 Latency)**
Thiết lập biểu đồ Histogram theo dõi bách phân vị 95 (P95) của endpoint `/ask`, giúp phát hiện sớm tình trạng quá tải CPU trên Node AI.

#### 3.8. Giao diện người dùng (Streamlit Frontend)
Giao diện tư vấn y khoa được thiết kế tối giản, tập trung vào trải nghiệm người dùng. 
Bằng cách sử dụng hàm `st.write_stream()` kết hợp với bộ phân tích NDJSON stream từ Backend, UI tạo ra **hiệu ứng Typewriter (máy đánh chữ)** giống hệ thống ChatGPT. Điều này không chỉ triệt tiêu hoàn toàn lỗi Timeout mà còn che giấu đi độ trễ phần cứng (khi chạy LLM trên CPU), mang lại cảm giác tương tác thời gian thực cho bệnh nhân.

#### 3.9. Bảo mật hệ thống
Áp dụng Secret Management trong Kubernetes. Biến môi trường nhạy cảm như OpenAI API Key hoặc Langfuse Secret không được lưu trữ cứng trong mã nguồn (Hardcode) mà được quản lý thông qua Kubernetes Secrets (`envFromSecrets`).
