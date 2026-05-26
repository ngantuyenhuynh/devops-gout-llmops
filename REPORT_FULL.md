# BÁO CÁO ĐỒ ÁN: XÂY DỰNG HỆ THỐNG LLMOPS CHO ỨNG DỤNG TƯ VẤN BỆNH GÚT (RAG)

## CHƯƠNG 1. GIỚI THIỆU TỔNG QUAN

### 1.1. Mục tiêu của đề tài

Mục tiêu cốt lõi của đề tài là xây dựng một hệ thống LLMOps (Large Language Model Operations) hoàn chỉnh từ con số không, áp dụng các tiêu chuẩn công nghiệp nhằm tự động hóa quy trình phát triển, triển khai và đánh giá liên tục (Continuous Evaluation) cho một trợ lý AI tư vấn y khoa chuyên biệt về bệnh Gút (Gout).

Cụ thể, đề tài hướng đến việc:

- **Tự động hóa hoàn toàn chuỗi cung ứng phần mềm (CI/CD/CE):** Không có bất kỳ sự can thiệp thủ công nào trong việc đưa code từ môi trường phát triển lên môi trường Production.
- **Xây dựng kiến trúc Microservices trên Kubernetes:** Tách bạch rõ ràng giữa Giao diện (UI), Xử lý logic (Orchestrator), Cơ sở dữ liệu vector (Qdrant), và Inference Engine (Ollama).
- **Tích hợp Continuous Evaluation (Đánh giá liên tục):** Đảm bảo mỗi phiên bản AI được sinh ra đều phải trải qua bước kiểm định tự động (Smoke Test) dựa trên tập Ground Truth y khoa trước khi được xem là thành công.
- **Giám sát thời gian thực (Observability):** Thu thập và trực quan hóa toàn bộ metrics hệ thống, logs, và trace để dễ dàng phát hiện nút thắt cổ chai (bottleneck) và lỗi.

### 1.2. Tóm tắt nội dung thực hiện

Trong quá trình thực hiện, dự án đã hoàn thành các hạng mục kỹ thuật chuyên sâu:

1. **Thiết lập hạ tầng Kubeadm:** Tự xây dựng cụm Kubernetes thay vì dùng dịch vụ Managed Cloud (GKE/EKS) nhằm tối ưu chi phí và nắm vững nguyên lý hoạt động sâu nhất của K8s. Cấu hình Node Labeling để điều phối các tác vụ AI nặng (LLM Inference) vào đúng Node có tài nguyên phù hợp (`worker-ai`).
2. **Thiết kế luồng CI/CD thông minh:** Sử dụng GitHub Actions kết hợp `dorny/paths-filter` để chỉ build lại đúng vi dịch vụ có sự thay đổi mã nguồn, giúp giảm 80% thời gian chạy CI.
3. **Triển khai GitOps với ArgoCD:** Loại bỏ việc áp dụng Manifest thủ công (Push-based deployment). Thay vào đó, CI sẽ cập nhật GitOps tag (SHA) vào Helm Values, và ArgoCD sẽ tự động kéo (Pull) sự thay đổi này về cụm Kubernetes.
4. **Phát triển RAG Pipeline:** Ingest tài liệu y khoa định dạng PDF/JSONL vào Qdrant Vector Database, xây dựng FastAPI Orchestrator để tạo prompt engineering kết hợp với model `qwen2.5:1.5b` chạy trên Ollama.
5. **Giải quyết triệt để các bài toán kỹ thuật thực tế:** Khắc phục lỗi Timeout 120s bằng kỹ thuật Streaming Response (NDJSON), xử lý Race Condition giữa ArgoCD Sync và Evaluation Job bằng cơ chế Auto-Retry 150s, cấu hình PromQL để loại bỏ lỗi Blank Data trên Grafana.

### 1.3. Ý nghĩa khoa học và thực tiễn

#### 1.3.1. Ý nghĩa khoa học

Đề tài cung cấp một kiến trúc mẫu (Reference Architecture) toàn diện về việc tích hợp MLOps/LLMOps vào phát triển phần mềm hiện đại. Nó minh chứng cho việc lý thuyết RAG (Retrieval-Augmented Generation) khi kết hợp với hệ sinh thái Cloud-Native (Kubernetes, GitOps) có thể giải quyết được bài toán về "Ảo giác" (Hallucination) của LLM và khả năng triển khai liên tục không gián đoạn.

#### 1.3.2. Ý nghĩa thực tiễn

Đối với lĩnh vực y tế, một hệ thống AI tư vấn có độ chính xác cao dựa trên phác đồ điều trị thực tế (Ground Truth) sẽ hỗ trợ đắc lực cho bác sĩ và bệnh nhân. Đối với lĩnh vực kỹ thuật phần mềm, dự án là một cẩm nang sống động về cách vượt qua giới hạn phần cứng (chạy LLM trên Node 2vCPU - 4vCPU) bằng các kỹ thuật tinh chỉnh giới hạn tài nguyên (Resource Quotas) và Streaming.

### 1.4. Phạm vi nghiên cứu

- **Mô hình AI:** Quản lý và vận hành mô hình mã nguồn mở kích thước nhỏ gọn (Qwen 2.5 1.5B) bằng Ollama.
- **Hạ tầng:** Cụm Kubernetes tự host trên máy ảo (OpenStack/KVM).
- **Phạm vi dữ liệu:** Tập trung vào kiến thức, phác đồ, và chế độ dinh dưỡng cho người bệnh Gút.

---

## CHƯƠNG 2. CÁC CÔNG NGHỆ LIÊN QUAN

*(Lưu ý: Đã sửa lại số thứ tự lỗi ở mục lục cũ và điều chỉnh GKE thành Kubeadm để bám sát thực tế dự án).*

### 2.1. Công nghệ 1: Large Language Models (LLMs)

#### 2.1.1. Khái niệm

LLM là các mô hình trí tuệ nhân tạo được huấn luyện trên lượng dữ liệu văn bản khổng lồ (thường lên tới hàng nghìn tỷ tokens) bằng kiến trúc Transformer, có khả năng hiểu và sinh ngôn ngữ tự nhiên.

#### 2.1.2. Đặc điểm

- Sở hữu hàng tỷ đến hàng nghìn tỷ tham số (Parameters).
- Đòi hỏi tài nguyên phần cứng lớn (GPU, RAM cao) để huấn luyện (Training) và suy luận (Inference).

#### 2.1.3. Khả năng

Có khả năng tóm tắt, dịch thuật, lập luận, trả lời câu hỏi và sinh mã nguồn lập trình.

#### 2.1.4. Chức năng và nhiệm vụ

Đóng vai trò là "bộ não" tổng hợp và phân tích ngôn ngữ trong dự án, cụ thể là phân tích câu hỏi bệnh nhân và tổng hợp câu trả lời dựa trên tài liệu được cung cấp.

### 2.2. Công nghệ 2: Retrieval-Augmented Generation (RAG)

#### 2.2.1. Khái niệm

RAG là kỹ thuật tăng cường khả năng của LLM bằng cách cho phép mô hình "truy xuất" (Retrieve) thông tin từ một cơ sở dữ liệu tri thức bên ngoài trước khi "sinh" (Generate) câu trả lời.

#### 2.2.2. Đặc điểm

- Tách biệt rạch ròi giữa bộ nhớ dài hạn (Vector Database) và khả năng suy luận (LLM).
- Không yêu cầu Fine-tuning lại mô hình khi có dữ liệu mới.

#### 2.2.3. Khả năng

Loại bỏ tình trạng "ảo giác" (Hallucination) do LLM bịa đặt thông tin, giúp câu trả lời mang tính thời sự và chuyên ngành.

#### 2.2.4. Chức năng và nhiệm vụ

Chức năng cốt lõi giúp hệ thống AI bám sát phác đồ điều trị bệnh Gút thay vì trả lời kiến thức chung chung trên mạng.

### 2.3. Công nghệ 3: DevOps

#### 2.3.1. Khái niệm

DevOps là sự kết hợp giữa triết lý văn hóa, thực tiễn và công cụ nhằm tăng cường khả năng phân phối ứng dụng và dịch vụ ở tốc độ cao.

#### 2.3.2. Đặc điểm

- Phá bỏ bức tường ngăn cách giữa đội ngũ phát triển (Dev) và vận hành (Ops).

#### 2.3.3. Khả năng

Tự động hóa toàn bộ quy trình tích hợp và triển khai liên tục.

#### 2.3.4. Chức năng và nhiệm vụ

Là nguyên lý chỉ đạo cho việc xây dựng kiến trúc CI/CD/CE trong dự án.

### 2.4. Công nghệ 4: Docker

#### 2.4.1. Khái niệm

Nền tảng cung cấp công nghệ ảo hóa cấp hệ điều hành, cho phép đóng gói phần mềm thành các "Container".

#### 2.4.2. Đặc điểm

Nhẹ, nhanh, độc lập với môi trường Host.

#### 2.4.3. Khả năng

Đảm bảo phần mềm chạy ổn định ở mọi môi trường (Máy dev, Test, Production).

#### 2.4.4. Chức năng và nhiệm vụ

Đóng gói Orchestrator, UI, Evaluation Job và Ingestion Job thành các Artifacts tiêu chuẩn.

### 2.5. Công nghệ 5: Kubernetes (K8s)

#### 2.5.1. Khái niệm

Hệ thống mã nguồn mở tự động hóa việc triển khai, mở rộng và quản lý các ứng dụng container hóa.

#### 2.5.2. Đặc điểm

Sử dụng kiến trúc Master-Worker, quản lý tài nguyên dưới dạng Declarative (khai báo).

#### 2.5.3. Khả năng

Cung cấp High Availability, Self-healing, Load Balancing.

#### 2.5.4. Chức năng và nhiệm vụ

Là hệ điều hành nền tảng cho toàn bộ hệ thống LLMOps.

### 2.6. Công nghệ 6: Kubeadm (Self-hosted K8s)

#### 2.6.1. Khái niệm

Là công cụ do chính Kubernetes cung cấp để khởi tạo cụm (cluster) K8s chuẩn mực trên nền tảng máy ảo IaaS (Infrastructure as a Service) như OpenStack hoặc Bare-metal.

#### 2.6.2. Đặc điểm

Cung cấp quyền kiểm soát sâu nhất (Root level) tới Control Plane, API Server và Etcd.

#### 2.6.3. Khả năng

Xây dựng cụm K8s với chi phí tối ưu, không phụ thuộc vào vendor-lock của các Cloud Provider.

#### 2.6.4. Chức năng và nhiệm vụ

Công cụ chính để Bootstrap hệ thống hạ tầng 3 Node (VM1, VM2, VM3) cho dự án.

### 2.7. Công nghệ 7: Google Artifact Registry (GAR)

#### 2.7.1. Khái niệm

Dịch vụ lưu trữ Artifacts và Container Images do Google Cloud cung cấp.

#### 2.7.2. Đặc điểm

Bảo mật cao, tích hợp chặt chẽ với IAM và có khả năng quét lỗ hổng bảo mật tự động.

#### 2.7.3. Khả năng

Cung cấp đường truyền tốc độ cao để pull images về môi trường K8s.

#### 2.7.4. Chức năng và nhiệm vụ

Là kho lưu trữ trung tâm cho tất cả Docker Image được build từ GitHub Actions.

### 2.8. Công nghệ 8: GitHub Actions

#### 2.8.1. Khái niệm

Nền tảng CI/CD được tích hợp trực tiếp vào hệ sinh thái GitHub.

#### 2.8.2. Đặc điểm

Dễ cấu hình bằng file YAML, miễn phí cho kho lưu trữ public và hỗ trợ hệ sinh thái Action khổng lồ.

#### 2.8.3. Khả năng

Thực thi các Workflow khi có sự kiện đẩy code (Push) hoặc tạo Pull Request.

#### 2.8.4. Chức năng và nhiệm vụ

Công cụ điều phối toàn bộ luồng CI (Build, Push) và kích hoạt luồng CD.

### 2.9. Công nghệ 9: GitOps

#### 2.9.1. Khái niệm

Phương pháp luận quản lý hạ tầng và ứng dụng trong đó Git Repository được dùng làm "Nguồn sự thật duy nhất" (Single Source of Truth).

#### 2.9.2. Đặc điểm

Mọi thay đổi trên K8s đều phải thông qua Git commit.

#### 2.9.3. Khả năng

Hỗ trợ Rollback siêu nhanh và đảm bảo tính nhất quán (Consistency) của môi trường.

#### 2.9.4. Chức năng và nhiệm vụ

Thay thế phương pháp `kubectl apply` thủ công dễ gây rủi ro.

### 2.10. Công nghệ 10: ArgoCD

#### 2.10.1. Khái niệm

Công cụ Declarative, GitOps continuous delivery tool dành cho Kubernetes.

#### 2.10.2. Đặc điểm

Hoạt động theo cơ chế Pull, liên tục giám sát Git Repo và so sánh với trạng thái thực tế của Cụm.

#### 2.10.3. Khả năng

Tự động đồng bộ hóa (Auto-Sync), tự chữa lành (Self-Heal) khi ai đó sửa bậy cấu hình trên Cluster.

#### 2.10.4. Chức năng và nhiệm vụ

Là bộ não điều phối triển khai Orchestrator, UI và Prometheus Stack trong dự án.

### 2.11. Công nghệ 11: Observability

#### 2.11.1. Khái niệm

Khả năng thấu hiểu trạng thái bên trong của hệ thống phức tạp dựa trên dữ liệu đầu ra bên ngoài.

#### 2.11.2. Đặc điểm

Dựa trên 3 trụ cột: Metrics, Logs, Traces.

#### 2.11.3. Khả năng

Giúp truy vết gốc rễ lỗi (Root Cause Analysis).

#### 2.11.4. Chức năng và nhiệm vụ

Theo dõi hiệu năng của LLM và tài nguyên Kubernetes.

### 2.12. Công nghệ 12: Continuous Evaluation (CE)

#### 2.12.1. Khái niệm

Kỹ thuật đánh giá chất lượng mô hình tự động, liên tục trong pipeline MLOps.

#### 2.12.2. Đặc điểm

Đánh giá dựa trên bộ Testset chuẩn (Ground Truth).

#### 2.12.3. Khả năng

Bảo vệ người dùng cuối khỏi những bản cập nhật LLM kém chất lượng (Quality Gate).

#### 2.12.4. Chức năng và nhiệm vụ

Chạy Job Smoke-Test tự động mỗi khi có code mới để quyết định bản Build đó có an toàn không.

### 2.13. Công nghệ 13: Qdrant

#### 2.13.1. Khái niệm

Cơ sở dữ liệu Vector mã nguồn mở hiệu năng cao.

#### 2.13.2. Đặc điểm

Hỗ trợ tìm kiếm xấp xỉ (ANN), viết bằng Rust nên cực kỳ tối ưu bộ nhớ.

#### 2.13.3. Khả năng

Lưu trữ và truy xuất hàng triệu Vectors nhúng với độ trễ tính bằng mili-giây.

#### 2.13.4. Chức năng và nhiệm vụ

Lưu trữ tài liệu Y khoa (Gút) đã được băm (embedded) bằng HuggingFace.

### 2.14. Công nghệ 14: Ollama

#### 2.14.1. Khái niệm

Công cụ giúp chạy các mô hình ngôn ngữ lớn (LLM) cục bộ một cách dễ dàng.

#### 2.14.2. Đặc điểm

Đóng gói cả Model Weight, Config và Engine suy luận (Llama.cpp) vào chung một thực thể.

#### 2.14.3. Khả năng

Hỗ trợ load mô hình nhanh, tự động offload vRAM và RAM linh hoạt.

#### 2.14.4. Chức năng và nhiệm vụ

Chạy inference mô hình `qwen2.5:1.5b` bên trong cluster K8s.

### 2.15. Công nghệ 15: FastAPI

#### 2.15.1. Khái niệm

Web framework Python dùng để xây dựng API tốc độ cao.

#### 2.15.2. Đặc điểm

Hỗ trợ Async/Await, tự động sinh tài liệu Swagger UI.

#### 2.15.3. Khả năng

Xử lý đồng thời hàng nghìn request, stream dữ liệu hiệu quả.

#### 2.15.4. Chức năng và nhiệm vụ

Làm Orchestrator kết nối UI, Qdrant và Ollama.

### 2.16. Công nghệ 16: Streamlit

#### 2.16.1. Khái niệm

Thư viện Python mã nguồn mở giúp xây dựng ứng dụng web dữ liệu nhanh chóng.

#### 2.16.2. Đặc điểm

Code 100% bằng Python, không cần biết HTML/CSS/JS.

#### 2.16.3. Khả năng

Hiển thị UI tương tác, hỗ trợ cơ chế Chat UI và Streaming.

#### 2.16.4. Chức năng và nhiệm vụ

Là giao diện người dùng trực tiếp để bác sĩ/bệnh nhân giao tiếp với AI.

### 2.17. Công nghệ 17: Langfuse (Bổ sung mới)

#### 2.17.1. Khái niệm

Nền tảng Observability dành riêng cho các ứng dụng LLM/GenAI.

#### 2.17.2. Đặc điểm

Hỗ trợ phân tích độ trễ (Latency), chi phí Token và theo dõi từng bước (Traces) của chuỗi RAG.

#### 2.17.3. Khả năng

Gỡ lỗi quá trình Prompt Engineering và tìm kiếm Vector.

#### 2.17.4. Chức năng và nhiệm vụ

Log mọi thao tác của Orchestrator, cung cấp Dashboard phân tích hiệu suất model LLM.

---

## CHƯƠNG 3. TRIỂN KHAI

### 3.1. Tổng quan kiến trúc hệ thống

Hệ thống được thiết kế theo kiến trúc Microservices và định hướng Cloud-Native. Toàn bộ các dịch vụ được container hóa và điều phối trên Kubernetes. Mã nguồn được đặt trong một Monorepo trên GitHub, phân tách rõ ràng thành các thư mục: `src/` (chứa code ứng dụng), `k8s/` (chứa manifest triển khai), và `.github/workflows/` (chứa CI/CD pipelines).

Luồng đi của hệ thống: Người dùng => Giao diện Streamlit => FastAPI Orchestrator => (Truy vấn Qdrant + Gọi Inference Ollama) => Trả kết quả Streaming về Giao diện. Mọi log và metric được hút về Prometheus/Grafana và Langfuse.

### 3.2. Quản trị hạ tầng và kiến trúc Kubernetes

#### 3.2.1. Thiết kế mạng và phân vùng hạ tầng

Cụm được triển khai trên 3 máy ảo (VM) liên kết nội bộ. Giao tiếp giữa các Node được quản lý thông qua Ansible và giao thức SSH bảo mật. Hệ thống Load Balancer và NodePort được cấu hình để phơi bày (expose) UI ra môi trường bên ngoài một cách an toàn.

#### 3.2.2. Quyết định kỹ thuật về SSH và bảo mật truy cập

Toàn bộ quy trình cài đặt hạ tầng được tự động hóa thông qua Ansible Roles (như Docker, Kubeadm, Calico CNI). Việc sử dụng Ansible giúp loại bỏ hoàn toàn việc SSH tay từng máy, giảm thiểu Human Error.

#### 3.2.3. Khởi tạo Kubernetes bằng Kubeadm

Cụm K8s được khởi tạo bằng công cụ Kubeadm thay vì dùng Managed Service. Điều này đòi hỏi kiến thức sâu về việc khởi tạo Control Plane, join các Worker nodes, và triển khai CNI (Container Network Interface) như Calico để kết nối mạng lưới Pod.

#### 3.2.4. Kiến trúc cụm và giới hạn HA

Hệ thống gồm 1 Control Plane và 2 Worker Nodes. Các ứng dụng được thiết kế Stateless để có thể bị tắt và khởi động lại ngẫu nhiên mà không gây mất mát dữ liệu, trừ Qdrant được gắn Persistent Volume (PV) lưu trữ vector.

#### 3.2.5. Phân bổ tài nguyên và Node Labeling

Trong quá trình triển khai, dự án đã bộc lộ vấn đề về tắc nghẽn phần cứng AI. Giải pháp được áp dụng là **Node Labeling**. Node có tài nguyên mạnh nhất được dán nhãn `role=worker-ai`. Trong file Manifest của Ollama, thuộc tính `nodeSelector` được khai báo để ép Kubernetes Scheduler chỉ được xếp Pod Ollama vào đúng con Node này, giúp cô lập tài nguyên không gây ảnh hưởng đến UI và Orchestrator.

#### 3.2.6. Tối ưu Resource Limits cho Node hạn chế (Khắc phục lỗi Scheduling)

Khi hệ thống chạy trên máy ảo cấp phát cấu hình thấp (2 vCPU hoặc 4 vCPU), việc thiết lập Limits không cẩn thận (Ví dụ: `cpu: 3000m` trên máy 2 vCPU) sẽ khiến Pod rơi vào trạng thái `Pending` do Insufficient CPU, hoặc gây OOMKilled nếu RAM thiếu hụt.
Giải pháp thực tiễn được áp dụng:

- Giám sát linh hoạt sức mạnh Node. Khi Node được cấp 4 vCPU, hiệu chỉnh limits CPU của Ollama lên `3000m` (3 cores), chừa lại đúng 1 core cho hệ điều hành và Kubelet, giúp tăng tốc độ sinh token lên 50% mà vẫn giữ cụm K8s ổn định.
- Tăng RAM Limit của Orchestrator từ 2Gi lên 4Gi để tránh lỗi tràn RAM khi load thư viện Embedding của HuggingFace trong môi trường cục bộ.

### 3.3. Chuỗi cung ứng CI/CD và GitOps

#### 3.3.1. Tối ưu hoá Continuous Integration Pipeline bằng Paths-filter

Monorepo chứa mã nguồn của 4 dịch vụ (UI, Orchestrator, Ingestion, Eval-Job). Nếu một dòng code UI thay đổi mà bắt hệ thống build lại cả 4 dịch vụ thì quá lãng phí. Giải pháp kỹ thuật là sử dụng Action `dorny/paths-filter`. Hệ thống sẽ phân tích Tree Hash của Git, định tuyến chính xác service nào có file thay đổi, và **chỉ Build Image cho Service đó**, các service khác sẽ nhận lệnh `skip`.

#### 3.3.2. SHA Tagging và Immutable Deployment

Quản lý phiên bản không dùng `latest` mà sử dụng chính mã Git Commit SHA làm Docker Tag. Điều này bảo chứng tính bất biến (Immutable), giúp kỹ sư biết chính xác Pod đang chạy code ở thời điểm nào, và Rollback chỉ trong tíc tắc.

#### 3.3.3. Cơ chế GitOps với ArgoCD

CI Pipeline không dùng lệnh `kubectl apply` để đẩy trực tiếp vào cụm (Push-based). Thay vào đó, Job `gitops-update` sẽ sử dụng lệnh `sed` để sửa mã SHA tag trong file Helm `values.yaml` và tự động Commit/Push lại vào Git. ArgoCD đứng trong cụm K8s phát hiện thay đổi này và Pull cấu hình về. Cách tiếp cận này giúp cô lập hoàn toàn môi trường K8s khỏi mạng Internet bên ngoài.

#### 3.3.4. Xử lý Race Condition giữa CI/CD và GitOps (Xử lý lỗi Job dependencies)

Một lỗi nghiêm trọng trong GitHub Actions xuất hiện: Vì Job `gitops-update` cần (`needs`) 4 Job Build hoàn thành, nhưng nhờ công nghệ Paths-filter, 3/4 Job có thể bị `skipped`. Mặc định GitHub Actions sẽ `skip` luôn cả `gitops-update` nếu các Job phụ thuộc bị skip.
**Giải pháp:** Áp dụng biểu thức điều kiện chính xác và nghiêm ngặt:

```yaml
if: always() && !contains(needs.*.result, 'failure') && !contains(needs.*.result, 'cancelled')
```

Luật này buộc GitOps Job vẫn chạy khi có Build Job bị bỏ qua, miễn là không có Job nào bị Crash (Failure), đảm bảo quy trình GitOps luôn được thông suốt.

### 3.4. Thiết kế Pipeline RAG cho hệ thống tư vấn bệnh Gút

#### 3.4.1. Kiến trúc RAG tổng thể

Kiến trúc đi theo mô hình: Load -> Chunk -> Embed -> Store -> Retrieve -> Generate.

#### 3.4.2. Data Ingestion Pipeline

Job `gout-ingestion-job` được triển khai như một K8s Job, tự động cào dữ liệu PDF và JSONL. Việc phân tách văn bản (Splitter) áp dụng thuật toán `RecursiveCharacterTextSplitter` với `chunk_size=1000` và `overlap=200` để đảm bảo ngữ cảnh của một đoạn hội thoại y khoa không bị đứt đoạn.

#### 3.4.3. Embedding Model

Sử dụng thư viện mã nguồn mở `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` chạy trực tiếp trong Orchestrator thay vì dùng API của OpenAI, đảm bảo tiết kiệm chi phí 100% và bảo mật dữ liệu nội bộ.

#### 3.4.4. Vector Database Qdrant

Cấu hình Qdrant với Collection `gout_knowledge_base`, thuật toán tìm kiếm Cosine Similarity.
Để tăng tốc truy vấn LLM trên môi trường CPU giới hạn, dự án quyết định tinh chỉnh tham số k (số lượng chunk trả về) từ `k=3` xuống `k=2`. Điều này giảm tới 33% kích thước Prompt, đẩy nhanh đáng kể tốc độ phản hồi (Prompt Evaluation) của AI.

#### 3.4.5. Prompt Engineering và chống Hallucination

Prompt được khóa chặt bằng câu lệnh: *"Tuyệt đối không được bịa đặt. Nếu tài liệu không có thông tin, hãy trả lời: 'Dựa theo phác đồ hiện tại, tôi không tìm thấy thông tin'."* Điều này đóng vai trò như một rào cản phòng vệ an toàn Y khoa, buộc AI phải "thành thật" và bám sát tài liệu RAG.

### 3.5. Thiết kế Orchestrator Service

#### 3.5.1. FastAPI Gateway Architecture

Được viết bằng Python FastAPI, đóng vai trò là não bộ điều phối mọi request từ UI, liên lạc với Qdrant và gọi HTTP Request sang hệ thống Inference Ollama cục bộ.

#### 3.5.2. Kỹ thuật Streaming Response (Giải quyết lỗi UI timeout 120s)

Quá trình xử lý LLM Inference trên CPU rất chậm. Nếu áp dụng Restful API truyền thống, Giao diện (UI) sẽ phải chờ hơn 120 giây mới nhận được một khối văn bản khổng lồ, gây ra lỗi `Read timed out` kinh điển và làm gián đoạn hệ thống.
**Giải pháp đột phá:** Thiết kế Endpoint mới `/ask/stream`. Bằng cách thiết lập `stream=True` trong thư viện Request và trả về dưới dạng `StreamingResponse (NDJSON)`, Orchestrator lập tức ném ra thông tin `sources` (Nguồn tài liệu) trong vỏn vẹn vài mili-giây, thỏa mãn ngay lập tức điều kiện Connection Timeout của thư viện UI Request.
Sau đó, mỗi khi Ollama rặn ra được một token, FastAPI dùng `yield` để ném token đó về Client. Ở UI, lệnh `st.write_stream()` tiếp nhận và hiển thị chữ chạy ra từ từ như ChatGPT. Trải nghiệm người dùng tăng vọt và lỗi Timeout 120s bị triệt tiêu hoàn toàn.

#### 3.5.3. Langfuse Observability

Mọi câu hỏi và luồng truy xuất RAG đều được theo dõi sát sao bằng decorator `@observe` của Langfuse. Hệ thống đã áp dụng kỹ thuật tiêm ID thủ công (Manual Trace ID Injection) để vượt qua triệt để giới hạn của kiến trúc đa luồng trong FastAPI Streaming, đảm bảo 100% dữ liệu Input và Output đều được luân chuyển an toàn lên Dashboard đám mây. (Chi tiết kỹ thuật xem tại *Case Study 3.7.4*).

### 3.6. Hệ thống đánh giá mô hình (Continuous Evaluation - CE)

#### 3.6.1. Pipeline Evaluation-Job

Evaluation Job được kích hoạt tự động ở bước cuối của CD Pipeline. Job này lấy một tập dữ liệu chuẩn (Testset / Ground Truth) để hỏi AI và tiến hành đối chiếu. Hệ thống sử dụng phương pháp LLM-as-a-judge kết hợp bộ metrics đặc thù (Accuracy, Faithfulness) để chấm điểm tự động.

#### 3.6.2. Cơ chế Auto-Retry và Resilience (Xử lý Race Condition khi Deploy)

Một lỗi mạng (`Connection refused`) cực kỳ tinh vi đã phát sinh: Khi Pipeline CD kích hoạt Evaluation Job, thì bên kia ArgoCD vẫn đang tải Image mới cho Orchestrator (mất 60 giây do dung lượng 600MB). Hệ quả là Evaluation Job gọi API đến Orchestrator chưa sẵn sàng và bị văng lỗi TCP.
**Giải pháp:** Xây dựng cơ chế Auto-Retry mạnh mẽ trong mã Python của Evaluation Job (Lặp lại 15 lần, nghỉ 10s = 150s chờ đợi). Nhờ đó, Job sẽ kiên nhẫn bám trụ đến khi Orchestrator được ArgoCD dựng lên xong xuôi rồi mới đánh giá, chống vỡ Pipeline (Race Condition) thành công rực rỡ.

#### 3.6.3. Quality Gates

Nếu tỷ lệ Pass Rate dưới một mức Quality Gate quy định (Ví dụ: 80%), Job sẽ ném Exit Code 1, Workflow thất bại, tự động đánh dấu cờ đỏ và cảnh báo đội ngũ kỹ sư không được release phiên bản LLM này ra Public.

#### 3.6.4. Lưu trữ Artifacts

Kết quả đánh giá (Log files) và Metadata được GitHub Actions đóng gói tự động thành Artifacts (`eval-results-SHA.zip`), hỗ trợ kỹ sư theo dõi sự thoái hóa (Regression) hay thăng hạng của mô hình qua các thế hệ code khác nhau.

### 3.7. Observability và Monitoring (Giám sát hệ thống)

#### 3.7.1. Triển khai Kube-Prometheus-Stack

Hệ thống giám sát được xây dựng trên nền tảng `kube-prometheus-stack` thông qua Helm và được quản lý vòng đời hoàn toàn tự động bởi ArgoCD. Điểm mạnh của kiến trúc này là khả năng tự động khám phá các endpoint cần theo dõi thông qua `ServiceMonitor` mà không cần can thiệp thủ công.

#### 3.7.2. Bảng điều khiển All-In-One (AIO) Dashboard trên Grafana

Thay vì sử dụng các dashboard rời rạc, dự án đã thiết kế một bảng điều khiển "Tất cả trong một" (All-In-One Dashboard) độc quyền mang tên **Gout LLMOps - AIO**. Dashboard này cung cấp một cái nhìn toàn cảnh (Single pane of glass) xuyên suốt từ tầng cụm K8s, ứng dụng đến cả các batch jobs. Các nhóm chỉ số cốt lõi bao gồm:

*   **Cluster Overview (Tổng quan K8s):** Đo lường số lượng *Nodes Ready*, *Pods Running* trong namespace `gout-eval`, cùng với mức tiêu thụ CPU và RAM tổng thể của toàn cụm.
*   **Node Resources (Tài nguyên Máy ảo):** Trực quan hóa phần trăm ngốn CPU và RAM của từng Node riêng biệt (giúp theo dõi sát sao sức chịu đựng của Node `worker-ai`).
*   **gout-eval Applications (Ứng dụng Gout-LLM):**
    *   *Pod Status:* Sử dụng hàm Instant Query `sum by(pod) (kube_pod_container_status_restarts_total)` để bắt chính xác số lần Restart của từng Pod (UI, Orchestrator, Qdrant, Ollama). Khắc phục hoàn toàn lỗi ẩn Pod thường gặp khi dùng hàm `increase`.
    *   *CPU & Memory per Pod:* Giám sát tài nguyên thực tế của từng container để chống OOM.
*   **Jobs (Giám sát CI/CD & Batch Processing):**
    *   Đo lường sự thành công/thất bại (Status) và Tổng thời gian chạy (Duration) của 2 luồng Job trọng yếu: `gout-ingestion-job` (Bơm dữ liệu RAG) và `evaluation-job` (Chấm điểm mô hình tự động). Giúp phát hiện sớm nếu Job chấm điểm bị treo quá lâu.

Toàn bộ biểu đồ AIO này được lưu trữ theo triết lý **Configuration-as-Code** dưới dạng `K8s ConfigMap` (`k8s/grafana/aio-dashboard.yaml`). Nhờ đó, bất kỳ sự thay đổi biểu đồ nào cũng được quản lý phiên bản qua Git và đồng bộ hóa tự động qua ArgoCD, triệt tiêu hoàn toàn rủi ro sai lệch cấu hình thường gặp khi thao tác tay trên giao diện Grafana.

#### 3.7.3. Khắc phục lỗi PromQL hiển thị Pod Status (Blank Data)

Trong bảng Pod Status, công thức PromQL ban đầu dùng hàm `increase([1h])`. Tuy nhiên, với tính chất CI/CD liên tục, các Pod (UI, Orchestrator) thường xuyên bị tiêu hủy và sinh ra mới. Hàm `increase` bị thiếu mốc dữ liệu quá khứ nên trả về khoảng trắng (Blank) trên Grafana.
Dự án đã cải tiến bằng cách đổi sang hàm Instant Query tính tổng trực tiếp `sum by(pod) (kube_pod_container_status_restarts_total)`, qua đó khắc phục hoàn toàn lỗi ẩn Pod, giúp bảng Dashboard theo dõi Restart count chạy chính xác theo thời gian thực (Real-time).

#### 3.7.4. Case Study: Khắc phục lỗi mất dấu ContextVar và Cơ chế Upsert của Langfuse trong FastAPI Streaming

**Ngữ cảnh:** Hệ thống sử dụng Langfuse API để truy vết (Trace) quá trình tạo văn bản của mô hình AI. Tuy nhiên, khi kết hợp với công nghệ Streaming (sinh chữ theo thời gian thực), một lỗi kiến trúc nghiêm trọng đã xảy ra khiến Dashboard trên Cloud hoàn toàn trắng dữ liệu.

**Vấn đề 1: Thất thoát ContextVar do Đa luồng (Multithreading)**
FastAPI xử lý đối tượng `StreamingResponse` bằng cách đẩy hàm Generator (hàm dùng `yield`) sang một Background Thread nhằm chống nghẽn nghẽn luồng chính. Sự chuyển dịch này vô tình làm mất hoàn toàn `ContextVar` gốc mà Python dùng để lưu ID của Trace hiện tại. Hệ quả là câu lệnh `langfuse_context.update_current_trace` bị sụp đổ kèm log lỗi: *"No trace found in the current context"*.
*   **Giải pháp 1 (Tiêm ID thủ công):** Đội ngũ kỹ thuật đã tái cấu trúc mã nguồn bằng cách "chộp" lấy `trace_id` bằng lệnh `get_current_trace_id()` ngay tại luồng chính (Main Thread) *trước* khi Generator được gọi. Sau đó, truyền chuỗi ID tĩnh này vào bên trong Thread ngầm, rồi khởi tạo trực tiếp một Client `Langfuse()` để cưỡng ép cập nhật dữ liệu.

**Vấn đề 2: Cơ chế Ghi đè (Upsert Overwrite) phá hủy Input**
Mặc dù dữ liệu Output đã bay lên Cloud thành công, hệ thống lại tiếp tục ghi nhận hiện tượng "bốc hơi" dữ liệu Input. Nguyên nhân sâu xa nằm ở cơ chế Upsert của kiến trúc REST API Langfuse. Khi gửi gói tin cập nhật chứa `trace_id` và `output` ở cuối luồng Streaming, thư viện Python ngầm định trường `input` là `Null`. Langfuse Cloud tiếp nhận gói tin và ghi đè sự trống rỗng này lên dữ liệu Input cũ đã được khởi tạo lúc đầu.
*   **Giải pháp 2 (Bảo tồn Trạng thái toàn vẹn):** Cấu trúc lại tham số của hàm cập nhật thủ công: Bắt buộc đính kèm cả `input` ban đầu và `metadata` của mô hình vào lệnh gọi `lf.trace(...)` ở cuối luồng. Giải pháp tinh xảo này đã bảo vệ thành công tính toàn vẹn của dữ liệu, khắc phục hoàn hảo điểm yếu chí tử khi giám sát hệ thống AI Streaming.

### 3.8. Giao diện người dùng

Sử dụng công nghệ **Streamlit** mạnh mẽ. Giao diện có tích hợp Sidebar cho phép lựa chọn Model linh hoạt (như `qwen2.5:1.5b`), kết hợp cơ chế State Management (`st.session_state`) lưu trữ ngữ cảnh hội thoại. Điểm nhấn là ứng dụng thành thạo Generator Function để parse luồng NDJSON từ Backend, đem lại tốc độ phản hồi chớp nhoáng. Việc đồng bộ hóa Hardcode Options trong Python UI và Backend Fallbacks được thực hiện thông suốt qua các bản cập nhật CD liên tục.

### 3.9. Bảo mật hệ thống

- Mọi dữ liệu nhạy cảm (Telegram Tokens, OpenAI API keys) được mã hóa trong GitHub Repository Secrets.
- Trong Kubernetes, Secret được triển khai để kết nối biến môi trường (EnvVar) một cách an toàn cho các ứng dụng nội bộ.
- Ngăn cấm truy cập từ ngoài vào nội bộ K8s, chỉ phơi bày duy nhất giao diện UI thông qua Service định tuyến (NodePort/LoadBalancer) đã quy định. Hệ thống bảo mật toàn vẹn khỏi các cuộc tấn công quét lỗ hổng bên ngoài.

### 3.10. Quản lý lưu trữ bền vững (Persistent Storage)

Trong môi trường Kubernetes IaaS (như OpenStack Kubeadm), việc đảm bảo tính toàn vẹn dữ liệu (Statefulness) cho các Pod chứa cơ sở dữ liệu là một thách thức vô cùng lớn. Nếu Pod bị sập hoặc khởi động lại, các dữ liệu tạm thời (Ephemeral Storage) bên trong Container sẽ bị tiêu hủy hoàn toàn, dẫn đến việc Vector Database (Qdrant) mất sạch tài liệu y khoa đã Ingest và Inference Worker (Ollama) rơi rụng mất mô hình ngôn ngữ lớn (LLM).

Để khắc phục rủi ro "mất trí nhớ" này và đưa hệ thống lên đẳng cấp Production-ready, dự án đã triển khai chiến thuật **Node Affinity kết hợp HostPath Volume**:

1. **Cố định Node (Node Selector):** Bổ sung cờ `nodeSelector: role: worker-ai` vào file cấu hình của cả Qdrant và Ollama. Thao tác này sẽ "khóa chặt" hai ứng dụng này, buộc chúng phải chạy vĩnh viễn trên một máy ảo duy nhất có sức mạnh tính toán cao nhất cụm (Node `vm3` - 4vCPU), ngăn chặn triệt để hiện tượng Kubernetes tự động dời Pod sang máy ảo khác làm mất kết nối với ổ cứng vật lý cũ.
2. **Khai thông Ổ cứng Vật lý (HostPath):** Gắn trực tiếp thư mục vật lý của máy ảo (`/mnt/data/qdrant` và `/mnt/data/ollama`) vào sâu bên trong Container bằng cơ chế `hostPath`. Kết hợp với tùy chọn `type: DirectoryOrCreate`, K8s sẽ tự động khởi tạo thư mục trên ổ cứng nếu chưa tồn tại.

**Hiệu quả mang lại:** Nhờ chiến lược khôn khéo này, toàn bộ khối lượng dữ liệu khổng lồ của Qdrant và Ollama được "bất tử hóa" trên ổ cứng gốc của Node `worker-ai`. Ngay cả khi gặp sự cố cúp điện toàn hệ thống hoặc buộc phải khởi động lại máy ảo (Reboot VM), cụm K8s vừa khởi động lên là AI đã lập tức kết nối lại được với "Bộ não" (Mô hình LLM) và "Trí nhớ" (Vector DB) trong tích tắc. Đội ngũ không cần phải tốn hàng giờ đồng hồ để kéo lại mô hình hay chạy lại các luồng Ingestion Job cồng kềnh như trước nữa.

*Bên cạnh cụm AI, hệ thống Giám sát (Monitoring Stack) cũng áp dụng triết lý lưu trữ bền vững tương tự. Thông qua cấu hình Helm trong ArgoCD, cả `Prometheus` và `Alertmanager` đều được cấp phát ổ cứng động bằng `volumeClaimTemplate` (Yêu cầu PVC 1Gi). Việc này đảm bảo toàn bộ lịch sử dữ liệu Time-series của hệ thống không bị bốc hơi mỗi khi Pod giám sát khởi động lại, giúp Grafana luôn có chuỗi dữ liệu quá khứ để vẽ biểu đồ.*

---

## CHƯƠNG 5: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### 5.1. Kết luận

Dự án đã thực hiện thành công việc xây dựng và triển khai một hệ thống **LLMOps toàn diện** cho trợ lý AI y khoa (Gout-LLM), tích hợp mượt mà chu trình từ lập trình, kiểm thử đến vận hành trên nền tảng hạ tầng đám mây tự quản (Self-hosted Cloud). Những thành tựu kỹ thuật nổi bật đạt được bao gồm:

1. **Làm chủ Hạ tầng tự động (IaC):** Sử dụng xuất sắc Terraform và Ansible để cấp phát và cấu hình cụm máy ảo OpenStack, sau đó bootstrapping thành công cụm Kubernetes (Kubeadm) 3-nodes chuẩn Production.
2. **Triển khai GitOps hiện đại:** Chuẩn hóa quy trình phân phối phần mềm bằng ArgoCD, đưa toàn bộ cấu hình hệ thống (từ ứng dụng đến Grafana Dashboard) về dạng Configuration-as-Code, giúp ngăn chặn triệt để lỗi thao tác thủ công (Configuration Drift).
3. **Xây dựng luồng RAG tối ưu:** Kết hợp sức mạnh của Qdrant (Vector DB) và Ollama (Local LLM), được điều phối bởi FastAPI hỗ trợ công nghệ **Streaming NDJSON** mượt mà, giúp triệt tiêu hoàn toàn lỗi Timeout 120s và tăng cường trải nghiệm người dùng trên Streamlit UI.
4. **Giám sát và Đánh giá liên tục (CE & Observability):** Thiết lập thành công luồng Evaluation tự động (LLM-as-a-judge) đóng vai trò như một Quality Gate vững chắc. Đồng thời, hệ thống AIO Dashboard trên Grafana và Langfuse Cloud mang lại khả năng giám sát "Xuyên thấu" (Single pane of glass) từ tài nguyên máy ảo vật lý, trạng thái Pod, độ trễ API, cho đến chất lượng sinh từ của mô hình AI.
5. **Giải quyết triệt để các bài toán hóc búa:** Đội ngũ đã xử lý thành công hàng loạt thách thức kiến trúc mức cao như: Khắc phục Race Condition trong CI/CD bằng cơ chế Auto-Retry, trị dứt điểm lỗi mất `ContextVar` khi kết hợp Streaming và Langfuse, và thiết lập lưu trữ bền vững (Persistent Storage) với Node Affinity trên K8s.

### 5.2. Hạn chế

Bên cạnh những thành tựu đạt được, hệ thống ở phiên bản hiện tại vẫn tồn tại một số giới hạn nhất định:

1. **Giới hạn phần cứng (Hardware Bottleneck):** Việc chạy inference cho mô hình LLM (như Qwen2.5) hoàn toàn dựa trên CPU của node `worker-ai` (4vCPU) dẫn đến thời gian sinh token (Time-To-First-Token) còn khá chậm so với kỳ vọng thực tế.
2. **Điểm yếu lưu trữ cục bộ (Local Storage Limitation):** Dù đã cấu hình an toàn bằng cơ chế `nodeSelector` và `hostPath`, nhưng việc neo chặt dữ liệu của Qdrant và Ollama vào ổ cứng của một máy ảo duy nhất làm giảm đi tính sẵn sàng cao (High Availability) và khả năng di dời Pod linh hoạt của Kubernetes.
3. **Bảo mật nội bộ (Internal Security):** Mạng lưới giao tiếp giữa các vi dịch vụ (Microservices) bên trong cụm K8s vẫn đang sử dụng giao thức HTTP thuần, chưa được mã hóa. Giao diện UI hướng người dùng cũng chưa tích hợp cơ chế xác thực (Authentication/SSO).

### 5.3. Hướng phát triển

Để khắc phục các hạn chế trên và mở rộng quy mô hệ thống, định hướng phát triển trong tương lai sẽ tập trung vào các khía cạnh sau:

1. **Tăng tốc phần cứng (GPU Acceleration):** Trang bị thêm các Node có GPU (như NVIDIA T4 hoặc A10G) và triển khai NVIDIA Device Plugin cho Kubernetes. Bước tiến này sẽ giảm thiểu độ trễ sinh chữ xuống mức mili-giây và cho phép hệ thống gánh vác các mô hình lớn hơn (7B - 13B tham số).
2. **Lưu trữ phân tán (Distributed Storage):** Thay thế `hostPath` bằng các giải pháp lưu trữ khối phân tán chuyên dụng cho K8s như **Ceph**, **Longhorn** hoặc NFS. Điều này giúp tách biệt hoàn toàn dữ liệu khỏi máy ảo vật lý, cho phép Pod Qdrant và Ollama tự do di chuyển (Failover) sang các Node khác khi xảy ra sự cố sập máy chủ.
3. **Nâng cấp kỹ thuật RAG:** Tích hợp thêm các kỹ thuật truy xuất tiên tiến như Re-ranking (Cohere/BGE), Hybrid Search (Kết hợp BM25 và Vector), hoặc GraphRAG để cải thiện độ chính xác và khả năng lý luận y khoa của trợ lý AI.
4. **Kiến trúc Service Mesh:** Triển khai Istio hoặc Linkerd để tự động mã hóa toàn bộ lưu lượng mạng nội bộ (mTLS) và mở ra khả năng phân luồng giao thông tinh vi (Canary Deployment, Circuit Breaking).
5. **Huấn luyện mô hình tự động (Continuous Training):** Mở rộng đường ống CI/CD hiện tại để bao gồm thêm chặng Fine-tuning (sử dụng LoRA/QLoRA). Nếu điểm số Evaluation rớt xuống dưới ngưỡng an toàn do dữ liệu bị trôi (Data Drift), hệ thống sẽ tự động kích hoạt Job huấn luyện lại mô hình để thích ứng với tập kiến thức mới.
