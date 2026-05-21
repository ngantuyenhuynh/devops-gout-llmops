# Luồng deploy: Terraform → Ansible → GitHub Secrets → CD tự động

## Bước 0 — Tạo GCP Service Accounts (chạy 1 lần khi reproduct trên project mới)

```bash
PROJECT=<your-gcp-project-id>

# --- SA cho Ansible (provision cluster, tạo imagePullSecret) ---
gcloud iam service-accounts create sa-ansible --project=$PROJECT

gcloud projects add-iam-policy-binding $PROJECT \
  --member="serviceAccount:sa-ansible@${PROJECT}.iam.gserviceaccount.com" \
  --role="roles/compute.admin"

gcloud projects add-iam-policy-binding $PROJECT \
  --member="serviceAccount:sa-ansible@${PROJECT}.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

gcloud projects add-iam-policy-binding $PROJECT \
  --member="serviceAccount:sa-ansible@${PROJECT}.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.reader"

gcloud iam service-accounts keys create ~/sa-ansible-key.json \
  --iam-account=sa-ansible@${PROJECT}.iam.gserviceaccount.com

# --- SA cho GitHub Actions (CI build + push image) ---
gcloud iam service-accounts create github-actions-sa --project=$PROJECT

gcloud artifacts repositories create gout-llmops-repo \
  --repository-format=docker \
  --location=asia-southeast1 \
  --project=$PROJECT

gcloud artifacts repositories add-iam-policy-binding gout-llmops-repo \
  --location=asia-southeast1 \
  --member="serviceAccount:github-actions-sa@${PROJECT}.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer" \
  --project=$PROJECT

gcloud iam service-accounts add-iam-policy-binding \
  github-actions-sa@${PROJECT}.iam.gserviceaccount.com \
  --member="serviceAccount:github-actions-sa@${PROJECT}.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountTokenCreator" \
  --project=$PROJECT

gcloud iam service-accounts keys create ~/github-actions-sa-key.json \
  --iam-account=github-actions-sa@${PROJECT}.iam.gserviceaccount.com
```

## Bước 1 — Chuẩn bị local

```bash
gcloud auth login
gcloud config set project <your-gcp-project-id>

# Tạo SSH key nếu chưa có
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""
```

## Bước 2 — Terraform: tạo 3 VMs

```bash
cd infra/terraform
terraform init
terraform apply
```

Sau khi xong, lấy IPs:

```bash
terraform output vm_ips
# hoặc lấy sẵn inventory:
terraform output ansible_inventory
```

## Bước 3 — Điền IPs vào Ansible inventory

Mở [infra/ansible/inventory.yml](ansible/inventory.yml), thay IP hiện tại bằng IPs thực tế từ bước trên.

## Bước 4 — Ansible: cài K8s + deploy apps

```bash
cd infra/ansible

ansible-playbook -i inventory.yml site.yml \
  -e gcp_sa_key_file=~/sa-ansible-key.json
```

Playbook chạy theo thứ tự:

```
vm1, vm2, vm3 → common         (containerd + kubeadm)
vm1           → control-plane  (kubeadm init + Flannel)
vm2, vm3      → worker         (kubeadm join)
localhost     → local-kubectl  (copy kubeconfig)
              → namespaces     (gout-eval, monitoring, argocd)
              → argocd         (install ArgoCD)
              → argocd-apps    (local-path-provisioner + deploy tất cả apps)
```

## Bước 5 — Thêm GitHub Secrets và Variables

GitHub → repo → Settings → Secrets and variables → Actions

**Secrets:**

| Tên | Giá trị |
| --- | --- |
| `KUBECONFIG_B64` | `cat ~/.kube/config \| base64 -w 0` |
| `GCP_SA_KEY` | nội dung `~/sa-ansible-key.json` |
| `GCP_CREDENTIALS` | nội dung `~/github-actions-sa-key.json` |
| `OPENAI_API_KEY` | OpenAI API key |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key |

**Variables:**

| Tên | Giá trị |
| --- | --- |
| `PROJECT_ID` | GCP project ID |
| `GCP_REGION` | `asia-southeast1` |
| `REPO_NAME` | `gout-llmops-repo` |
| `K8S_NAMESPACE` | `gout-eval` |

## Bước 6 — Trigger CI/CD

```bash
# CI tự động khi push thay đổi src/ hoặc k8s/ lên main
git push origin main

# Hoặc trigger thủ công:
# GitHub Actions → CI: Build, Scan & Push Services → Run workflow
# GitHub Actions → CD: Deploy to GKE & Smoke Test → Run workflow
```

## Sơ đồ tổng quát

```
terraform apply
      ↓
   vm1, vm2, vm3 (GCE)
      ↓
ansible-playbook
      ↓
   K8s cluster (kubeadm)
   ArgoCD + apps running
      ↓
git push src/ → main
      ↓
   CI: build + push image → Artifact Registry
      ↓
   CD: kubectl apply + smoke test
      ↓
   ArgoCD: sync helm values → gout-ui, eval-orchestrator
```
