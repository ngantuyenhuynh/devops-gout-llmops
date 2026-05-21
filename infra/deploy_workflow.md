# Luồng deploy: Terraform → Ansible → GitHub Secrets → CD tự động

# Bước 1 — Chuẩn bị local

```
# Cài tools
gcloud auth login
gcloud config set project macro-mender-494903-v0

# Tạo SSH key nếu chưa có
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""
```

# Bước 2 — Terraform: tạo 3 VMs

```
cd infra/terraform
terraform init
terraform apply
```

# Sau khi xong, lấy IPs

```
terraform output vm_ips
# hoặc lấy sẵn inventory:
terraform output ansible_inventory
```

# Bước 3 — Điền IPs vào Ansible inventory

Mở infra/ansible/inventory.yml, thay FILL_FROM_TERRAFORM_OUTPUT bằng IPs thực tế từ bước trên.

# Bước 4 — Ansible: cài K8s + deploy apps

```
cd infra/ansible

# Export GCP SA key để tạo imagePullSecret
export GCP_SA_KEY=$(cat path/to/sa-key.json)

ansible-playbook -i inventory.yml site.yml
```

# Playbook sẽ chạy theo thứ tự

```
vm1, vm2, vm3 → common    (containerd + kubeadm)
vm1           → control-plane  (kubeadm init + Flannel)
vm2, vm3      → worker         (kubeadm join)
localhost     → local-kubectl  (copy kubeconfig)
              → namespaces     (gout-eval, monitoring, argocd)
              → argocd         (install ArgoCD)
              → argocd-apps    (deploy tất cả apps)
```

# Bước 5 — Thêm GitHub Secrets

```
# KUBECONFIG_B64
cat ~/.kube/config | base64 -w 0
# → copy output, thêm vào GitHub Secrets với tên KUBECONFIG_B64

# GCP_SA_KEY (để CD tạo imagePullSecret mỗi lần deploy)
cat path/to/sa-key.json
# → copy toàn bộ JSON, thêm vào GitHub Secrets với tên GCP_SA_KEY
GitHub → Settings → Secrets and variables → Actions → New repository secret
```

# Bước 6 — Trigger CD

```
Push bất kỳ thay đổi src/ lên main → CI build → CD deploy tự động.

Hoặc trigger thủ công: GitHub Actions → CD workflow → Run workflow.
```

# Sơ đồ tổng quát

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

# Tạo key

```
PROJECT=macro-mender-494903-v0
SA=sa-ansible@${PROJECT}.iam.gserviceaccount.com

# Gán roles
gcloud projects add-iam-policy-binding $PROJECT \
  --member="serviceAccount:$SA" \
  --role="roles/compute.admin"

gcloud projects add-iam-policy-binding $PROJECT \
  --member="serviceAccount:$SA" \
  --role="roles/iam.serviceAccountUser"

gcloud projects add-iam-policy-binding $PROJECT \
  --member="serviceAccount:$SA" \
  --role="roles/artifactregistry.reader"

# Tạo key
gcloud iam service-accounts keys create ~/sa-ansible-key.json \
  --iam-account=$SA
```
