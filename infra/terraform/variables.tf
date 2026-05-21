variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "asia-southeast1"
}

variable "zone" {
  description = "GCP zone"
  type        = string
  default     = "asia-southeast1-a"
}

variable "cluster_name" {
  description = "Prefix cho tên các VM"
  type        = string
  default     = "gout-llmops"
}

variable "control_plane_machine_type" {
  description = "vm1: k3s server + ArgoCD + monitoring"
  type        = string
  default     = "e2-medium" # 2 vCPU, 4 GiB
}

variable "worker_machine_type" {
  description = "vm2: qdrant + eval-orchestrator + gout-ui"
  type        = string
  default     = "e2-medium" # 2 vCPU, 4 GiB
}

variable "ai_worker_machine_type" {
  description = "vm3: Ollama (qwen2:1.5b ~2 GiB runtime)"
  type        = string
  default     = "e2-standard-2" # 2 vCPU, 8 GiB
}

variable "ssh_user" {
  description = "Linux user để SSH vào VM"
  type        = string
  default     = "devops"
}

variable "ssh_public_key_path" {
  description = "Đường dẫn tới SSH public key"
  type        = string
  default     = "~/.ssh/id_rsa.pub"
}
