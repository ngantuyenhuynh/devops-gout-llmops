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
  description = "GCP zone for zonal GKE cluster"
  type        = string
  default     = "asia-southeast1-a"
}

variable "cluster_name" {
  description = "GKE cluster name"
  type        = string
  default     = "gout-llmops-cluster"
}

variable "app_node_count" {
  description = "Number of app nodes (vm1, vm2)"
  type        = number
  default     = 2
}

variable "app_node_machine_type" {
  description = "Machine type for app nodes"
  type        = string
  default     = "e2-standard-2" # 2 vCPU, 8 GiB — đủ cho qdrant+eval+ui+monitoring+argocd
}

variable "ai_node_machine_type" {
  description = "Machine type for dedicated AI/Ollama node (vm3)"
  type        = string
  default     = "e2-standard-2" # 2 vCPU, 8 GiB — đủ cho qwen2:1.5b (~2 GiB runtime)
}
