variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region (Autopilot yêu cầu regional)"
  type        = string
  default     = "asia-southeast1"
}

variable "cluster_name" {
  description = "GKE cluster name"
  type        = string
  default     = "gout-llmops-cluster"
}
