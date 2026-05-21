output "cluster_name" {
  value = google_container_cluster.main.name
}

output "cluster_zone" {
  value = google_container_cluster.main.location
}

output "get_credentials_command" {
  value = "gcloud container clusters get-credentials ${google_container_cluster.main.name} --zone ${google_container_cluster.main.location} --project ${var.project_id}"
}
