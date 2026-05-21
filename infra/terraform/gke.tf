resource "google_container_cluster" "main" {
  name     = var.cluster_name
  location = var.region # Autopilot bắt buộc regional

  enable_autopilot = true

  network    = google_compute_network.vpc.name
  subnetwork = google_compute_subnetwork.subnet.name

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  deletion_protection = false
}
