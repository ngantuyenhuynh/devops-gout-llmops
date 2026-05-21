resource "google_container_cluster" "main" {
  name     = var.cluster_name
  location = var.zone

  remove_default_node_pool = true
  initial_node_count       = 1

  network    = google_compute_network.vpc.name
  subnetwork = google_compute_subnetwork.subnet.name

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  deletion_protection = false
}

# vm1, vm2 — app nodes (qdrant, eval-orchestrator, gout-ui, monitoring, argocd)
resource "google_container_node_pool" "app_nodes" {
  name       = "app-node-pool"
  location   = var.zone
  cluster    = google_container_cluster.main.name
  node_count = var.app_node_count

  node_config {
    machine_type = var.app_node_machine_type
    disk_size_gb = 30
    disk_type    = "pd-standard"

    labels = {
      workload-type = "app"
    }

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]

    metadata = {
      disable-legacy-endpoints = "true"
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# vm3 — dedicated AI node for Ollama (qwen2:1.5b ~2 GiB runtime)
resource "google_container_node_pool" "ai_node" {
  name       = "ai-node-pool"
  location   = var.zone
  cluster    = google_container_cluster.main.name
  node_count = 1

  node_config {
    machine_type = var.ai_node_machine_type
    disk_size_gb = 30
    disk_type    = "pd-standard"

    labels = {
      workload-type = "ai"
    }

    # Prevent non-AI pods from landing on this node
    taint {
      key    = "dedicated"
      value  = "ai"
      effect = "NO_SCHEDULE"
    }

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]

    metadata = {
      disable-legacy-endpoints = "true"
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}
