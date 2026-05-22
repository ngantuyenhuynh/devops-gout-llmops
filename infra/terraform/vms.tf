locals {
  vms = {
    vm1 = {
      machine_type = var.control_plane_machine_type
      role         = "control-plane"
    }
    vm2 = {
      machine_type = var.worker_machine_type
      role         = "worker"
    }
    vm3 = {
      machine_type = var.ai_worker_machine_type
      role         = "worker-ai"
    }
  }
}

resource "google_compute_firewall" "k3s" {
  name    = "${var.cluster_name}-k3s"
  network = google_compute_network.vpc.name

  allow {
    protocol = "tcp"
    ports    = ["22", "6443", "9100", "10250", "30000-32767"]
                               # ^^^ thêm 9100
  }
  allow {
    protocol = "udp"
    ports    = ["8472"]
  }
  allow {
    protocol = "icmp"
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["k3s-node"]
}

resource "google_compute_instance" "vms" {
  for_each     = local.vms
  name         = "${var.cluster_name}-${each.key}"
  machine_type = each.value.machine_type
  zone         = var.zone

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-12"
      size  = 30
      type  = "pd-standard"
    }
  }

  network_interface {
    network    = google_compute_network.vpc.name
    subnetwork = google_compute_subnetwork.subnet.name
    access_config {} # Cấp External IP để SSH vào
  }

  metadata = {
    ssh-keys = "${var.ssh_user}:${file(var.ssh_public_key_path)}"
  }

  tags = ["k3s-node"]

  labels = {
    role = each.value.role
  }
}
