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

resource "google_compute_firewall" "allow_internal" {
  name    = "${var.cluster_name}-allow-internal"
  network = google_compute_network.vpc.name

  # Không chặn IP nội bộ: Cho phép toàn bộ giao tiếp giữa các node (Kubeadm, Docker, Flannel, v.v.)
  allow {
    protocol = "tcp"
  }
  allow {
    protocol = "udp"
  }
  allow {
    protocol = "icmp"
  }

  # Dải IP của VPC subnet và Pod/Service network (theo vpc.tf)
  source_ranges = ["10.10.0.0/20", "10.20.0.0/16", "10.30.0.0/20"]
  target_tags   = ["k3s-node"]
}

resource "google_compute_firewall" "allow_public" {
  name    = "${var.cluster_name}-allow-public"
  network = google_compute_network.vpc.name

  # Thắt chặt truy cập từ Internet: Chỉ mở các cổng thiết yếu
  allow {
    protocol = "tcp"
    ports    = ["22", "80", "443", "6443", "30000-32767"]
    # 22: SSH
    # 80, 443: HTTP/HTTPS (Ingress)
    # 6443: Kube API Server (nếu cần dùng kubectl từ bên ngoài)
    # 30000-32767: NodePorts (nếu cần expose)
    # Lưu ý: Các cổng 9100, 10250 đã được ẩn khỏi public và chỉ giao tiếp qua internal
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
