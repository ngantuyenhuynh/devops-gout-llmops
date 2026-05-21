output "vm_ips" {
  description = "External IPs của các VM — dùng để điền vào Ansible inventory"
  value = {
    for name, vm in google_compute_instance.vms :
    name => vm.network_interface[0].access_config[0].nat_ip
  }
}

output "ansible_inventory" {
  description = "Copy nội dung này vào infra/ansible/inventory.yml"
  value = <<-EOT
    all:
      children:
        control_plane:
          hosts:
            vm1:
              ansible_host: ${google_compute_instance.vms["vm1"].network_interface[0].access_config[0].nat_ip}
        workers:
          hosts:
            vm2:
              ansible_host: ${google_compute_instance.vms["vm2"].network_interface[0].access_config[0].nat_ip}
            vm3:
              ansible_host: ${google_compute_instance.vms["vm3"].network_interface[0].access_config[0].nat_ip}
  EOT
}
