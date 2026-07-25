resource "google_container_cluster" "manager" {
  project  = var.project_id
  name     = "${var.name_prefix}-manager"
  location = var.manager_region

  network    = var.network
  subnetwork = var.manager_subnetwork

  deletion_protection         = var.deletion_protection
  remove_default_node_pool    = true
  initial_node_count          = 1
  node_locations              = var.manager_zones
  networking_mode             = "VPC_NATIVE"
  enable_shielded_nodes       = true
  enable_intranode_visibility = true

  release_channel {
    channel = var.release_channel
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  fleet {
    project = var.project_id
  }

  # These labels ask GKE Fleet to generate ClusterProfile inventory objects in
  # kueue-system. MultiKueue can then use federated credentials instead of a
  # stored worker kubeconfig and bearer token.
  resource_labels = merge(local.common_labels, {
    fleet-clusterinventory-management-cluster = "true"
    fleet-clusterinventory-namespace          = "kueue-system"
    role                                      = "manager"
  })

  ip_allocation_policy {
    cluster_secondary_range_name  = var.manager_cluster_secondary_range_name
    services_secondary_range_name = var.manager_services_secondary_range_name
  }

  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = var.enable_private_endpoint
    master_ipv4_cidr_block  = var.manager_master_ipv4_cidr_block

    master_global_access_config {
      enabled = true
    }
  }

  dynamic "master_authorized_networks_config" {
    for_each = var.enable_private_endpoint ? [] : [1]
    content {
      dynamic "cidr_blocks" {
        for_each = var.master_authorized_networks
        content {
          cidr_block   = cidr_blocks.value.cidr_block
          display_name = cidr_blocks.value.display_name
        }
      }
    }
  }

  addons_config {
    gce_persistent_disk_csi_driver_config {
      enabled = true
    }
  }

  secret_manager_config {
    enabled = true
    rotation_config {
      enabled           = true
      rotation_interval = "300s"
    }
  }

  secret_sync_config {
    enabled = true
    rotation_config {
      enabled           = true
      rotation_interval = "300s"
    }
  }

  security_posture_config {
    mode               = "BASIC"
    vulnerability_mode = "VULNERABILITY_BASIC"
  }

  monitoring_config {
    enable_components = [
      "APISERVER",
      "CONTROLLER_MANAGER",
      "DAEMONSET",
      "DEPLOYMENT",
      "HPA",
      "KUBELET",
      "POD",
      "SCHEDULER",
      "STATEFULSET",
      "STORAGE",
      "SYSTEM_COMPONENTS",
    ]
    managed_prometheus {
      enabled = true
    }
  }

  maintenance_policy {
    recurring_window {
      start_time = "2026-01-04T09:00:00Z"
      end_time   = "2026-01-04T13:00:00Z"
      recurrence = "FREQ=WEEKLY;BYDAY=SU"
    }
  }

  lifecycle {
    precondition {
      condition     = var.enable_private_endpoint || length(var.master_authorized_networks) > 0
      error_message = "Provide master_authorized_networks or enable the private control-plane endpoint."
    }
  }

  depends_on = [google_project_service.required]
}

resource "google_container_node_pool" "manager_system" {
  project  = var.project_id
  name     = "system"
  location = google_container_cluster.manager.location
  cluster  = google_container_cluster.manager.name

  # Regional node-pool initial counts are per zone. Start with one in each
  # configured zone, then let total autoscaling enforce the aggregate floor.
  initial_node_count = 1

  autoscaling {
    total_min_node_count = var.manager_system_min_nodes
    total_max_node_count = var.manager_system_max_nodes
    location_policy      = "BALANCED"
  }

  node_config {
    machine_type    = var.manager_system_machine_type
    image_type      = "COS_CONTAINERD"
    service_account = google_service_account.manager_nodes.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]
    labels = {
      "tpu-ci.google.com/role" = "system"
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    shielded_instance_config {
      enable_integrity_monitoring = true
      enable_secure_boot          = true
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
  }

  lifecycle {
    ignore_changes = [initial_node_count]

    precondition {
      condition     = var.manager_system_min_nodes >= length(var.manager_zones)
      error_message = "manager_system_min_nodes must provide at least one system node per manager zone."
    }
  }
}

resource "google_container_cluster" "worker" {
  for_each = var.worker_clusters

  project  = var.project_id
  name     = "${var.name_prefix}-${each.key}"
  location = each.value.location

  network    = var.network
  subnetwork = each.value.subnetwork

  deletion_protection         = var.deletion_protection
  remove_default_node_pool    = true
  initial_node_count          = 1
  networking_mode             = "VPC_NATIVE"
  enable_shielded_nodes       = true
  enable_intranode_visibility = true

  release_channel {
    channel = var.release_channel
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  fleet {
    project = var.project_id
  }

  resource_labels = merge(local.common_labels, {
    role   = "worker"
    worker = each.key
  })

  ip_allocation_policy {
    cluster_secondary_range_name  = each.value.cluster_secondary_range_name
    services_secondary_range_name = each.value.services_secondary_range_name
  }

  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = var.enable_private_endpoint
    master_ipv4_cidr_block  = each.value.master_ipv4_cidr_block

    master_global_access_config {
      enabled = true
    }
  }

  dynamic "master_authorized_networks_config" {
    for_each = var.enable_private_endpoint ? [] : [1]
    content {
      dynamic "cidr_blocks" {
        for_each = var.master_authorized_networks
        content {
          cidr_block   = cidr_blocks.value.cidr_block
          display_name = cidr_blocks.value.display_name
        }
      }
    }
  }

  addons_config {
    gce_persistent_disk_csi_driver_config {
      enabled = true
    }
  }

  secret_manager_config {
    enabled = true
    rotation_config {
      enabled           = true
      rotation_interval = "300s"
    }
  }

  secret_sync_config {
    enabled = true
    rotation_config {
      enabled           = true
      rotation_interval = "300s"
    }
  }

  security_posture_config {
    mode               = "BASIC"
    vulnerability_mode = "VULNERABILITY_BASIC"
  }

  monitoring_config {
    enable_components = [
      "APISERVER",
      "CONTROLLER_MANAGER",
      "DAEMONSET",
      "DEPLOYMENT",
      "HPA",
      "KUBELET",
      "POD",
      "SCHEDULER",
      "STATEFULSET",
      "STORAGE",
      "SYSTEM_COMPONENTS",
    ]
    managed_prometheus {
      enabled = true
    }
  }

  maintenance_policy {
    recurring_window {
      start_time = "2026-01-04T09:00:00Z"
      end_time   = "2026-01-04T13:00:00Z"
      recurrence = "FREQ=WEEKLY;BYDAY=SU"
    }
  }

  lifecycle {
    precondition {
      condition     = var.enable_private_endpoint || length(var.master_authorized_networks) > 0
      error_message = "Provide master_authorized_networks or enable the private control-plane endpoint."
    }
  }

  depends_on = [google_project_service.required]
}

resource "google_container_node_pool" "worker_system" {
  for_each = var.worker_clusters

  project  = var.project_id
  name     = "system"
  location = google_container_cluster.worker[each.key].location
  cluster  = google_container_cluster.worker[each.key].name

  initial_node_count = each.value.system_min_nodes

  autoscaling {
    min_node_count = each.value.system_min_nodes
    max_node_count = each.value.system_max_nodes
  }

  node_config {
    machine_type    = each.value.system_machine_type
    image_type      = "COS_CONTAINERD"
    service_account = google_service_account.worker_nodes[each.key].email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]
    labels = {
      "tpu-ci.google.com/role" = "system"
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    shielded_instance_config {
      enable_integrity_monitoring = true
      enable_secure_boot          = true
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
  }

  lifecycle {
    ignore_changes = [initial_node_count]
  }
}

resource "google_container_node_pool" "worker_tpu" {
  for_each = local.tpu_node_pools

  project  = var.project_id
  name     = each.value.profile_name
  location = google_container_cluster.worker[each.value.worker_name].location
  cluster  = google_container_cluster.worker[each.value.worker_name].name

  initial_node_count = each.value.min_nodes

  autoscaling {
    min_node_count  = each.value.min_nodes
    max_node_count  = each.value.max_nodes
    location_policy = each.value.reservation_name == null ? "BALANCED" : "ANY"
  }

  placement_policy {
    type         = "COMPACT"
    tpu_topology = each.value.topology
  }

  node_config {
    machine_type    = each.value.machine_type
    image_type      = "COS_CONTAINERD"
    service_account = google_service_account.worker_nodes[each.value.worker_name].email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]
    labels = {
      "tpu-ci.google.com/profile"        = each.value.profile_name
      "tpu-ci.google.com/chips-per-node" = tostring(each.value.chips_per_node)
    }
    resource_labels = merge(local.common_labels, {
      profile = each.value.profile_name
      worker  = each.value.worker_name
    })

    dynamic "reservation_affinity" {
      for_each = each.value.reservation_name == null ? [] : [each.value.reservation_name]
      content {
        consume_reservation_type = "SPECIFIC_RESERVATION"
        key                      = "compute.googleapis.com/reservation-name"
        values                   = [reservation_affinity.value]
      }
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    shielded_instance_config {
      enable_integrity_monitoring = true
      enable_secure_boot          = true
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  # Reserved TPU pools often cannot create a surge node. This setting accepts
  # one unavailable node during an upgrade; schedule maintenance accordingly.
  upgrade_settings {
    max_surge       = 0
    max_unavailable = 1
  }

  lifecycle {
    ignore_changes = [initial_node_count]
  }
}
