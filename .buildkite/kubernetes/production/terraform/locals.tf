locals {
  worker_project = coalesce(var.worker_project, var.project_id)
  cross_project  = local.worker_project != var.project_id

  common_labels = merge(var.labels, {
    managed_by = "terraform"
    component  = "buildkite-tpu-ci"
  })

  required_services = toset([
    "artifactregistry.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "compute.googleapis.com",
    "connectgateway.googleapis.com",
    "container.googleapis.com",
    "gkehub.googleapis.com",
    "iamcredentials.googleapis.com",
    "secretmanager.googleapis.com",
  ])

  worker_regions = toset([
    for worker in values(var.worker_clusters) :
    regexreplace(worker.location, "-[a-z]$", "")
  ])
  artifact_registry_regions = setunion(toset([var.manager_region]), local.worker_regions)

  node_service_account_roles = toset([
    "roles/artifactregistry.reader",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/monitoring.viewer",
    "roles/stackdriver.resourceMetadata.writer",
  ])

  worker_node_role_bindings = {
    for item in flatten([
      for worker_name in keys(var.worker_clusters) : [
        for role in local.node_service_account_roles : {
          key         = "${worker_name}/${role}"
          worker_name = worker_name
          role        = role
        }
      ]
    ]) : item.key => item
  }

  worker_registry_pull_bindings = {
    for item in flatten([
      for worker_name in keys(var.worker_clusters) : [
        for location in local.artifact_registry_regions : {
          key         = "${worker_name}/${location}"
          worker_name = worker_name
          location    = location
        }
      ]
    ]) : item.key => item
  }

  tpu_node_pools = {
    for item in flatten([
      for worker_name, worker in var.worker_clusters : [
        for profile_name, pool in worker.tpu_pools : {
          key              = "${worker_name}/${profile_name}"
          worker_name      = worker_name
          location         = worker.location
          profile_name     = profile_name
          machine_type     = pool.machine_type
          topology         = pool.topology
          chips_per_node   = pool.chips_per_node
          min_nodes        = pool.min_nodes
          max_nodes        = pool.max_nodes
          reservation_name = try(pool.reservation_name, null)
        }
      ]
    ]) : item.key => item
  }

  secret_sync_principal   = "principal://iam.googleapis.com/projects/${data.google_project.current.number}/locations/global/workloadIdentityPools/${var.project_id}.svc.id.goog/subject/ns/${var.secret_sync_namespace}/sa/${var.secret_sync_service_account}"
  kueue_manager_principal = "principal://iam.googleapis.com/projects/${data.google_project.current.number}/locations/global/workloadIdentityPools/${var.project_id}.svc.id.goog/subject/ns/kueue-system/sa/${var.kueue_fleet_service_account}"
}
