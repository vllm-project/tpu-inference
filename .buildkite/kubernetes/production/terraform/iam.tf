resource "google_service_account" "manager_nodes" {
  project      = var.project_id
  account_id   = substr("${var.name_prefix}-manager-nodes", 0, 30)
  display_name = "${var.name_prefix} manager GKE nodes"
}

resource "google_service_account" "worker_nodes" {
  provider = google.worker
  for_each = var.worker_clusters

  project      = local.worker_project
  account_id   = substr(replace("${var.name_prefix}-${each.key}-nodes", "_", "-"), 0, 30)
  display_name = "${var.name_prefix} ${each.key} GKE nodes"
}

resource "google_project_iam_member" "manager_node_roles" {
  for_each = local.node_service_account_roles

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.manager_nodes.email}"
}

resource "google_project_iam_member" "worker_node_roles" {
  provider = google.worker
  for_each = local.worker_node_role_bindings

  project = local.worker_project
  role    = each.value.role
  member  = "serviceAccount:${google_service_account.worker_nodes[each.value.worker_name].email}"
}

# This follows the GKE Fleet ClusterProfile integration. The KSA name is unique
# to the manager cluster; using the default Kueue KSA name would grant the same
# project roles to worker controllers because GKE workload identity subjects do
# not include a cluster name. Review whether IAM Conditions can scope these
# roles further once the exact membership names are stable.
resource "google_project_iam_member" "kueue_clusterprofile_access" {
  for_each = toset([
    "roles/container.developer",
    "roles/gkehub.gatewayEditor",
  ])

  project = var.project_id
  role    = each.value
  member  = local.kueue_manager_principal

  depends_on = [google_container_cluster.manager]
}

# Connect Gateway authorizes in the fleet host project, but the call still lands
# on a cluster in the worker project, so the manager identity needs GKE access
# there too. Only when the projects differ; otherwise the binding above already
# covers it and a duplicate would fight itself on every plan.
resource "google_project_iam_member" "kueue_worker_cluster_access" {
  provider = google.worker
  count    = local.cross_project ? 1 : 0

  project = local.worker_project
  role    = "roles/container.developer"
  member  = local.kueue_manager_principal

  depends_on = [google_container_cluster.manager]
}

# The test image is published to the manager project's Artifact Registry. Worker
# nodes in another project cannot pull it from their own project-level
# artifactregistry.reader grant. Without this the copied Job's Pod sits in
# ImagePullBackOff, which Buildkite surfaces only as a job that never starts.
resource "google_artifact_registry_repository_iam_member" "worker_node_pull" {
  for_each = local.cross_project ? local.worker_registry_pull_bindings : {}

  project    = var.project_id
  location   = each.value.location
  repository = google_artifact_registry_repository.ci_images[each.value.location].name
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_service_account.worker_nodes[each.value.worker_name].email}"
}
