resource "google_project_service" "required" {
  for_each = local.required_services

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "ci_images" {
  for_each = local.artifact_registry_regions

  project       = var.project_id
  location      = each.value
  repository_id = "${var.name_prefix}-images"
  description   = "Buildkite TPU CI images and ClusterProfile auth plugin"
  format        = "DOCKER"
  labels        = local.common_labels

  cleanup_policies {
    id     = "delete-untagged"
    action = "DELETE"

    condition {
      tag_state  = "UNTAGGED"
      older_than = "604800s"
    }
  }

  depends_on = [google_project_service.required]
}

resource "google_secret_manager_secret" "buildkite_agent_token" {
  project   = var.project_id
  secret_id = var.buildkite_secret_id
  labels    = local.common_labels

  replication {
    auto {}
  }

  deletion_protection = var.deletion_protection

  depends_on = [google_project_service.required]
}

# The secret value is intentionally absent. Creating google_secret_manager_secret_version
# would place the Buildkite token in Terraform state. Add and rotate versions through an
# approved secret-delivery workflow, then synchronize them into each cluster.
resource "google_secret_manager_secret_iam_member" "secret_sync" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.buildkite_agent_token.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = local.secret_sync_principal
}
