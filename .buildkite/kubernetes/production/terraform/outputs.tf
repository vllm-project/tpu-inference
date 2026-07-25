output "manager_cluster" {
  description = "Manager cluster identity used by the platform deployment stack."
  value = {
    name     = google_container_cluster.manager.name
    location = google_container_cluster.manager.location
    endpoint = google_container_cluster.manager.endpoint
  }
  sensitive = true
}

output "worker_clusters" {
  description = "Worker cluster identities and logical TPU profiles."
  value = {
    for name, cluster in google_container_cluster.worker : name => {
      name         = cluster.name
      location     = cluster.location
      endpoint     = cluster.endpoint
      tpu_profiles = keys(var.worker_clusters[name].tpu_pools)
    }
  }
  sensitive = true
}

output "artifact_registry_repository" {
  description = "Regional repositories. Promote one tested digest to every region; do not rebuild per region."
  value = {
    for region, repository in google_artifact_registry_repository.ci_images :
    region => repository.name
  }
}

output "buildkite_secret_id" {
  value = google_secret_manager_secret.buildkite_agent_token.secret_id
}

output "next_steps" {
  value = [
    "Add the Buildkite token as a Secret Manager version outside Terraform.",
    "Install the platform stack (Kueue, ClusterProfile auth plugin, secret sync, policies, and Agent Stack).",
    "Verify Fleet-generated ClusterProfiles before creating MultiKueueCluster objects.",
  ]
}
