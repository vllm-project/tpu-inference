terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 7.0, < 8.0"
    }
  }
}

# Fleet host and manager cluster.
provider "google" {
  project = var.project_id
  region  = var.manager_region
}

# Worker clusters, their TPU node pools, and the reservation they consume.
# Defaults to the same project, so a single-project deployment needs no extra
# configuration; set worker_project to split them.
provider "google" {
  alias   = "worker"
  project = local.worker_project
}

data "google_project" "current" {
  project_id = var.project_id
}

data "google_project" "worker" {
  provider   = google.worker
  project_id = local.worker_project
}
