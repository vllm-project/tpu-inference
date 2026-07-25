terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 7.0, < 8.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.manager_region
}

data "google_project" "current" {
  project_id = var.project_id
}
