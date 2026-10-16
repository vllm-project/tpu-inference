variable "project_id" {
  description = "Google Cloud project that owns the Fleet and GKE clusters."
  type        = string
}

variable "worker_project" {
  description = <<-EOT
    Project that owns the worker clusters, their TPU node pools, and the TPU
    reservation those pools consume. Leave null when manager and workers share
    one project.

    The split exists because a reservation is only consumable from the project
    that holds it, so the worker cluster has to live beside it. The Fleet, the
    manager cluster, and Artifact Registry stay in project_id.
  EOT
  type        = string
  default     = null
}

variable "name_prefix" {
  description = "Short prefix used for cluster, service-account, and registry names."
  type        = string
  default     = "tpu-ci"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,15}$", var.name_prefix))
    error_message = "name_prefix must be 2-16 lowercase letters, digits, or hyphens and start with a letter."
  }
}

variable "network" {
  description = "VPC network self-link or name. NAT and private Google access must be managed by the network stack."
  type        = string
}

variable "manager_region" {
  description = "Region for the highly available manager cluster."
  type        = string
  default     = "us-central1"
}

variable "manager_zones" {
  description = "Zones used by the manager cluster's CPU system node pool."
  type        = list(string)
  default     = ["us-central1-a", "us-central1-b", "us-central1-c"]
}

variable "manager_subnetwork" {
  description = "Subnetwork self-link or name for the manager cluster."
  type        = string
}

variable "manager_cluster_secondary_range_name" {
  description = "Existing secondary range for manager-cluster Pods."
  type        = string
}

variable "manager_services_secondary_range_name" {
  description = "Existing secondary range for manager-cluster Services."
  type        = string
}

variable "manager_master_ipv4_cidr_block" {
  description = "Private /28 CIDR for the manager control plane."
  type        = string
}

variable "manager_system_machine_type" {
  description = "Machine type for Agent Stack and platform controllers."
  type        = string
  default     = "e2-standard-8"
}

variable "manager_system_min_nodes" {
  type    = number
  default = 3
}

variable "manager_system_max_nodes" {
  type    = number
  default = 6
}

variable "worker_clusters" {
  description = <<-EOT
    Zonal Standard GKE workers. Each TPU pool maps one logical profile to a
    machine type, physical topology, reservation, and autoscaling boundary.
  EOT
  type = map(object({
    location                      = string
    subnetwork                    = string
    cluster_secondary_range_name  = string
    services_secondary_range_name = string
    master_ipv4_cidr_block        = string
    system_machine_type           = optional(string, "e2-standard-4")
    system_min_nodes              = optional(number, 1)
    system_max_nodes              = optional(number, 3)
    tpu_pools = map(object({
      machine_type     = string
      topology         = string
      chips_per_node   = number
      min_nodes        = optional(number, 0)
      max_nodes        = number
      reservation_name = optional(string)
    }))
  }))

  validation {
    condition = alltrue(flatten([
      for worker in values(var.worker_clusters) : [
        for pool in values(worker.tpu_pools) :
        pool.max_nodes >= pool.min_nodes && pool.chips_per_node > 0
      ]
    ]))
    error_message = "Every TPU pool must have max_nodes >= min_nodes and chips_per_node > 0."
  }
}

variable "master_authorized_networks" {
  description = "CIDRs allowed to reach public control-plane endpoints. Empty is rejected unless private endpoints are enabled."
  type = list(object({
    cidr_block   = string
    display_name = string
  }))
  default = []
}

variable "enable_private_endpoint" {
  description = "Make GKE control planes private-only. Requires private Terraform/operations connectivity."
  type        = bool
  default     = false
}

variable "release_channel" {
  description = "GKE release channel."
  type        = string
  default     = "REGULAR"

  validation {
    condition     = contains(["RAPID", "REGULAR", "STABLE"], var.release_channel)
    error_message = "release_channel must be RAPID, REGULAR, or STABLE."
  }
}

variable "deletion_protection" {
  description = "Protect clusters and secret containers from accidental Terraform deletion."
  type        = bool
  default     = true
}

variable "labels" {
  description = "Additional Google Cloud resource labels."
  type        = map(string)
  default = {
    environment = "production"
    workload    = "tpu-ci"
  }
}

variable "buildkite_secret_id" {
  description = "Secret Manager container for the Buildkite agent token. Terraform never creates a secret version."
  type        = string
  default     = "buildkite-tpu-ci-agent-token"
}

variable "secret_sync_namespace" {
  description = "Namespace of the KSA allowed to synchronize the Buildkite token from Secret Manager."
  type        = string
  default     = "external-secrets"
}

variable "secret_sync_service_account" {
  description = "KSA allowed to synchronize the Buildkite token. Install it separately through the platform stack."
  type        = string
  default     = "external-secrets"
}

variable "kueue_fleet_service_account" {
  description = "Manager-only KSA created by the multikueue-fleet Helm release for Fleet/Connect Gateway access. Do not reuse it on workers."
  type        = string
  default     = "multikueue-fleet-controller-manager"
}

variable "cache_bucket" {
  description = <<-EOT
    Existing bucket holding the shared JAX/XLA compilation cache. Terraform
    manages only its IAM; the bucket itself is shared with the bare-metal
    pipeline and is not owned by this stack.
  EOT
  type        = string
  default     = "ullm-ci-cache"
}

variable "cache_prefix" {
  description = "Object prefix within cache_bucket. Grants are conditioned on it, so Pods cannot reach the rest of the bucket."
  type        = string
  default     = "jax_cache"
}

variable "buildkite_namespace" {
  description = "Namespace holding Agent Stack Jobs and the cache ServiceAccounts on every cluster."
  type        = string
  default     = "buildkite"
}

variable "cache_writer_service_account" {
  description = "KSA used by TPU test Pods to publish newly compiled cache entries. Must exist in every worker cluster under this name."
  type        = string
  default     = "buildkite-cache"
}

variable "cache_reader_service_account" {
  description = "KSA used by the golden-refresh CronJob. Read-only on GCS."
  type        = string
  default     = "buildkite-cache-refresh"
}
