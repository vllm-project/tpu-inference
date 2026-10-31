# Compilation-cache access for worker Pods.
#
# The golden PVC gives each job a warm read-only base. These grants add the
# return path: a job publishes the entries it compiled, so a shape the golden
# does not cover is compiled once for the fleet instead of once per build.
#
# The bucket is shared with the bare-metal pipeline (run_in_docker.sh writes the
# same prefix) and long predates this stack, so it is referenced rather than
# managed. Importing it would put a live bare-metal dependency at the mercy of
# this state file.
data "google_storage_bucket" "cache" {
  name = var.cache_bucket
}

locals {
  cache_prefix_condition = "resource.name.startsWith('projects/_/buckets/${var.cache_bucket}/objects/${var.cache_prefix}/')"

  # Direct Workload Identity Federation principals. No Google service account
  # and no exported key: the binding names the Kubernetes ServiceAccount itself.
  # MultiKueue copies the PodSpec verbatim, so these KSA names must exist in
  # every worker cluster -- see ../cache/serviceaccounts.yaml.
  cache_writer_principal = "principal://iam.googleapis.com/projects/${data.google_project.worker.number}/locations/global/workloadIdentityPools/${local.worker_project}.svc.id.goog/subject/ns/${var.buildkite_namespace}/sa/${var.cache_writer_service_account}"
  cache_reader_principal = "principal://iam.googleapis.com/projects/${data.google_project.worker.number}/locations/global/workloadIdentityPools/${local.worker_project}.svc.id.goog/subject/ns/${var.buildkite_namespace}/sa/${var.cache_reader_service_account}"
}

# TPU test Pods: read and write, but only under the cache prefix. A test that
# is compromised or simply wrong cannot reach anything else in the bucket.
resource "google_storage_bucket_iam_member" "cache_writer" {
  bucket = data.google_storage_bucket.cache.name
  role   = "roles/storage.objectUser"
  member = local.cache_writer_principal

  condition {
    title       = "jax-cache-prefix-only"
    description = "Buildkite TPU Pods may only touch the compilation cache prefix."
    expression  = local.cache_prefix_condition
  }
}

# Golden-refresh CronJob: read only. It writes to the PVC, never to GCS, so a
# bug there cannot rewrite entries other jobs depend on.
resource "google_storage_bucket_iam_member" "cache_reader" {
  bucket = data.google_storage_bucket.cache.name
  role   = "roles/storage.objectViewer"
  member = local.cache_reader_principal

  condition {
    title       = "jax-cache-prefix-only"
    description = "Golden refresh may only read the compilation cache prefix."
    expression  = local.cache_prefix_condition
  }
}
