with open('/mnt/disks/jcgu/code/ullm/remote1/xla/jax/BUILD.bazel', 'a') as f:
    f.write('\npackage_group(name = "internal", packages = ["//..."])\n')
