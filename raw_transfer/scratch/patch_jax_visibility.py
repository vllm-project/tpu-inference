file_path = '/mnt/disks/jcgu/code/ullm/remote1/xla/third_party/py/jax/jaxlib/jax.bzl'
with open(file_path, 'r') as f:
    content = f.read()

# Replace jax_visibility implementation
old_vis = '''def jax_visibility(_target):
    """Returns the additional Bazel visibilities for `target`."""
    return [
        "//jax:__subpackages__",
        "//jax/jaxlib:__subpackages__",
    ]'''

new_vis = '''def jax_visibility(_target):
    """Returns the additional Bazel visibilities for `target`."""
    return ["//visibility:public"]'''

content = content.replace(old_vis, new_vis)

with open(file_path, 'w') as f:
    f.write(content)

print("Patched jax_visibility in jax.bzl")
