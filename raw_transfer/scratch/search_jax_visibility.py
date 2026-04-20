for i, line in enumerate(open('/mnt/disks/jcgu/code/ullm/remote1/xla/third_party/py/jax/jaxlib/jax.bzl')):
    if 'def jax_visibility' in line:
        print(i, line)
