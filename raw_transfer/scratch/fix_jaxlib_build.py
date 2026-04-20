with open('/mnt/disks/jcgu/code/ullm/remote1/jax/jaxlib/jax.bzl', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.strip() == '"//jax/jaxlib:__subpackages__",':
        new_lines.append('        "//jax/jaxlib:__subpackages__",\n')
        new_lines.append('    ]\n')
    else:
        new_lines.append(line)

with open('/mnt/disks/jcgu/code/ullm/remote1/jax/jaxlib/jax.bzl', 'w') as f:
    f.writelines(new_lines)
