import re

file_path = '/mnt/disks/jcgu/code/ullm/remote1/xla/third_party/py/jax/jaxlib/BUILD'
with open(file_path, 'r') as f:
    content = f.read()

# Revert previous includes if it was added
content = content.replace('includes = [".."],\n    ', '')

# Find py_client rule and add -I flag to copts
pattern = re.compile(
    r'(cc_library\(\s*name = "py_client",.*?copts = \[)(.*?)( \],)',
    re.DOTALL
)

content = pattern.sub(r'\1\2        "-Ithird_party/py/jax",\n\3', content)

with open(file_path, 'w') as f:
    f.write(content)

print("Added -I flag to py_client copts in BUILD")
