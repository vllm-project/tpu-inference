import re

file_path = '/mnt/disks/jcgu/code/ullm/remote1/xla/third_party/py/jax/jaxlib/py_host_callback.cc'
with open(file_path, 'r') as f:
    content = f.read()

# Remove include
content = content.replace('#include "jaxlib/py_host_callback.pb.h"', '// Removed for mock build')

# Stub out Serialize implementation
pattern = re.compile(
    r'(absl::StatusOr<std::string> PyHostSendAndRecvLoadedHostCallback::Serialize\(\)\s*const\s*\{).*?(return xla_host_callback_proto\.SerializeAsString\(\);\s*\})',
    re.DOTALL
)

content = pattern.sub(r'\1\n  return absl::UnimplementedError("Serialization not supported");\n}', content)

with open(file_path, 'w') as f:
    f.write(content)

print("Patched py_host_callback.cc")

# Now patch BUILD file
build_path = '/mnt/disks/jcgu/code/ullm/remote1/xla/third_party/py/jax/jaxlib/BUILD'
with open(build_path, 'r') as f:
    build_content = f.read()

# Use regex to replace the whole line containing the bad target, regardless of how many comment chars it has
build_content = re.sub(r'.*":py_host_callback_cc_proto",.*', '        # ":py_host_callback_cc_proto",', build_content)

with open(build_path, 'w') as f:
    f.write(build_content)

print("Patched BUILD file")
