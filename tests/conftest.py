import os
import site
import sys


def pytest_configure(config):
    if os.environ.get("FORCE_USE_SITE_PACKAGES", "0") != "1":
        print("[DEBUG] Running in Repo Mode. Using local source code.",
              file=sys.stderr)
        return

    print("\n[DEBUG] conftest.py is fixing sys.path...", file=sys.stderr)

    local_vllm_path = os.path.abspath("/workspace/vllm")

    removed = False
    for path in list(sys.path):
        if local_vllm_path in os.path.abspath(path):
            sys.path.remove(path)
            removed = True
            print(f"[DEBUG] Removed local source from path: {path}",
                  file=sys.stderr)

    site_packages = site.getsitepackages()
    for sp in site_packages:
        if sp not in sys.path:
            sys.path.insert(0, sp)
        else:
            sys.path.remove(sp)
            sys.path.insert(0, sp)

    if removed:
        print("Path fix applied successfully.", file=sys.stderr)
    else:
        print("Local vllm path not found in sys.path (maybe already clean?)",
              file=sys.stderr)
