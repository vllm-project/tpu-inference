import sys
sys.path.append('/mnt/disks/jcgu/code/ullm/remote1/xla/bazel-bin/xla/python/raw_transfer')
try:
    import raw_transfer
    print("SUCCESSfully imported raw_transfer!")
    print(raw_transfer)
except Exception as e:
    print("FAILED to import raw_transfer:")
    print(e)
