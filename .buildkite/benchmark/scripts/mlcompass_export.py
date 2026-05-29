import sys

# sys.argv[0] is always the script name itself
print(f"Script name: {sys.argv[0]}")

# Capture the rest of the arguments
if len(sys.argv) > 1:
    print(f"Arguments passed: {sys.argv[1:]}")

print('MLCompass export file called')
