#!/bin/bash

# This script substitutes the EngineCoreProc with DisaggEngineCoreProc in core.py,
# while preserving the line's original indentation.

# Define the target file
TARGET_FILE="/workspace/vllm/vllm/v1/engine/core.py"

# --- Validation ---
# Check if the target file exists
if [ ! -f "$TARGET_FILE" ]; then
    echo "Error: The file '$TARGET_FILE' was not found in the current directory."
    exit 1
fi

# --- Line Definitions ---
# The line to find. We escape special characters for grep and sed.
# We look for lines that may start with whitespace and then the specific code.
LINE_TO_FIND="engine_core = EngineCoreProc(\*args, \*\*kwargs)"
# The new content to be inserted.
REPLACEMENT_LINE="from tpu_commons.core.core_tpu import DisaggEngineCoreProc; engine_core = DisaggEngineCoreProc(*args, **kwargs)"

# --- Substitution Logic ---
# Use sed to find and replace the line.
# - `s/pattern/replacement/` is the substitution command.
# - `^\([[:space:]]*\)`: Matches and captures any whitespace characters (the indentation) at the beginning of the line.
# - `\1`: In the replacement string, this pastes the captured indentation back.
# - `-i.bak`: This option edits the file in-place and creates a backup of the original file with a .bak extension.
sed -i.bak "s/^\([[:space:]]*\)$LINE_TO_FIND/\1$REPLACEMENT_LINE/" "$TARGET_FILE"

# --- Confirmation ---
# Check if the substitution was successful by seeing if the file changed.
# If `sed` made a change, the original and backup will be different.
if cmp -s "$TARGET_FILE" "$TARGET_FILE.bak"; then
    echo "Could not find the target line in '$TARGET_FILE'. No changes were made."
    # Clean up the unnecessary backup file
    rm "$TARGET_FILE.bak"
    exit 1
else
    echo "Successfully updated '$TARGET_FILE'."
    echo "A backup of the original has been saved as '$TARGET_FILE.bak'."

    echo "Delete the '$TARGET_FILE.bak'."
    rm "$TARGET_FILE.bak"
fi

exit 0
