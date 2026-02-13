# README Automation Walkthrough

## Feature Overview
The **README Automation** feature ensures the TPU Support Matrix in the `README.md` is always up-to-date. It uses a Python script to parse CSV files and a GitHub Action to run the script automatically on every push.

## Changes Created
### 1. Python Script
- **File**: [update_readme.py](file:///Users/chaowan/Documents/vllm%20test%20project/update_readme.py)
- **Purpose**: Reads CSVs from `support_matrices/` and updates corresponding sections in `README.md`.
- **Key Logic**:
    - Uses markers like `<!-- START: model_support -->` to locate update zones.
    - Generates Markdown tables dynamically.

### 2. GitHub Workflow
- **File**: [.github/workflows/update_readme.yml](file:///Users/chaowan/Documents/vllm%20test%20project/.github/workflows/update_readme.yml)
- **Trigger**:
    - **Push**: Runs when CSV files in `support_matrices/` are modified.
- **Action**: Commits changes back to the repository using `github-actions[bot]`.

## How to Verify
### Local Verification
1.  Modify a CSV file in `support_matrices/` (e.g., change a status to `❌`).
2.  Run the script:
    ```bash
    python3 update_readme.py
    ```
3.  Check `README.md`:
    ```bash
    git diff README.md
    ```
    *You should see the table updated with your change.*

### GitHub Action Verification
1.  Push a change to any CSV file.
2.  Go to the **Actions** tab in GitHub.
3.  Verify the "Update README Tables" workflow runs and commits the change.
