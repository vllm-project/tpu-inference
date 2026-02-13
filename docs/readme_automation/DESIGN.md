# README Automation Implementation Plan

## Goal
Automate the maintenance of the TPU Support Matrix in `README.md` to ensure it always reflects the latest status from the CSV source of truth.

## Proposed Changes

### Scripts
#### [NEW] [update_readme.py](file:///Users/chaowan/Documents/vllm test project/update_readme.py)
A Python script that:
1.  Reads CSV files from `support_matrices/`.
2.  Generates Markdown tables from the CSV data.
3.  Injects the tables into `README.md` between specific HTML comments (e.g., `<!-- START: model_support -->`).

### Workflows
#### [NEW] [.github/workflows/update_readme.yml](file:///Users/chaowan/Documents/vllm test project/.github/workflows/update_readme.yml)
A GitHub Action workflow that:
1.  Triggers on push to main branches and when CSV files change.
2.  Runs the `update_readme.py` script.
3.  Commits and pushes changes back to the repo if the README was updated.

## Verification Plan

### Automated Tests
- **GitHub Action**: Verify the action runs successfully on push.
- **Content Check**: Verify `README.md` is updated with correct table data after the action runs.

### Manual Verification
- **Local Run**: Run `python3 update_readme.py` locally and check `git diff` to see if `README.md` is updated correctly.
