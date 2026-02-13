# Auto Mission Board Implementation Plan

## Goal
Automate the creation of "Good First Issues" to drive community engagement by identifying gaps (missing tests/benchmarks) in the TPU support matrices.

## User Review Required
> [!IMPORTANT]
> **GitHub Token Scope**: The GitHub Action will need `issues: write` permission.
> **Rate Limiting**: The script should be careful not to spam the repo if many items are unverified at once. We might want to limit the number of issues created per run (e.g., max 5).

## Proposed Changes

### Strategy: Target "Main" Support Metrics
We will target the **Main Support Matrices** (the ones currently in `support_matrices/*.csv`).

**Reasoning:**
- **Stability**: "Good First Issues" should be stable tasks. Nightly failures might be transient flakes or temporary regressions.
- **Clarity**: The main matrix represents the "official" status. If something is marked `unverified` there, it is a known gap that needs filling.
- **Workflow**: Contributors will verify against the stable release or main branch, which aligns with these matrices.

### Scripts
#### [NEW] [scripts/auto_mission_board.py](file:///Users/chaowan/Documents/vllm test project/scripts/auto_mission_board.py)
A new Python script that:
1.  Imports `CSV_MAP` from `update_readme.py` (or redefines it if import is messy).
2.  Iterates through each CSV.
3.  Identifies cells with `unverified` or `❌`.
4.  Generates a candidate Issue Title (e.g., `[Mission] Verify {Feature} for {Model}`).
5.  Checks if an issue with this title already exists (using `gh` CLI or PyGithub).
6.  Creates the issue if missing, adding label `good first issue`.

### workflows
#### [NEW] [.github/workflows/auto_mission_board.yml](file:///Users/chaowan/Documents/vllm test project/.github/workflows/auto_mission_board.yml)
A GitHub Action workflow that:
1.  Runs on a schedule (e.g., daily) and manually (`workflow_dispatch`).
2.  Checkout code.
3.  Sets up Python.
4.  Runs `scripts/auto_mission_board.py`.
5.  Uses `GITHUB_TOKEN` for authentication.

## Verification Plan

### Automated Tests
- **Dry Run**: Run the script locally with a `--dry-run` flag to print issues it *would* create without actually calling the GitHub API.
    ```bash
    python3 scripts/auto_mission_board.py --dry-run
    ```

### Manual Verification
- **Test Workflow**: Push the workflow to a test branch.
- **Trigger**: Manually trigger the workflow in the GitHub Actions tab.
- **Verify**: Check that issues are created (or printed in logs if verifying logic only).
