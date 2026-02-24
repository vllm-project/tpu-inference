# Auto Mission Board Walkthrough

## Feature Overview
The **Auto Mission Board** automates the creation of "Good First Issues" to drive community engagement. It identifies missing support metrics (marked as `unverified` or `❌`) in the project's CSV support matrices and generates GitHub issues for them.

## Changes Created
### 1. Python Script
- **File**: [scripts/auto_mission_board.py](file:///Users/chaowan/Documents/vllm%20test%20project/scripts/auto_mission_board.py)
- **Purpose**: Parses CSV files in `support_matrices/`, identifies unverified items, and uses the GitHub API to create issues.
- **Key Logic**:
    - Limits creation to **5 issues per run** to avoid spam.
    - Checks for existing issues to avoid duplicates.
    - Adds labels: `good first issue`, `contribution-welcome`, `auto-generated`.

### 2. GitHub Workflow
- **File**: [.github/workflows/auto_mission_board.yml](file:///Users/chaowan/Documents/vllm%20test%20project/.github/workflows/auto_mission_board.yml)
- **Trigger**:
    - **Scheduled**: Every Monday at 00:00 UTC.
    - **Manual**: Via the "Run workflow" button in GitHub Actions.
- **Permissions**: Grants `issues: write` to the `GITHUB_TOKEN`.

## Verification Results
### Automated Dry-Run
Verified locally that the script correctly identifies targets without calling the API:
```bash
python3 scripts/auto_mission_board.py --dry-run
```
*Output confirmed it found "unverified" status for models like `moonshotai/Kimi-K2-Thinking`.*

### Live GitHub Action Test
- **Run ID**: [Manual Trigger]
- **Result**: Successfully created 5 issues on the repository.
- **Cleanup**: Removed the temporary `push` trigger to ensure it only runs on schedule or manually going forward.

## How to Manage
- **Modify Frequency**: Edit the `cron` schedule in `.github/workflows/auto_mission_board.yml`.
- **Change Target Metrics**: Update `CSV_MAP` in `scripts/auto_mission_board.py`.
