# Inclusive Contributor Wall Walkthrough

## Feature Overview
The **Inclusive Contributor Wall** replaces static lists with a continuously-updating, visual grid of contributors on the `README.md`. It automatically recognizes and celebrates all types of contributions (Code, Issues, Reviews) exactly like the popular `all-contributors` bot, but without requiring any external apps or manual slash-commands.

## Implementation Details

### Automated Discovery (`scripts/update_contributors.py`)
This script uses the GitHub API to dynamically identify all project contributors:
1.  **Code (💻)**: Fetching commit history.
2.  **Issues (🐛)**: Fetching authors of all opened issues (excluding PRs).
3.  **Reviews (👀)**: Fetching authors of PR reviews.

It builds a unique profile for each user, aggregates their contribution types, and generates an HTML table matching the established `all-contributors` specification.

### Ranking Algorithm
The script automatically builds an Effort-Based Leaderboard by calculating a `Total Effort Score` for each user:
- `Effort = Commit Count + Unique Issue Count + PR Review Count`
Contributors are strictly sorted from highest effort to lowest effort. If two contributors have the same effort, they are sorted alphabetically.

### Rendering & UI
The `README.md` layout integrates the generated HTML to ensure a premium look:
1.  **Contribution Type Legend** (🌟) explaining the emojis above the grid.
2.  **Collapsible View**: The first **3 rows** (21 contributors) are shown by default to save vertical space, with the remainder hidden behind a expandable `<details>` tag.
3.  **Strict Image Bounds**: Avatars use strict inline CSS (`width="100" style="max-width: 100px; width: 100%; border-radius: 20px;"`) to ensure they remain perfectly uniform 100px squares regardless of column stretching or original aspect ratios.

### Automated Updates (`update_contributors.yml`)
A GitHub Action workflow guarantees the wall is never out of date. It triggers:
1.  Automatically on pushes to `main` and pushes to `auto-leaderboard`.
2.  Automatically on a weekly schedule (Sundays at midnight).
3.  Manually via the "Run workflow" button.
The action runs the python script and commits any changes directly back to the repo.

## Verification
- Local testing confirmed the script successfully finds non-code contributors (over 100 people found compared to ~20 committers).
- Validated that the first **3 rows** (21 contributors) are shown by default, with the rest collapsible behind a "View all" click.
- GitHub Actions testing confirmed the workflow has permission to commit and push changes back to the repository.
- The UI matches the interactive, standard `all-contributors` layout.
