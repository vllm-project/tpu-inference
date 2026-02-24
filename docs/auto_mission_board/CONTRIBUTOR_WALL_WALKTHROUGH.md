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

### README Integration
The `README.md` was updated with:
1.  A **Contribution Type Legend** explaining the emojis.
2.  Placeholder HTML comments (`<!-- START: contributors -->` and `<!-- END: contributors -->`) where the python script injects the generated grid.

### Automated Updates (`update_contributors.yml`)
A GitHub Action workflow guarantees the wall is never out of date. It triggers:
1.  Automatically on pushes to `main` and pushes to `auto-leaderboard`.
2.  Automatically on a weekly schedule (Sundays at midnight).
3.  Manually via the "Run workflow" button.
The action runs the python script and commits any changes directly back to the repo.

## Verification
- Local testing confirmed the script successfully finds non-code contributors (over 100 people found compared to ~20 committers).
- GitHub Actions testing confirmed the workflow has permission to commit and push changes back to the repository.
- The UI matches the interactive, standard `all-contributors` layout.
