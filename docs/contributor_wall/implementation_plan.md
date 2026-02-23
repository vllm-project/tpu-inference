# Inclusive Contributor Wall Implementation Plan (All-Contributors Clone)

## Goal
Create an "Inclusive Contributor Wall" in the `README.md` that automatically matches the functionality of the `all-contributors` spec. It will recognize and celebrate *all* types of contributions (Code, Issues/Bugs, Reviews) and display them visually with their avatars and contribution emojis (💻, 🐛, 👀).

## Proposed Changes

### Documentation
#### [MODIFY] [README.md](file:///Users/chaowan/Documents/vllm test project/README.md)
Add a collapsible legend above the Contributor Wall, identical in format to the "TPU Support Matrix Dashboard" legend, explaining the emojis:

```html
<details>
<summary> <b>🌟 <i>Contribution Type Legend</i> </b> </summary>

> | Emoji | Contribution | Meaning |
> | :--- | :--- | :--- |
> | 💻 | **Code** | Submitted merged pull requests or code changes. |
> | 🐛 | **Issues** | Opened valid issues or bug reports. |
> | 👀 | **Reviews** | Reviewed pull requests and provided feedback. |

</details>
<br>
```

### Scripts
#### [MODIFY] [scripts/update_contributors.py](file:///Users/chaowan/Documents/vllm test project/scripts/update_contributors.py)
We will significantly enhance this script to become an **Automated Auto-Discovery Bot**:
1.  **Code (💻)**: Fetch `/repos/{owner}/{repo}/contributors` for code committers.
2.  **Issues/Bugs (🐛)**: Fetch `/repos/{owner}/{repo}/issues` to find authors of opened issues (excluding PRs).
3.  **Reviews (👀)**: Fetch `/repos/{owner}/{repo}/pulls` and their `/reviews` to find people who have actively reviewed code.
4.  **Aggregation**: Combine these into a single dictionary per user, collecting a list of their contribution types.
5.  **Ranking Algorithm**: Calculate a `Total Effort Score = Commits + Issues Opened + PRs Reviewed` for each user. Rank contributors from highest score to lowest (ties broken alphabetically) to promote the most active maintainers to the top.
6.  **Rendering**: Update the HTML grid to exactly match the `all-contributors` visual style:
    - Center-aligned avatar.
    - Username linked to profile.
    - Emoji list below the name indicating what they did (e.g., `💻 🐛 👀`).

### Workflows
#### [MODIFY] [.github/workflows/update_contributors.yml](file:///Users/chaowan/Documents/vllm test project/.github/workflows/update_contributors.yml)
- (No major changes needed, the existing cron/push triggers will automatically run the enhanced python script).

## Verification Plan
1.  **Local Run**: Run `python3 scripts/update_contributors.py` locally. Ensure the terminal logs that it is parsing issues and reviews, and correctly identifying non-code contributors.
2.  **Visual Check**: Check the resulting `README.md` to ensure the HTML looks identical to the official `all-contributors` tables.
