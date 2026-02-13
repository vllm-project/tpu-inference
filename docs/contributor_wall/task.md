# Auto Leaderboard Task List

- [ ] **Analyze Requirements** <!-- id: 0 -->
    - [x] Define "Leaderboard" criteria: Inclusive Contributor Wall (avatars of all contributors) <!-- id: 1 -->
    - [x] Determine where to display it: `README.md` <!-- id: 2 -->
- [x] **Analyze Requirements** <!-- id: 0 -->
    - [x] Define "Leaderboard" criteria: Inclusive Contributor Wall matching `all-contributors` spec <!-- id: 1 -->
    - [x] Determine where to display it: `README.md` <!-- id: 2 -->
- [x] **Design Implementation Plan** <!-- id: 3 -->
    - [x] Update `implementation_plan.md` for advanced API fetching <!-- id: 4 -->
    - [x] Define display format (HTML avatar grid with emojis) <!-- id: 5 -->
- [x] **Implementation** <!-- id: 7 -->
    - [x] Enhance `scripts/update_contributors.py` to fetch Issues and Reviews <!-- id: 9 -->
    - [x] Update HTML generation to include contribution emojis <!-- id: 10 -->
- [ ] **Verification** <!-- id: 11 -->
    - [ ] Test python script locally to confirm non-code contributors are found <!-- id: 12 -->
    - [ ] Verify GitHub Action runs with updated script <!-- id: 13 -->
- [x] **Add Collapsible Legend**
    - [x] Update `implementation_plan.md` to propose the legend structure
    - [x] Add `<details>` legend to `README.md` above the Contributor Wall
