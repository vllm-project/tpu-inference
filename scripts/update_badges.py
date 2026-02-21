import os
import re
import sys
import requests
from collections import Counter
from urllib.parse import quote

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO = os.environ.get("GITHUB_REPOSITORY", "vllm-project/tpu-inference")
README_PATH = "README.md"

def get_headers():
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    return headers

def fetch_all_open_issues():
    """Fetches all open issues to count labels accurately locally."""
    issues = []
    page = 1
    url = f"https://api.github.com/repos/{REPO}/issues"
    
    print("Fetching all open issues...")
    while True:
        params = {"state": "open", "per_page": 100, "page": page}
        try:
            resp = requests.get(url, headers=get_headers(), params=params)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            issues.extend(data)
            page += 1
        except Exception as e:
            print(f"Error fetching issues: {e}")
            break
            
    return issues

def fetch_repo_labels():
    """Fetches all labels to get their official colors."""
    labels_map = {} # name -> color (hex without #)
    page = 1
    url = f"https://api.github.com/repos/{REPO}/labels"
    
    print("Fetching repository labels for colors...")
    while True:
        params = {"per_page": 100, "page": page}
        try:
            resp = requests.get(url, headers=get_headers(), params=params)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            for label in data:
                labels_map[label["name"]] = label["color"]
            page += 1
        except Exception as e:
            print(f"Error fetching labels: {e}")
            break
    
    return labels_map

def generate_badge_markdown(label_name, count, color):
    """Generates a Shields.io badge markdown link."""
    # Shields.io format: /badge/<LABEL>-<MESSAGE>-<COLOR>
    # We need to URL encode the label and message segments
    
    # Shields.io requires dashes to be doubled
    def escape_shield_part(text):
        return quote(str(text).replace("-", "--").replace("_", "__"))

    safe_label = escape_shield_part(label_name)
    # Inspiration uses clean numbers, so just the number without 'open'
    # Use flat-square for smaller, cleaner look
    safe_count = escape_shield_part(f"{count}")
    
    # Remove logo and use flat-square style
    badge_url = f"https://img.shields.io/badge/{safe_label}-{safe_count}-{color}?style=flat-square"
    
    # Issue query link
    # GitHub search query needs quotes around label if it has spaces
    query_label = quote(f'"{label_name}"')
    issue_link = f"https://github.com/{REPO}/issues?q=is%3Aissue+is%3Aopen+label%3A{query_label}"
    
    return f"[![{label_name}]({badge_url})]({issue_link})"

def update_readme_badges(badges_md):
    """Injects the new badges into README.md."""
    if not os.path.exists(README_PATH):
        print(f"Error: {README_PATH} not found.")
        return

    with open(README_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    # Look for the marker or the old badge line to replace
    # We will establish a new marker system if not present
    start_marker = "<!-- START: issue_badges -->"
    end_marker = "<!-- END: issue_badges -->"
    
    if start_marker in content and end_marker in content:
        pattern = f"({re.escape(start_marker)})(.*?)({re.escape(end_marker)})"
        replacement = f"\\1\n{badges_md}\n\\3"
        new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    else:
        # Fallback: Replace the old hardcoded line
        # This regex matches the specific 3-badge structure we saw earlier
        print("Markers not found, replacing hardcoded badges...")
        old_pattern = r'\[!\[good first issue\].*?label%3Ablocked\)'
        replacement = f"{start_marker}\n{badges_md}\n{end_marker}"
        new_content = re.sub(old_pattern, replacement, content, flags=re.DOTALL)

    if new_content != content:
        with open(README_PATH, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("✅ README.md updated with dynamic badges.")
    else:
        print("ℹ️ No changes to README.md.")

def main():
    issues = fetch_all_open_issues()
    label_colors = fetch_repo_labels()
    
    # Count labels
    label_counts = Counter()
    for issue in issues:
        # Check if it's a pull request (GitHub API returns PRs as issues unless filtered?)
        # Actually 'issues' endpoint returns PRs too, but they have a 'pull_request' key.
        if "pull_request" in issue:
            continue
            
        for label in issue.get("labels", []):
            label_counts[label["name"]] += 1
            
    # Get top 5 labels
    top_labels = label_counts.most_common(5)
    
    badges = []
    for label_name, count in top_labels:
        color = label_colors.get(label_name, "lightgrey")
        badges.append(generate_badge_markdown(label_name, count, color))
        
    # Add 'View All' badge with total count
    total_open_issues = len(issues)
    view_all_url = f"https://github.com/{REPO}/issues"
    # Left: View All Issues, Right: <count>, Color: 238636 (GitHub Green)
    # Using flat-square as requested
    safe_view_all = quote("View All Issues")
    view_all_badge = f"[![View All Issues](https://img.shields.io/badge/{safe_view_all}-{total_open_issues}-238636?style=flat-square)]({view_all_url})"
    badges.append(view_all_badge)
    
    final_markdown = " ".join(badges)
    # Wrap in a paragraph for spacing?
    # final_markdown = f"{final_markdown}"
    
    print("\nGenerated Badges:")
    print(final_markdown)
    
    update_readme_badges(final_markdown)

if __name__ == "__main__":
    main()
