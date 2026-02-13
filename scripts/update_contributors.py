import os
import re
import sys
import requests
from collections import defaultdict

GITHUB_REPO = os.environ.get("GITHUB_REPOSITORY", "vllm-project/tpu-inference")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
README_PATH = "README.md"

def get_headers():
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    return headers

def fetch_paginated_api(url, params=None):
    """Yields all items from a paginated GitHub API endpoint."""
    if params is None:
        params = {}
    params['per_page'] = 100
    page = 1
    
    while True:
        params['page'] = page
        response = requests.get(url, headers=get_headers(), params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            break
            
        for item in data:
            yield item
            
        page += 1

def is_bot(user_info):
    login = user_info.get("login", "")
    type_str = user_info.get("type", "")
    return type_str == "Bot" or login.endswith("[bot]") or login == "dependabot[bot]"

def fetch_all_contributors():
    """
    Fetches Code, Issue, and Review contributors.
    Returns a dict: { login: { "avatar_url": url, "html_url": url, "contributions": set() } }
    """
    contributors_data = defaultdict(lambda: {"contributions": set()})

    # 1. Fetch Code Contributors (💻)
    print("Fetching code contributors...")
    url_commits = f"https://api.github.com/repos/{GITHUB_REPO}/contributors"
    for user in fetch_paginated_api(url_commits):
        if is_bot(user): continue
        login = user["login"]
        contributors_data[login]["avatar_url"] = user["avatar_url"]
        contributors_data[login]["html_url"] = user["html_url"]
        contributors_data[login]["contributions"].add("💻")

    # 2. Fetch Issue Authors (🐛)
    # We use the search API to find issues created in this repo (excludes PRs)
    print("Fetching issue contributors...")
    url_issues = f"https://api.github.com/repos/{GITHUB_REPO}/issues"
    # GitHub issue API returns PRs too if not filtered, but /issues endpoint includes both.
    # To be safe and simple without search rate limits, we iterate issues and check if it's not a PR.
    # For large repos, Search API is better, but iteration is fine here.
    try:
        for issue in fetch_paginated_api(url_issues, params={"state": "all"}):
            if "pull_request" not in issue: # It's a real issue
                user = issue.get("user")
                if user and not is_bot(user):
                    login = user["login"]
                    contributors_data[login]["avatar_url"] = user["avatar_url"]
                    contributors_data[login]["html_url"] = user["html_url"]
                    contributors_data[login]["contributions"].add("🐛")
    except Exception as e:
        print(f"Warning fetching issues: {e}")

    # 3. Fetch PR Reviewers (👀)
    print("Fetching review contributors...")
    url_pulls = f"https://api.github.com/repos/{GITHUB_REPO}/pulls"
    try:
        # Fetch up to 200 most recent PRs to find reviewers to save API calls
        response = requests.get(url_pulls, headers=get_headers(), params={"state": "all", "per_page": 100})
        if response.status_code == 200:
            for pr in response.json():
                pr_number = pr["number"]
                url_reviews = f"https://api.github.com/repos/{GITHUB_REPO}/pulls/{pr_number}/reviews"
                rev_resp = requests.get(url_reviews, headers=get_headers())
                if rev_resp.status_code == 200:
                    for review in rev_resp.json():
                        user = review.get("user")
                        if user and not is_bot(user):
                            login = user["login"]
                            contributors_data[login]["avatar_url"] = user["avatar_url"]
                            contributors_data[login]["html_url"] = user["html_url"]
                            contributors_data[login]["contributions"].add("👀")
    except Exception as e:
        print(f"Warning fetching reviews: {e}")

    return contributors_data

def generate_html_grid(contributors_data):
    """Generates an HTML grid matching the all-contributors visual style."""
    html = [
        '<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->',
        '<!-- prettier-ignore-start -->',
        '<!-- markdownlint-disable -->',
        '<table>',
        '  <tbody>'
    ]
    
    # Sort alphabetically by username
    sorted_users = sorted(contributors_data.items(), key=lambda x: x[0].lower())
    
    columns = 7
    for i in range(0, len(sorted_users), columns):
        row_users = sorted_users[i:i+columns]
        
        # Open row
        html.append('    <tr>')
        
        for login, data in row_users:
            avatar = data["avatar_url"]
            profile = data["html_url"]
            emojis = " ".join(sorted(list(data["contributions"])))
            
            # The all-contributors style cell
            cell = (
                f'      <td align="center" valign="top" width="14.28%">'
                f'<a href="{profile}"><img src="{avatar}?s=100" width="100px;" alt="{login}"/><br />'
                f'<sub><b>{login}</b></sub></a><br />'
                f'<a href="{profile}" title="Contributions">{emojis}</a>'
                f'</td>'
            )
            html.append(cell)
            
        # Close row
        html.append('    </tr>')
        
    html.extend([
        '  </tbody>',
        '</table>',
        '',
        '<!-- markdownlint-restore -->',
        '<!-- prettier-ignore-end -->',
        '<!-- ALL-CONTRIBUTORS-LIST:END -->'
    ])
    
    return "\n".join(html)

def update_readme(html_grid):
    """Injects the HTML grid into the README.md file."""
    if not os.path.exists(README_PATH):
        print(f"❌ Error: {README_PATH} not found.")
        sys.exit(1)

    with open(README_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # We use our custom markers from earlier
    start_marker = "<!-- START: contributors -->"
    end_marker = "<!-- END: contributors -->"
    
    pattern = f"({re.escape(start_marker)})(.*?)({re.escape(end_marker)})"
    replacement = f"\\1\n{html_grid}\n\\3"
    
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    if new_content == content:
        print("ℹ️ No changes needed in README.md")
        return False

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)
        
    print(f"✅ Successfully updated {README_PATH} with inclusive contributors grid.")
    return True

def main():
    print(f"🔍 Starting Auto-Discovery for '{GITHUB_REPO}'...")
    try:
        contributors_data = fetch_all_contributors()
        print(f"👥 Found {len(contributors_data)} totally unique contributors across code, issues, and reviews.")
        
        html_grid = generate_html_grid(contributors_data)
        update_readme(html_grid)
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error communicating with GitHub API: {e}")
        time.sleep(1) # just to clear io
        sys.exit(1)

if __name__ == "__main__":
    main()
