import os
import re
import sys
import requests

GITHUB_REPO = os.environ.get("GITHUB_REPOSITORY", "vllm-project/tpu-inference")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
README_PATH = "README.md"

def fetch_contributors(token):
    """Fetches all contributors for the repository using the GitHub API."""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contributors"
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    if token:
        headers["Authorization"] = f"token {token}"
        
    contributors = []
    page = 1
    
    while True:
        # Loop through pagination to get all contributors
        response = requests.get(url, headers=headers, params={"per_page": 100, "page": page})
        response.raise_for_status()
        data = response.json()
        
        if not data:
            break
            
        contributors.extend(data)
        page += 1
        
    return contributors

def generate_html_grid(contributors):
    """Generates an HTML grid of contributor avatars."""
    # We use a simple flexbox layout for the grid
    html = ['<div align="center" style="display: flex; flex-wrap: wrap; justify-content: center; gap: 10px;">']
    
    for contributor in contributors:
        login = contributor.get("login")
        avatar_url = contributor.get("avatar_url")
        html_url = contributor.get("html_url")
        
        # Skip bots
        if contributor.get("type") == "Bot" or login.endswith("[bot]"):
            continue
            
        # Create a linked avatar image, using width to control size and border-radius for circles
        img_tag = f'<a href="{html_url}" title="{login}"><img src="{avatar_url}" width="60" style="border-radius: 50%; margin: 5px;"/></a>'
        html.append(img_tag)
        
    html.append('</div>')
    return "\n".join(html)

def update_readme(html_grid):
    """Injects the HTML grid into the README.md file."""
    if not os.path.exists(README_PATH):
        print(f"❌ Error: {README_PATH} not found.")
        sys.exit(1)

    with open(README_PATH, "r", encoding="utf-8") as f:
        content = f.read()

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
        
    print(f"✅ Successfully updated {README_PATH} with contributors grid.")
    return True

def main():
    print(f"🔍 Fetching contributors for '{GITHUB_REPO}'...")
    try:
        contributors = fetch_contributors(GITHUB_TOKEN)
        print(f"👥 Found {len(contributors)} contributors.")
        
        html_grid = generate_html_grid(contributors)
        update_readme(html_grid)
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error communicating with GitHub API: {e}")
        # Print extra info if auth failed
        if e.response is not None and e.response.status_code in [401, 403]:
            print("⚠️ Check your GITHUB_TOKEN permissions.")
        sys.exit(1)

if __name__ == "__main__":
    main()
