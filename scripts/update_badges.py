import os
import re
import requests

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO = os.environ.get("GITHUB_REPOSITORY", "vllm-project/tpu-inference")
README_PATH = "README.md"

def get_issue_count(query):
    """Query GitHub API for the total count of issues matching the search query."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    url = f"https://api.github.com/search/issues?q=repo:{REPO} {query}"
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json().get("total_count", 0)
    else:
        print(f"Error fetching data for query '{query}': {response.status_code}")
        print(response.text)
        return None

def update_readme_badges(good_first_count, docs_count, blocked_count):
    """Read README.md, replace badge tags with new counts, and save."""
    with open(README_PATH, 'r', encoding='utf-8') as f:
        readme_content = f.read()

    # Good First Issue Badge
    # Example: ![good first issue](https://img.shields.io/github/issues/vllm-project/tpu-inference/good%20first%20issue.svg?label=good%20first%20issue&color=green&style=flat-square)
    readme_content = re.sub(
        r'(\[!\[good first issue\]\(https://img\.shields\.io/github/issues/[^/]+/[^/]+/good%20first%20issue\.svg\?label=)[^&]+(&color=)',
        rf'\g<1>{good_first_count}%20open\g<2>',
        readme_content
    )

    # Documentation Bugs Badge
    # Example: ![documentation bugs](https://img.shields.io/github/issues-search/vllm-project/tpu-inference?query=is%3Aopen%20is%3Aissue%20label%3Adocumentation%20label%3Abug&label=documentation%20bugs&color=orange&style=flat-square)
    readme_content = re.sub(
        r'(\[!\[documentation bugs\]\(https://img\.shields\.io/github/issues-search/[^?]+\?query=[^&]+&label=)[^&]+(&color=)',
        rf'\g<1>{docs_count}%20open\g<2>',
        readme_content
    )

    # Blocked Issues Badge 
    # Example: ![blocked issues](https://img.shields.io/github/issues/vllm-project/tpu-inference/blocked.svg?label=🛑%20Blocked&color=brightred&style=flat-square)
    readme_content = re.sub(
        r'(\[!\[blocked issues\]\(https://img\.shields\.io/github/issues/[^/]+/[^/]+/blocked\.svg\?label=🛑%20Blocked)[^&]*(&color=)',
        rf'\g<1>%20%7C%20{blocked_count}%20open\g<2>',
        readme_content
    )

    with open(README_PATH, 'w', encoding='utf-8') as f:
        f.write(readme_content)

def main():
    print(f"Fetching issue counts for {REPO}...")
    
    count_good_first = get_issue_count('is:issue is:open label:"good first issue"')
    count_docs = get_issue_count('is:issue is:open label:documentation label:bug')
    count_blocked = get_issue_count('is:issue is:open label:blocked')

    if None in (count_good_first, count_docs, count_blocked):
        print("Failed to retrieve all counts. Aborting README update.")
        return

    print(f"Good First Issues: {count_good_first}")
    print(f"Documentation Bugs: {count_docs}")
    print(f"Blocked Issues: {count_blocked}")

    print("Updating README.md...")
    update_readme_badges(count_good_first, count_docs, count_blocked)
    print("Successfully updated README.md.")

if __name__ == "__main__":
    main()
