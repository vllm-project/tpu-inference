import os
import requests
from google import genai

def main():
    github_token = os.environ.get("GITHUB_TOKEN")
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    repo = os.environ.get("GITHUB_REPOSITORY")
    pr_number = os.environ.get("PR_NUMBER")

    if not all([github_token, gemini_api_key, repo, pr_number]):
        print("Missing required environment variables (GITHUB_TOKEN, GEMINI_API_KEY, GITHUB_REPOSITORY, PR_NUMBER).")
        return

    # Fetch PR diff from GitHub API
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3.diff"
    }
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Failed to fetch PR diff: {response.text}")
        return
        
    pr_diff = response.text
    if not pr_diff.strip():
        print("Empty PR diff. Nothing to review.")
        return

    # Call Gemini to review the diff
    client = genai.Client(api_key=gemini_api_key)
    prompt = f"""
You are an expert Google engineer reviewing a Pull Request for the `tpu-inference` repository.
Please review the following code changes (provided as a git diff). 

Look for:
- Logical bugs or edge cases
- Python code quality (PEP 8, best practices, maintainability)
- If modifying README table generators, watch out for HTML formatting bugs
- Any missing test coverage

Provide your feedback in GitHub-flavored Markdown. Be constructive, concise, and clear.
Highlight specific lines of code where possible. If the code looks perfect, just say so.

### PR Diff:
```diff
{pr_diff}
```
"""
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        review_comment = response.text
    except Exception as e:
        print(f"Failed to call Gemini API: {e}")
        return

    # Post comment back to GitHub PR
    comment_url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    comment_headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    comment_data = {
        "body": f"🤖 **Gemini AI Review (gemini-2.5-flash):**\n\n{review_comment}"
    }
    
    post_response = requests.post(comment_url, headers=comment_headers, json=comment_data)
    if post_response.status_code == 201:
        print("Successfully posted AI review comment to PR!")
    else:
        print(f"Failed to post comment: {post_response.text}")

if __name__ == "__main__":
    main()
