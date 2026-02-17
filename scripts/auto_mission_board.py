import csv
import os
import argparse
import sys
import requests
import time

# --- CONFIGURATION ---
# Map CSVs to their "Area" for the issue title/body
CSV_MAP = {
    "Model Support": [
        "support_matrices/text_only_model_support_matrix.csv",
        "support_matrices/multimodal_model_support_matrix.csv"
    ],
    "Core Features": "support_matrices/feature_support_matrix.csv",
    "Parallelism": "support_matrices/parallelism_support_matrix.csv",
    "Quantization": "support_matrices/quantization_support_matrix.csv",
    "Kernel Support": "support_matrices/kernel_support_matrix.csv"
}

GITHUB_REPO = os.environ.get("GITHUB_REPOSITORY", "vllm-project/tpu-inference")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
MAX_ISSUES_PER_RUN = 5  # Rate limit safety

def read_csv_data(file_path):
    """Reads a CSV file and returns headers and data rows."""
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: File not found: {file_path}")
        return None, []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
        return (rows[0], rows[1:]) if rows else (None, [])

def check_issue_exists(title, token):
    """Checks if an issue with the given title already exists (open or closed)."""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    params = {
        "state": "all",
        "search": title, # Warning: This is a loose search, we need to filter results
        "per_page": 100
    }
    
    # Better approach: Search via search API to be more precise
    search_url = f"https://api.github.com/search/issues?q=repo:{GITHUB_REPO}+type:issue+in:title+\"{title}\""
    try:
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()
        data = response.json()
        total_count = data.get("total_count", 0)
        return total_count > 0
    except requests.exceptions.RequestException as e:
        print(f"❌ Error searching for issue: {e}")
        return False

def create_issue(title, body, token):
    """Creates a new issue on GitHub."""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    payload = {
        "title": title,
        "body": body,
        "labels": ["good first issue", "contribution-welcome", "auto-generated"]
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        issue_data = response.json()
        print(f"✅ Created issue: {issue_data['html_url']}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Error creating issue: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Auto Mission Board: Generate issues from support matrices.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without creating issues.")
    args = parser.parse_args()

    if not args.dry_run and not GITHUB_TOKEN:
        print("❌ Error: GITHUB_TOKEN environment variable is required (unless --dry-run is used).")
        sys.exit(1)

    issues_created = 0
    
    print(f"🔍 Analyzing support matrices for '{GITHUB_REPO}'...")

    for category, file_sources in CSV_MAP.items():
        sources = file_sources if isinstance(file_sources, list) else [file_sources]
        
        for file_path in sources:
            headers, rows = read_csv_data(file_path)
            if not headers: continue

            # Identify "unverified" columns (usually indices starting from 1)
            # We assume Col 0 is the Item Name (Model, Feature, etc.)
            status_cols = range(1, len(headers)) 

            for row in rows:
                if not row: continue
                item_name = row[0]
                
                for col_idx in status_cols:
                    if col_idx >= len(row): continue
                    
                    status = row[col_idx].strip()
                    metric_name = headers[col_idx]

                    # Criteria for creating an issue
                    if status.lower() in ["unverified", "x", "❌", "untested", "❓ untested", "❌ failing"]:
                        if issues_created >= MAX_ISSUES_PER_RUN:
                            print(f"🛑 Reached max issues limit ({MAX_ISSUES_PER_RUN}). Stopping.")
                            return

                        issue_title = f"[Mission] Verify {metric_name} for {item_name}"
                        issue_body = (
                            f"### Mission: Verify {metric_name} for {item_name}\n\n"
                            f"**Category**: {category}\n"
                            f"**Item**: {item_name}\n"
                            f"**Metric**: {metric_name}\n\n"
                            f"This functionality is currently marked as **{status}** in our [Support Matrix](https://github.com/{GITHUB_REPO}/tree/main/support_matrices).\n\n"
                            f"### How to Contribute\n"
                            f"1.  Claim this issue by commenting below.\n"
                            f"2.  Run the relevant tests or benchmarks on a TPU environment.\n"
                            f"3.  If it passes, submit a PR updating the CSV file status to `✅`.\n"
                            f"4.  If it fails, report the error logs here.\n\n"
                            f"Happy hacking! 🚀"
                        )

                        if args.dry_run:
                            print(f"[DRY RUN] Would create issue: '{issue_title}'")
                            issues_created += 1
                        else:
                            # Check if exists
                            if check_issue_exists(issue_title, GITHUB_TOKEN):
                                print(f"ℹ️ Issue already exists (skipping): {issue_title}")
                            else:
                                if create_issue(issue_title, issue_body, GITHUB_TOKEN):
                                    issues_created += 1
                                    time.sleep(1) # Rate limit courtesy

    print(f"\n✨ Done. Total issues created: {issues_created}")

if __name__ == "__main__":
    main()
