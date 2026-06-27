#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Function to generate a GitHub App Installation Access Token
generate_installation_token() {
    # 1. Generate the JWT using embedded Python
    local jwt
    if ! jwt=$(python3 <<EOF
import jwt
import time
import os
import sys

app_id = os.environ.get('GITHUB_CI_BOT_APP_ID')
pem_key = os.environ.get('GITHUB_CI_BOT_PEM')

if not app_id or not pem_key:
    print("Error: Secrets GITHUB_CI_BOT_APP_ID or GITHUB_CI_BOT_PEM are missing.", file=sys.stderr)
    sys.exit(1)

payload = {
    'iat': int(time.time()) - 60,
    'exp': int(time.time()) + (10 * 60),
    'iss': app_id
}

try:
    encoded = jwt.encode(payload, pem_key, algorithm='RS256')
    # jwt.encode returns a string in newer PyJWT versions, but bytes in older ones. 
    # This handles both safely.
    print(encoded.decode('utf-8') if isinstance(encoded, bytes) else encoded)
except Exception as e:
    print(f"JWT Encoding Error: {e}", file=sys.stderr)
    sys.exit(1)
EOF
); then
        echo "Failed to generate JWT." >&2
        return 1
    fi

    # 2. Fetch the Installation ID
    # Note: If your app is installed on multiple organizations/accounts, 
    # .[0].id selects the FIRST installation. Adjust the jq filter if you need a specific one.
    local installations_response
    installations_response=$(curl -s -X GET \
        -H "Authorization: Bearer $jwt" \
        -H "Accept: application/vnd.github+json" \
        https://api.github.com/app/installations)
    
    local installation_id
    installation_id=$(echo "$installations_response" | jq -r '.[0].id')

    if [ "$installation_id" = "null" ] || [ -z "$installation_id" ]; then
        echo "Error: Could not retrieve Installation ID. Make sure the App is installed on a repository/account." >&2
        echo "API Response: $installations_response" >&2
        return 1
    fi

    # 3. Exchange JWT and Installation ID for an Installation Access Token
    local token_response
    token_response=$(curl -s -X POST \
        -H "Authorization: Bearer $jwt" \
        -H "Accept: application/vnd.github+json" \
        https://api.github.com/app/installations/"${installation_id}"/access_tokens)

    local installation_token
    installation_token=$(echo "$token_response" | jq -r '.token')

    if [ "$installation_token" = "null" ] || [ -z "$installation_token" ]; then
        echo "Error: Could not generate Installation Access Token." >&2
        echo "API Response: $token_response" >&2
        return 1
    fi

    # Echo the final token to stdout so it can be captured
    echo "$installation_token"
}

# ==========================================
# Usage
# ==========================================

# GITHUB_CI_BOT_PEM and GITHUB_CI_BOT_APP_ID are stored as secrets in Buildkite, the yml step should look like this:
# steps:
#   - label: "Export CI Bot Token"
#     secrets:
#       - "GITHUB_CI_BOT_PEM"
#       - "GITHUB_CI_BOT_APP_ID"
#     command: ".buildkite/scripts/export_ci_bot_token.sh"
# In this way the GITHUB_CI_BOT_PEM and GITHUB_CI_BOT_APP_ID are already exported in your ci env vars.
# Token can be expired so might need to refresh it as needed.
echo "🚀 Generating CI Bot token..."
# Capture the output of the function into a variable
GITHUB_CI_BOT_TOKEN="$(generate_installation_token)"
export GITHUB_CI_BOT_TOKEN
