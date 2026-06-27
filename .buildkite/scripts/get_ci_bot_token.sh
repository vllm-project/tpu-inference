#!/bin/bash

# Function to generate a GitHub App Installation Access Token
generate_installation_token() {
    # 1. Generate the JWT using embedded Python
    local jwt
    jwt=$(python3 <<EOF
import jwt
import time
import os
import sys

app_id = os.environ.get('GITHUB_CI_BOT_APP_ID')
pem_key = os.environ.get('GITHUB_CI_BOT_PEM')

if not app_id or not pem_key:
    print("Error: GITHUB_CI_BOT_APP_ID or GITHUB_CI_BOT_PEM environment variables are missing.", file=sys.stderr)
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
)

    # Check if Python script failed
    if [ $? -ne 0 ]; then
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
# Example Usage
# ==========================================

# Assuming GITHUB_CI_BOT_PEM and GITHUB_CI_BOT_APP_ID are already exported in your CI env:
echo "Generating GitHub App token..."

# Capture the output of the function into a variable
GITHUB_TOKEN=$(generate_installation_token)

if [ $? -eq 0 ]; then
    echo "Successfully generated token."
    
    # Mask the token in CI logs (GitHub Actions format, remove if using another CI like Jenkins/GitLab)
    echo "::add-mask::$GITHUB_TOKEN"
    
    # Export it for subsequent steps in your script to use
    export GITHUB_TOKEN
    
    # Test the token (Optional)
    # curl -s -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/rate_limit
else
    echo "Failed to generate GitHub App token. Exiting."
    exit 1
fi