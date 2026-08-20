#!/bin/bash
# Report which HF token this agent hands to jobs. Prints only sha256 prefixes, key names and HTTP codes.
d12() { sha256sum | cut -c1-12; }
echo "host=$(hostname) agent_env_HF_TOKEN_digest=$(printf %s "${HF_TOKEN:-}" | d12) (set=$([ -n "${HF_TOKEN:-}" ] && echo yes || echo no))"
echo "/etc/environment keys: $(sed -E 's/=.*//' /etc/environment | tr '\n' ' ')"
echo "/etc/environment HF_TOKEN lines: $(grep -c '^HF_TOKEN=' /etc/environment)"
echo "/etc/environment HF_TOKEN digests (in order): $(grep '^HF_TOKEN=' /etc/environment | cut -d= -f2- | while read -r t; do printf %s "$t" | d12; done | tr '\n' ' ')"
HOOK=/etc/buildkite-agent/hooks/environment
if [ -r "$HOOK" ]; then echo "hook keys: $(grep -oE '^export [A-Z_]+' "$HOOK" | tr '\n' ' ')"; echo "hook HF_TOKEN digest: $(bash -c ". $HOOK; printf %s \"\$HF_TOKEN\"" | d12)"; else echo "hook: not readable/absent"; fi
source /etc/environment
echo "after source /etc/environment: HF_TOKEN digest=$(printf %s "${HF_TOKEN:-}" | d12)"
echo "gated_http=$(curl -s -o /dev/null -w '%{http_code}' -L -I -H "Authorization: Bearer ${HF_TOKEN:-}" https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/resolve/main/config.json)"
