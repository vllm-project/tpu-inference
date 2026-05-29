import json
import os
import sys
import time
import uuid
import google.auth
from google.cloud import bigquery

credentials, project_id = google.auth.default()
print(project_id)
print(credentials)
print(credentials.service_account_email)

# sys.argv[0] is always the script name itself
print(f"Script name: {sys.argv[0]}")

# Capture the rest of the arguments
if len(sys.argv) > 1:
    print(f"Arguments passed: {sys.argv[1:]}")

row_id = uuid.uuid4().hex

row = {
    'entry_id': row_id,
    'sponge_id': os.getenv('MLCOMPASS_SPONGE_ID'),
    'test_name': 'orti-vllm-test',
    'succeeded': False,
    'metrics': json.dumps({}),
    'link_map': json.dumps({}),
    'exc_timestamp_millis': int(time.time() * 1000),
    'client_info': {
        'github_commit': 'commit_sha_core',
        'commit_branch_name': 'commit_branch',
    },
    'env_commit_map': json.dumps({}),
    'mlcompass_tracking_id': os.getenv('MLCOMPASS_TRACKING_ID', row_id),
    'mlcopmass_execution_mode': os.getenv('MLCOPMASS_EXECUTION_MODE', 'oneshot'),
}

client = bigquery.Client(project='google.com:ml-compass-benchmarks')
table_ref = client.dataset('benchmarks_dataset').table('benchmarks_dev')
table_obj = client.get_table(table_ref)
client.insert_rows(table_obj, [row])

print('MLCompass export file called')
