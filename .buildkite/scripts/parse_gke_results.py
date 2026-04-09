import json
import sys
import os

def parse_and_dump(file_path, record_id):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    # Extract input_len and output_len from filename
    # Assuming filename format like path/to/1024_8192.json
    base_name = os.path.basename(file_path)
    name_part, _ = os.path.splitext(base_name)
    try:
        input_len_str, output_len_str = name_part.split('_')
        input_len = int(input_len_str)
        output_len = int(output_len_str)
    except ValueError:
        print(f"Warning: Could not parse input/output len from filename: {base_name}. Using NULL.", file=sys.stderr)
        input_len = 'NULL'
        output_len = 'NULL'

    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from line: {line}", file=sys.stderr)
                continue

            rate = data.get('request_rate')
            throughput = data.get('request_throughput')
            
            # Latency metrics
            median_ttft = data.get('median_ttft_ms')
            p90_ttft = data.get('p90_ttft_ms')
            p99_ttft = data.get('p99_ttft_ms')
            
            median_tpot = data.get('median_tpot_ms')
            p90_tpot = data.get('p90_tpot_ms')
            p99_tpot = data.get('p99_tpot_ms')
            
            median_itl = data.get('median_itl_ms')
            p99_itl = data.get('p99_itl_ms')

            # Generate a unique RecordId for this rate
            unique_record_id = f"{record_id}_{rate}"

            sql = f"""
            INSERT INTO RunRecord (
                RecordId, Status, CreatedTime, LastUpdate,
                Throughput, MedianTTFT, P90TTFT, P99TTFT,
                MedianTPOT, P90TPOT, P99TPOT,
                MedianITL, P99ITL,
                Device, Model, RunType, CodeHash, Dataset, CreatedBy,
                InputLen, OutputLen
            ) VALUES (
                '{unique_record_id}', 'COMPLETED', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(),
                {throughput or 'NULL'}, {median_ttft or 'NULL'}, {p90_ttft or 'NULL'}, {p99_ttft or 'NULL'},
                {median_tpot or 'NULL'}, {p90_tpot or 'NULL'}, {p99_tpot or 'NULL'},
                {median_itl or 'NULL'}, {p99_itl or 'NULL'},
                'GKE', '{data.get('model_id', 'N/A')}', 'GKE_DISAGG', 'N/A', 'random', 'scheduler',
                {input_len}, {output_len}
            );
            """
            print(" ".join(sql.split()))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python parse_gke_results.py <file_path> <record_id>", file=sys.stderr)
        sys.exit(1)
    parse_and_dump(sys.argv[1], sys.argv[2])

