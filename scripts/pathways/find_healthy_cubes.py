
import json
import subprocess
from collections import defaultdict

def get_node_data():
    # Fetch all data once
    cmd = ["kubectl", "get", "nodes", "-o", "json"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(result.stdout)

def is_node_ready(node):
    # Check Kubernetes Node Conditions for 'Ready' status
    conditions = node.get('status', {}).get('conditions', [])
    for c in conditions:
        if c.get('type') == 'Ready':
            return c.get('status') == 'True'
    return False

def process_and_print(data):
    # Data Structure: block -> cube_id -> list of nodes
    topology = defaultdict(lambda: {"cubes": defaultdict(list), "reservation": "UNKNOWN"})

    LABEL_BLOCK = "cloud.google.com/gce-topology-block"
    LABEL_CUBE  = "cloud.google.com/gke-tpu-partition-4x4x4-id"
    LABEL_STATE = "cloud.google.com/gke-tpu-partition-4x4x4-state"
    LABEL_RESERVATION = "cloud.google.com/reservation-name"

    # 1. ORGANIZE DATA
    for node in data.get('items', []):
        labels = node.get('metadata', {}).get('labels', {})
        block = labels.get(LABEL_BLOCK)
        cube = labels.get(LABEL_CUBE)
        reservation = labels.get(LABEL_RESERVATION)

        if block and cube:
            topology[block]["cubes"][cube].append({
                "name": node.get("metadata", {}).get("name"),
                "ready": is_node_ready(node),
                "state": labels.get(LABEL_STATE, "UNKNOWN"),
                "labels": labels
            })
            if reservation and topology[block]["reservation"] == "UNKNOWN":
                topology[block]["reservation"] = reservation

    # 2. PRINT REPORT
    # formatting strings
    h_fmt = "{:<30} | {:<30} | {:<20} | {:>6} | {:>6} | {:>8}"
    row_fmt = "{:<30} | {:<30} | {:<20} | {:>6} | {:>6} | {:>8}"

    print(h_fmt.format("RESERVATION", "TOPOLOGY BLOCK", "CATEGORY", "NODES", "CUBES", "TPUS"))
    print("-" * 115)

    for block, block_data in topology.items():
        reservation_name = block_data["reservation"]
        cubes = block_data["cubes"]
        # --- CALCULATION VARIABLES ---

        # A. TOTALS
        total_nodes = sum(len(nodes) for nodes in cubes.values())
        total_cubes = len(cubes)
        total_tpus  = total_nodes * 4  # Physical TPUs present

        # B. INFRASTRUCTURE FAILURES (NOT READY)
        # Logic: If ANY node in a cube is NotReady, the cube is disqualified.
        bad_infra_cubes = set()
        not_ready_node_count = 0

        for cube_id, nodes in cubes.items():
            cube_has_failure = False
            for n in nodes:
                if not n['ready']:
                    cube_has_failure = True
                    not_ready_node_count += 1

            if cube_has_failure:
                bad_infra_cubes.add(cube_id)

        bad_infra_cube_count = len(bad_infra_cubes)
        # Per requirements: 64 TPUs disqualified per bad cube
        bad_infra_tpu_loss = bad_infra_cube_count * 64

        # C. STATE FAILURES (UNHEALTHY LABEL)
        # Logic: Only look at cubes NOT already disqualified by infra issues.
        bad_state_cubes = set()
        bad_state_node_count = 0

        for cube_id, nodes in cubes.items():
            if cube_id in bad_infra_cubes:
                continue # Skip, already counted in row above

            # Check label state (assuming all nodes in cube share the label, check first one)
            # We treat anything NOT 'HEALTHY' as a failure here
            state = nodes[0]['state']
            if state != "HEALTHY":
                bad_state_cubes.add(cube_id)
                bad_state_node_count += len(nodes) # All nodes in this cube

        bad_state_cube_count = len(bad_state_cubes)
        bad_state_tpu_loss = bad_state_cube_count * 64

        # D. AVAILABLE (HEALTHY)
        # Logic: Cubes that survived both previous filters
        avail_cubes = set()
        avail_node_count = 0

        for cube_id, nodes in cubes.items():
            if cube_id not in bad_infra_cubes and cube_id not in bad_state_cubes:
                avail_cubes.add(cube_id)
                avail_node_count += len(nodes)

        avail_cube_count = len(avail_cubes)
        avail_tpus = avail_node_count * 4 # Actual available hardware

        # --- OUTPUT ROWS ---

        # 1. Total Row
        print(row_fmt.format(reservation_name, block, "Total", total_nodes, total_cubes, total_tpus))

        # 2. Infra/NotReady Row
        print(row_fmt.format(
            "", "",
            "Nodes NotReady",
            not_ready_node_count,
            bad_infra_cube_count,
            f"-{bad_infra_tpu_loss}"
        ))

        # 3. Unhealthy Label Row
        print(row_fmt.format(
            "", "",
            "Label Unhealthy",
            bad_state_node_count,
            bad_state_cube_count,
            f"-{bad_state_tpu_loss}"
        ))

        # 4. Available Row
        print(row_fmt.format(
            "", "",
            "Available (Healthy)",
            avail_node_count,
            avail_cube_count,
            avail_tpus
        ))

        print("-" * 115)

if __name__ == "__main__":
    try:
        data = get_node_data()
        process_and_print(data)
    except Exception as e:
        print(f"Error: {e}")
