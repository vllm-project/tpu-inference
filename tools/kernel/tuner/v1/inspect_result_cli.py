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
"""CLI tool for inspecting kernel tuning results.

Usage:
    python -m tools.kernel.tuner.v1.inspect_result_cli [--source local|spanner] [--db-path PATH] <command> [options]

Commands:
    list_case_sets      List available case sets (--filter KEYWORD)
    list_runs           List runs for a case set (--case_set_id ID --filter KEYWORD)
    count_buckets       Count buckets (--case_set_id ID --run_id ID)
    list_bucket_status  Show completed vs pending counts (--case_set_id ID --run_id ID)
    query_run_status    Show timing info for a run (--case_set_id ID --run_id ID)
    query_min_latency   Show best latency per TuningKey (--case_set_id ID --run_id ID)
"""

import argparse
import json
import os
from collections import defaultdict

# ---------------------------------------------------------------------------
# Local backend helpers
# ---------------------------------------------------------------------------


def _read_json(db_path, table_name):
    path = os.path.join(db_path, f'{table_name}.json')
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def local_list_case_sets(db_path, filter_kw=None):
    case_sets = _read_json(db_path, 'CaseSet')
    buckets = _read_json(db_path, 'WorkBuckets')

    run_counts = defaultdict(set)
    for b in buckets:
        run_counts[b['ID']].add(b['RunId'])

    rows = []
    for cs in case_sets:
        cs_id = cs['ID']
        desc = cs.get('Description', '')
        if filter_kw and filter_kw.lower() not in cs_id.lower(
        ) and filter_kw.lower() not in desc.lower():
            continue
        rows.append({
            'case_set_id': cs_id,
            'description': desc,
            'status': cs.get('Status', '?'),
            'scan_space': cs.get('ScanSpace', '?'),
            'num_runs': len(run_counts[cs_id]),
        })
    return rows


def local_list_runs(db_path, case_set_id=None, filter_kw=None):
    case_sets = _read_json(db_path, 'CaseSet')
    buckets = _read_json(db_path, 'WorkBuckets')

    cs_desc = {cs['ID']: cs.get('Description', '') for cs in case_sets}

    grouped = defaultdict(list)
    for b in buckets:
        grouped[(b['ID'], b['RunId'])].append(b)

    rows = []
    for (cs_id, run_id), bkts in sorted(grouped.items()):
        if case_set_id and cs_id != case_set_id:
            continue
        desc = cs_desc.get(cs_id, '')
        if filter_kw and filter_kw.lower() not in str(
                run_id) and filter_kw.lower() not in desc.lower():
            continue
        rows.append({
            'case_set_id': cs_id,
            'run_id': run_id,
            'description': desc,
            'num_buckets': len(bkts),
        })
    return rows


def local_count_buckets(db_path, case_set_id, run_id):
    buckets = _read_json(db_path, 'WorkBuckets')
    return sum(1 for b in buckets
               if b['ID'] == case_set_id and str(b['RunId']) == str(run_id))


def local_list_bucket_status(db_path, case_set_id, run_id):
    buckets = _read_json(db_path, 'WorkBuckets')
    counts = defaultdict(int)
    for b in buckets:
        if b['ID'] == case_set_id and str(b['RunId']) == str(run_id):
            counts[b.get('Status', 'UNKNOWN')] += 1
    return dict(counts)


def local_query_run_status(db_path, case_set_id, run_id):
    buckets = _read_json(db_path, 'WorkBuckets')
    relevant = [
        b for b in buckets
        if b['ID'] == case_set_id and str(b['RunId']) == str(run_id)
    ]
    if not relevant:
        return None

    timestamps = [b['UpdatedAt'] for b in relevant if b.get('UpdatedAt')]
    start_time = min(timestamps) if timestamps else 'N/A'
    last_completed = max(
        (b['UpdatedAt'] for b in relevant
         if b.get('Status') == 'COMPLETED' and b.get('UpdatedAt')),
        default='N/A',
    )
    total_us = sum(
        b.get('TotalTime', 0) or 0 for b in relevant
        if b.get('Status') == 'COMPLETED')
    return {
        'case_set_id': case_set_id,
        'run_id': run_id,
        'start_time': start_time,
        'last_completed_time': last_completed,
        'total_completed_time_us': total_us,
        'total_completed_time_s': f'{total_us / 1_000_000:.2f}',
    }


def local_query_min_latency(db_path, case_set_id, run_id):
    results = _read_json(db_path, 'CaseResults')
    cases = _read_json(db_path, 'KernelTuningCases')

    case_kv_map = {
        (c['ID'], c['CaseId']): c.get('CaseKeyValue')
        for c in cases
    }

    relevant = [
        r for r in results
        if r['ID'] == case_set_id and str(r['RunId']) == str(run_id)
        and r.get('ProcessedStatus') == 'SUCCESS' and r.get('Latency')
    ]

    key_best = {}
    for r in relevant:
        kv_str = case_kv_map.get((r['ID'], r['CaseId']))
        if not kv_str:
            continue
        try:
            kv = json.loads(kv_str)
        except (json.JSONDecodeError, TypeError):
            continue
        tk_str = json.dumps(kv.get('tuning_key', {}), sort_keys=True)
        lat = r.get('Latency', float('inf'))
        if tk_str not in key_best or lat < key_best[tk_str]['Latency']:
            key_best[tk_str] = {
                'tuning_key': kv.get('tuning_key'),
                'tunable_params': kv.get('tunable_params'),
                'Latency': lat,
                'WarmupTime': r.get('WarmupTime'),
                'CaseId': r.get('CaseId'),
            }

    return sorted(key_best.values(),
                  key=lambda x: json.dumps(x['tuning_key'], sort_keys=True))


# ---------------------------------------------------------------------------
# Spanner backend helpers
# ---------------------------------------------------------------------------


def _get_spanner_db(project, instance_id, database_id):
    from google.cloud import \
        spanner as gspanner  # pylint: disable=import-outside-toplevel
    client = gspanner.Client(project=project, disable_builtin_metrics=True)
    return client.instance(instance_id).database(database_id)


def spanner_list_case_sets(db, filter_kw=None):
    query = """
        SELECT cs.ID, cs.Description, cs.Status, cs.ScanSpace,
               COUNT(DISTINCT wb.RunId) AS num_runs
        FROM CaseSet cs
        LEFT JOIN WorkBuckets wb ON cs.ID = wb.ID
        GROUP BY cs.ID, cs.Description, cs.Status, cs.ScanSpace
        ORDER BY cs.ID
    """
    rows = []
    with db.snapshot() as snap:
        for cs_id, desc, status, scan_space, num_runs in snap.execute_sql(
                query):
            if filter_kw and filter_kw.lower() not in cs_id.lower() and \
                    filter_kw.lower() not in (desc or '').lower():
                continue
            rows.append({
                'case_set_id': cs_id,
                'description': desc,
                'status': status,
                'scan_space': scan_space,
                'num_runs': num_runs,
            })
    return rows


def spanner_list_runs(db, case_set_id=None, filter_kw=None):
    from google.cloud import \
        spanner as gspanner  # pylint: disable=import-outside-toplevel
    where = "WHERE wb.ID = @id" if case_set_id else ""
    query = f"""
        SELECT wb.ID, wb.RunId, cs.Description, COUNT(*) AS num_buckets
        FROM WorkBuckets wb JOIN CaseSet cs ON wb.ID = cs.ID
        {where}
        GROUP BY wb.ID, wb.RunId, cs.Description
        ORDER BY wb.ID, wb.RunId
    """
    params = {'id': case_set_id} if case_set_id else {}
    param_types = {'id': gspanner.param_types.STRING} if case_set_id else {}
    rows = []
    with db.snapshot() as snap:
        for cs_id, run_id, desc, num_buckets in snap.execute_sql(
                query, params=params, param_types=param_types):
            if filter_kw and filter_kw.lower() not in str(run_id) and \
                    filter_kw.lower() not in (desc or '').lower():
                continue
            rows.append({
                'case_set_id': cs_id,
                'run_id': run_id,
                'description': desc,
                'num_buckets': num_buckets
            })
    return rows


def spanner_count_buckets(db, case_set_id, run_id):
    from google.cloud import \
        spanner as gspanner  # pylint: disable=import-outside-toplevel
    with db.snapshot() as snap:
        return list(
            snap.execute_sql(
                "SELECT COUNT(*) FROM WorkBuckets WHERE ID = @id AND RunId = @rid",
                params={
                    'id': case_set_id,
                    'rid': run_id
                },
                param_types={
                    'id': gspanner.param_types.STRING,
                    'rid': gspanner.param_types.STRING
                },
            ))[0][0]


def spanner_list_bucket_status(db, case_set_id, run_id):
    from google.cloud import \
        spanner as gspanner  # pylint: disable=import-outside-toplevel
    result = {}
    with db.snapshot() as snap:
        for status, count in snap.execute_sql(
                "SELECT Status, COUNT(*) FROM WorkBuckets "
                "WHERE ID = @id AND RunId = @rid GROUP BY Status",
                params={
                    'id': case_set_id,
                    'rid': run_id
                },
                param_types={
                    'id': gspanner.param_types.STRING,
                    'rid': gspanner.param_types.STRING
                }):
            result[status] = count
    return result


def spanner_query_run_status(db, case_set_id, run_id):
    from google.cloud import \
        spanner as gspanner  # pylint: disable=import-outside-toplevel
    query = """
        SELECT
            MIN(UpdatedAt),
            MAX(CASE WHEN Status = 'COMPLETED' THEN UpdatedAt END),
            SUM(CASE WHEN Status = 'COMPLETED' THEN TotalTime ELSE 0 END)
        FROM WorkBuckets WHERE ID = @id AND RunId = @rid
    """
    with db.snapshot() as snap:
        start, last, total_us = list(
            snap.execute_sql(
                query,
                params={
                    'id': case_set_id,
                    'rid': run_id
                },
                param_types={
                    'id': gspanner.param_types.STRING,
                    'rid': gspanner.param_types.STRING
                },
            ))[0]
    total_us = total_us or 0
    return {
        'case_set_id': case_set_id,
        'run_id': run_id,
        'start_time': str(start),
        'last_completed_time': str(last),
        'total_completed_time_us': total_us,
        'total_completed_time_s': f'{total_us / 1_000_000:.2f}',
    }


def spanner_query_min_latency(db, case_set_id, run_id):
    from google.cloud import \
        spanner as gspanner  # pylint: disable=import-outside-toplevel
    query = """
        SELECT cr.CaseId, cr.Latency, cr.WarmupTime, ktc.CaseKeyValue
        FROM CaseResults cr
        JOIN KernelTuningCases ktc ON cr.ID = ktc.ID AND cr.CaseId = ktc.CaseId
        WHERE cr.ID = @id AND cr.RunId = @rid AND cr.ProcessedStatus = 'SUCCESS'
        ORDER BY cr.CaseId
    """
    key_best = {}
    with db.snapshot() as snap:
        for case_id, lat, warmup, kv_str in snap.execute_sql(
                query,
                params={
                    'id': case_set_id,
                    'rid': run_id
                },
                param_types={
                    'id': gspanner.param_types.STRING,
                    'rid': gspanner.param_types.STRING
                }):
            try:
                kv = json.loads(kv_str)
            except (json.JSONDecodeError, TypeError):
                continue
            tk_str = json.dumps(kv.get('tuning_key', {}), sort_keys=True)
            if tk_str not in key_best or lat < key_best[tk_str]['Latency']:
                key_best[tk_str] = {
                    'tuning_key': kv.get('tuning_key'),
                    'tunable_params': kv.get('tunable_params'),
                    'Latency': lat,
                    'WarmupTime': warmup,
                    'CaseId': case_id,
                }
    return sorted(key_best.values(),
                  key=lambda x: json.dumps(x['tuning_key'], sort_keys=True))


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _print_table(rows, headers=None):
    if not rows:
        print('  (no results)')
        return
    if headers is None:
        headers = list(rows[0].keys())
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row.get(h, ''))))
    fmt = '  '.join(f'{{:<{widths[h]}}}' for h in headers)
    print(fmt.format(*headers))
    print('  '.join('-' * widths[h] for h in headers))
    for row in rows:
        print(fmt.format(*[str(row.get(h, '')) for h in headers]))


def _print_min_latency(rows):
    if not rows:
        print('  (no successful results)')
        return
    for r in rows:
        print(f"  tuning_key={json.dumps(r['tuning_key'])}"
              f"  best_latency_us={r['Latency']}"
              f"  warmup_us={r['WarmupTime']}"
              f"  tunable_params={json.dumps(r['tunable_params'])}"
              f"  case_id={r['CaseId']}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser():
    parser = argparse.ArgumentParser(
        description='Inspect kernel tuning results.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--source',
                        choices=['local', 'spanner'],
                        default=None,
                        help='Storage backend.')
    parser.add_argument('--db-path',
                        default=None,
                        help='Path to local DB folder (local source only).')
    parser.add_argument('--project',
                        default='cloud-tpu-inference-test',
                        help='GCP project ID (spanner only).')
    parser.add_argument('--instance',
                        default='vllm-bm-inst',
                        help='Spanner instance ID (spanner only).')
    parser.add_argument('--database',
                        default='tune-gmm',
                        help='Spanner database ID (spanner only).')

    sub = parser.add_subparsers(dest='command', metavar='COMMAND')

    p = sub.add_parser('list_case_sets', help='List available case sets.')
    p.add_argument('--filter',
                   dest='filter_kw',
                   default=None,
                   help='Filter by keyword in case_set_id or description.')

    p = sub.add_parser('list_runs', help='List runs for a case set.')
    p.add_argument('--case_set_id', default=None)
    p.add_argument('--filter',
                   dest='filter_kw',
                   default=None,
                   help='Filter by keyword in run_id or description.')

    p = sub.add_parser('count_buckets',
                       help='Count buckets for a case set / run.')
    p.add_argument('--case_set_id', required=True)
    p.add_argument('--run_id', required=True)

    p = sub.add_parser('list_bucket_status',
                       help='Show completed vs pending bucket counts.')
    p.add_argument('--case_set_id', required=True)
    p.add_argument('--run_id', required=True)

    p = sub.add_parser('query_run_status', help='Show timing info for a run.')
    p.add_argument('--case_set_id', required=True)
    p.add_argument('--run_id', required=True)

    p = sub.add_parser('query_min_latency',
                       help='Show best latency per TuningKey.')
    p.add_argument('--case_set_id', required=True)
    p.add_argument('--run_id', required=True)

    return parser


# ---------------------------------------------------------------------------
# Source resolution (prompts when not supplied on CLI)
# ---------------------------------------------------------------------------


def _resolve_source(args):
    source = args.source
    if source is None:
        print('Select result source:')
        print('  1) local   – local JSON files')
        print('  2) spanner – Google Cloud Spanner')
        while True:
            choice = input('Enter 1 or 2: ').strip()
            if choice in ('1', 'local'):
                source = 'local'
                break
            elif choice in ('2', 'spanner'):
                source = 'spanner'
                break
            else:
                print('  Please enter 1 or 2.')

    db_path = getattr(args, 'db_path', None)
    if source == 'local' and not db_path:
        candidates = sorted(
            os.path.join('/tmp', d) for d in os.listdir('/tmp') if
            os.path.isdir(os.path.join('/tmp', d)) and 'kernel_tuner_run' in d)
        if candidates:
            print('Available local DB folders:')
            for i, c in enumerate(candidates, 1):
                print(f'  {i}) {c}')
            choice = input(
                f'Enter number or full path [default: {candidates[-1]}]: '
            ).strip()
            if choice.isdigit() and 1 <= int(choice) <= len(candidates):
                args.db_path = candidates[int(choice) - 1]
            elif choice:
                args.db_path = choice
            else:
                args.db_path = candidates[-1]
        else:
            args.db_path = input('Enter path to local DB folder: ').strip()

    return source


# ---------------------------------------------------------------------------
# Command dispatch
# ---------------------------------------------------------------------------


def _run_command(args, source, db_path=None, spanner_db=None):
    if source == 'local':
        if args.command == 'list_case_sets':
            _print_table(
                local_list_case_sets(db_path, filter_kw=args.filter_kw),
                [
                    'case_set_id', 'description', 'status', 'scan_space',
                    'num_runs'
                ],
            )

        elif args.command == 'list_runs':
            _print_table(
                local_list_runs(db_path,
                                case_set_id=args.case_set_id,
                                filter_kw=args.filter_kw),
                ['case_set_id', 'run_id', 'description', 'num_buckets'],
            )

        elif args.command == 'count_buckets':
            n = local_count_buckets(db_path, args.case_set_id, args.run_id)
            print(
                f'Total buckets for case_set_id={args.case_set_id}, run_id={args.run_id}: {n}'
            )

        elif args.command == 'list_bucket_status':
            for status, n in sorted(
                    local_list_bucket_status(db_path, args.case_set_id,
                                             args.run_id).items()):
                print(f'  {status}: {n}')

        elif args.command == 'query_run_status':
            info = local_query_run_status(db_path, args.case_set_id,
                                          args.run_id)
            if info is None:
                print('No data found.')
            else:
                for k, v in info.items():
                    print(f'  {k}: {v}')

        elif args.command == 'query_min_latency':
            _print_min_latency(
                local_query_min_latency(db_path, args.case_set_id,
                                        args.run_id))

    else:  # spanner
        if args.command == 'list_case_sets':
            _print_table(
                spanner_list_case_sets(spanner_db, filter_kw=args.filter_kw),
                [
                    'case_set_id', 'description', 'status', 'scan_space',
                    'num_runs'
                ],
            )

        elif args.command == 'list_runs':
            _print_table(
                spanner_list_runs(spanner_db,
                                  case_set_id=args.case_set_id,
                                  filter_kw=args.filter_kw),
                ['case_set_id', 'run_id', 'description', 'num_buckets'],
            )

        elif args.command == 'count_buckets':
            n = spanner_count_buckets(spanner_db, args.case_set_id,
                                      args.run_id)
            print(
                f'Total buckets for case_set_id={args.case_set_id}, run_id={args.run_id}: {n}'
            )

        elif args.command == 'list_bucket_status':
            for status, n in sorted(
                    spanner_list_bucket_status(spanner_db, args.case_set_id,
                                               args.run_id).items()):
                print(f'  {status}: {n}')

        elif args.command == 'query_run_status':
            for k, v in spanner_query_run_status(spanner_db, args.case_set_id,
                                                 args.run_id).items():
                print(f'  {k}: {v}')

        elif args.command == 'query_min_latency':
            _print_min_latency(
                spanner_query_min_latency(spanner_db, args.case_set_id,
                                          args.run_id))


# ---------------------------------------------------------------------------
# Interactive console
# ---------------------------------------------------------------------------

_COMMANDS_HELP = """\
Commands:
  set_case_set_id ID               Set default case_set_id for this session
  set_run_id ID                    Set default run_id for this session
  list_case_sets [--filter KEYWORD]
  list_runs [--case_set_id ID] [--filter KEYWORD]
  count_buckets [--case_set_id ID] [--run_id ID]
  list_bucket_status [--case_set_id ID] [--run_id ID]
  query_run_status [--case_set_id ID] [--run_id ID]
  query_min_latency [--case_set_id ID] [--run_id ID]
  help
  exit / quit
"""


def _console_loop(source, db_path, spanner_db, global_args):
    """Run an interactive REPL until the user types exit/quit."""
    parser = _build_parser()
    session_case_set_id = None
    session_run_id = None

    print('\nKernel Tuning Inspector — console mode')
    print(f'Source: {source}' +
          (f'  DB: {db_path}' if source == 'local' else ''))
    print(_COMMANDS_HELP)

    def _prompt():
        parts = ['inspect']
        if session_case_set_id:
            parts.append(f'cs={session_case_set_id}')
        if session_run_id:
            parts.append(f'run={session_run_id}')
        return '|'.join(parts) + '> '

    while True:
        try:
            line = input(_prompt()).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line in ('exit', 'quit'):
            break
        if line in ('help', '?'):
            print(_COMMANDS_HELP)
            continue

        # Handle session-state commands before argparse
        tokens = line.split()
        if tokens[0] == 'set_case_set_id':
            if len(tokens) < 2:
                print('Usage: set_case_set_id ID')
            else:
                new_case_set_id = tokens[1]
                if new_case_set_id != session_case_set_id:
                    session_run_id = None
                    if session_run_id is not None:
                        print('  run_id cleared')
                session_case_set_id = new_case_set_id
                print(f'  case_set_id set to: {session_case_set_id}')
            continue
        if tokens[0] == 'set_run_id':
            if len(tokens) < 2:
                print('Usage: set_run_id ID')
            else:
                session_run_id = tokens[1]
                print(f'  run_id set to: {session_run_id}')
            continue

        # Inject session defaults for --case_set_id / --run_id only for commands
        # that actually accept those flags.
        _cmds_with_case_set_id = {
            'list_runs', 'count_buckets', 'list_bucket_status',
            'query_run_status', 'query_min_latency'
        }
        _cmds_with_run_id = {
            'count_buckets', 'list_bucket_status', 'query_run_status',
            'query_min_latency'
        }
        cmd = tokens[0]
        if '--case_set_id' not in line and session_case_set_id is not None \
                and cmd in _cmds_with_case_set_id:
            tokens += ['--case_set_id', session_case_set_id]
        if '--run_id' not in line and session_run_id is not None \
                and cmd in _cmds_with_run_id:
            tokens += ['--run_id', session_run_id]

        # Inject global source/db flags so the sub-parser sees them
        if '--source' not in line:
            tokens = ['--source', source] + tokens
        if source == 'local' and '--db-path' not in line:
            tokens = ['--db-path', db_path] + tokens

        try:
            args = parser.parse_args(tokens)
        except SystemExit:
            # argparse already printed the error
            continue

        if args.command is None:
            print(_COMMANDS_HELP)
            continue

        try:
            _run_command(args, source, db_path=db_path, spanner_db=spanner_db)
        except Exception as e:  # pylint: disable=broad-except
            print(f'Error: {e}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = _build_parser()
    args = parser.parse_args()

    # Console mode: no subcommand supplied → enter interactive REPL
    if args.command is None:
        source = _resolve_source(args)
        db_path = getattr(args, 'db_path', None)
        spanner_db = None
        if source == 'spanner':
            spanner_db = _get_spanner_db(args.project, args.instance,
                                         args.database)
        _console_loop(source, db_path, spanner_db, args)
        return

    # Non-interactive (subcommand given directly on the CLI)
    source = _resolve_source(args)
    db_path = getattr(args, 'db_path', None)
    spanner_db = None
    if source == 'spanner':
        spanner_db = _get_spanner_db(args.project, args.instance,
                                     args.database)
    _run_command(args, source, db_path=db_path, spanner_db=spanner_db)


if __name__ == '__main__':
    main()
