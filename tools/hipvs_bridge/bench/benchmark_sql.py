#!/usr/bin/env python3
"""Development SQL benchmark. Official scores still require the organizer's runner."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import uuid

FROZEN = '634888d9b037153fa143d2c913ba4b25503e1d62'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def quality(results, truth, valid_ids, k):
    if len(results) != len(truth) or not results:
        raise ValueError('query/ground-truth count mismatch')
    hits = 0
    integrity = True
    for actual, expected in zip(results, truth):
        if len(expected) < k or len(set(expected[:k])) != k:
            raise ValueError('ground truth needs k distinct IDs per query')
        integrity &= len(actual) == k and len(set(actual)) == k and all(i in valid_ids for i in actual)
        hits += len(set(actual) & set(expected[:k]))
    return {'integrity': bool(integrity), 'recall': hits / (len(results) * k)}


def positive_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def compare(baseline, candidate):
    if baseline['variant'] != 'baseline' or candidate['variant'] != 'candidate':
        raise ValueError('expected baseline and candidate measurements')
    if baseline['commit'] != FROZEN:
        raise ValueError('baseline must identify the frozen V5 commit')
    for key in ('dataset', 'rows', 'dim', 'nq', 'inputs'):
        if baseline[key] != candidate[key]:
            raise ValueError('unpaired input: ' + key)
    gates = []
    ratios = []
    if baseline['dataset'] == 'SIFT250K':
        minima = {10: .95, 100: .95}
    elif baseline['dataset'] == 'Hidden250K':
        minima = {10: .90, 100: .85}
    else:
        minima = {10: 0., 100: 0.}
    for k in (10, 100):
        a, b = baseline['queries'][str(k)], candidate['queries'][str(k)]
        if not positive_number(a['qps']) or not positive_number(b['qps']):
            raise ValueError('invalid QPS')
        if not a['integrity']:
            raise ValueError('baseline has invalid results')
        if not b['integrity']:
            gates.append(f'result_integrity@{k}')
        for value in (a['recall'], b['recall']):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('invalid recall')
        if b['recall'] + 1e-12 < minima[k]:
            gates.append(f'recall@{k}_minimum')
        if b['recall'] + .005 + 1e-12 < a['recall']:
            gates.append(f'recall@{k}_baseline_delta')
        ratios.append(b['qps'] / a['qps'])
    query_speedup = math.sqrt(ratios[0] * ratios[1])
    if query_speedup < .9:
        gates.append('query_regression')
    if not all(positive_number(x['build_seconds']) for x in (baseline, candidate)):
        raise ValueError('invalid build duration')
    return {
        'dataset': baseline['dataset'],
        'build_speedup': baseline['build_seconds'] / candidate['build_seconds'],
        'query_speedup': query_speedup,
        'measured_gate_failures': gates,
        'status': 'development_measurement_requires_GPU_path_and_resource_checks',
        'official_score': None,
    }


def measure(args):
    try:
        import pymysql
    except ImportError as exc:
        raise SystemExit('SQL benchmark needs PyMySQL in the test environment; no build dependency is added') from exc
    if not re.fullmatch(r'[0-9a-f]{40}', args.commit):
        raise ValueError('--commit must be the full source commit of the running binary')
    if args.variant == 'baseline' and args.commit != FROZEN:
        raise ValueError('baseline commit must be frozen V5')
    queries = json.loads(Path(args.queries).read_text())
    truth = json.loads(Path(args.truth).read_text())
    if args.nq <= 0 or len(queries) < args.nq or len(truth) != len(queries):
        raise ValueError('need nq queries with matching ground truth')
    queries, truth = queries[:args.nq], truth[:args.nq]
    if args.dim <= 0 or not all(isinstance(q, list) and len(q) == args.dim and
                                all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in q)
                                for q in queries):
        raise ValueError('query dimension mismatch')
    # Validate all JSON numbers, including NaN/Infinity, before touching SQL.
    probes = [json.dumps(q, allow_nan=False, separators=(',', ':')) for q in queries]
    valid_ids = set()
    rows = 0
    with open(args.base_tsv) as f:
        for line in f:
            identifier = int(line.split('\t', 1)[0])
            rows += 1
            valid_ids.add(identifier)
    if rows != len(valid_ids) or rows < 100:
        raise ValueError('base TSV needs >=100 rows with distinct integer IDs')
    if not quality([row[:100] for row in truth], truth, valid_ids, 100)['integrity']:
        raise ValueError('ground truth contains IDs absent from the base TSV')
    inputs = {key: digest(path) for key, path in [('base', args.base_tsv), ('queries', args.queries), ('truth', args.truth)]}
    binary_hash = digest(args.binary)
    database = 'amd_bench_' + uuid.uuid4().hex
    conn = pymysql.connect(host=args.host, port=args.port, user=args.user,
                           password=os.environ.get('MYSQL_PWD', ''), autocommit=True,
                           local_infile=True, read_timeout=900, write_timeout=900)
    result = {'variant': args.variant, 'commit': args.commit, 'binary_sha256': binary_hash,
              'dataset': args.dataset, 'rows': rows, 'dim': args.dim, 'nq': args.nq,
              'inputs': inputs, 'database': database, 'queries': {},
              'gpu_path_verified': False, 'resource_limits_verified': False,
              'measurement': 'client_wall_time_with_first_query_in_build'}
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT version()')
            result['server_version'] = cur.fetchone()[0]
            # Always creates a fresh database. Never drops or reuses existing data.
            cur.execute('CREATE DATABASE ' + database)
            cur.execute('USE ' + database)
            cur.execute('SET ob_query_timeout=900000000')
            cur.execute(f'CREATE TABLE items(id BIGINT PRIMARY KEY, v VECTOR({args.dim}))')
            cur.execute("LOAD DATA LOCAL INFILE %s INTO TABLE items FIELDS TERMINATED BY '\\t' LINES TERMINATED BY '\\n' (id,v)", (str(Path(args.base_tsv).resolve()),))
            cur.execute('SELECT COUNT(*) FROM items')
            if cur.fetchone()[0] != rows:
                raise ValueError('loaded row count differs from TSV')
            lib = 'vsag' if args.variant == 'cpu' else 'cuvs'
            select = 'SELECT id FROM items ORDER BY l2_distance(v,%s) APPROXIMATE LIMIT '
            start = time.perf_counter()
            cur.execute(f'CREATE VECTOR INDEX idx_items ON items(v) WITH (distance=l2,type=hnsw,lib={lib},m=16,ef_construction=200,ef_search=64)')
            cur.execute(select + '10', (probes[0],))
            first = [row[0] for row in cur.fetchall()]
            result['build_seconds'] = time.perf_counter() - start
            if not quality([first], [truth[0]], valid_ids, 10)['integrity']:
                raise ValueError('first query returned invalid IDs')
            for k in (10, 100):
                found, latencies = [], []
                group_start = time.perf_counter()
                for probe in probes:
                    query_start = time.perf_counter()
                    cur.execute(select + str(k), (probe,))
                    found.append([row[0] for row in cur.fetchall()])
                    latencies.append(time.perf_counter() - query_start)
                elapsed = time.perf_counter() - group_start
                metrics = quality(found, truth, valid_ids, k)
                metrics.update(qps=len(probes) / elapsed,
                               p95_seconds=sorted(latencies)[math.ceil(len(latencies)*.95)-1],
                               seconds=elapsed)
                result['queries'][str(k)] = metrics
    finally:
        conn.close()
    # Preserve the isolated database for restart/CPU-path diagnostics. Use a new
    # server process for the other variant and remove only this test DB later.
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    m = sub.add_parser('measure')
    m.add_argument('--variant', choices=['baseline', 'candidate', 'cpu'], required=True)
    m.add_argument('--host', default='127.0.0.1')
    m.add_argument('--port', type=int, default=2881)
    m.add_argument('--user', default='root')
    m.add_argument('--commit', required=True)
    m.add_argument('--binary', required=True)
    m.add_argument('--dataset', required=True)
    m.add_argument('--dim', type=int, required=True)
    m.add_argument('--nq', type=int, default=4096)
    for name in ('base-tsv', 'queries', 'truth', 'output'):
        m.add_argument('--' + name, required=True)
    c = sub.add_parser('compare')
    for name in ('baseline', 'candidate', 'output'):
        c.add_argument('--' + name, required=True)
    args = p.parse_args()
    if args.command == 'measure':
        result = measure(args)
    else:
        result = compare(json.loads(Path(args.baseline).read_text()), json.loads(Path(args.candidate).read_text()))
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
