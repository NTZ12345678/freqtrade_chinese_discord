"""Atomic deployment releases and persistent state (no exchange/order calls)."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

BASE = Path(__file__).resolve().parent
UTC = dt.timezone.utc
PARAMS = ('gamma', 'c', 'z_mean', 'z_std', 'half_life', 'adfstat')


def positive_count(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(name + ' must be a positive integer')
    return value


def configure_capacity(config, max_pairs=None):
    """Pin K without adding a search dimension; leave room for both legs in the engine."""
    fixed = dict(config.get('atr_fixed_parameters', {}))
    k = positive_count(fixed.get('max_pairs', 18) if max_pairs is None else max_pairs, 'max_pairs')
    fixed['max_pairs'] = k
    config['atr_fixed_parameters'] = fixed
    config['max_open_trades'] = max(2 * k, config.get('max_open_trades', 0))
    return k


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def canonical(row):
    return tuple(sorted((row['pair_A'], row['pair_B'])))


def pair_key(row):
    return row['pair_A'] + '_' + row['pair_B']


def validate_snapshot(snapshot, now=None):
    now = now or dt.datetime.now(UTC)
    if not snapshot.get('complete') or snapshot.get('exchange') != 'binance' or snapshot.get('margin_mode') != 'cross':
        raise ValueError('Incomplete or non-Binance/cross snapshot')
    at = dt.datetime.fromisoformat(snapshot['asof'])
    start = dt.datetime.fromisoformat(snapshot['lookback_start'])
    if at.tzinfo is None or start.tzinfo is None or at > now or at-start != dt.timedelta(days=90):
        raise ValueError('Expected causal, timezone-aware 90-day snapshot')
    if not snapshot.get('selected'):
        raise ValueError('No selected combinations; do not replace working release')
    seen = set()
    for row in snapshot['selected']:
        legs = canonical(row)
        if legs[0] == legs[1] or legs in seen or not all(p.endswith('/USDT:USDT') for p in legs):
            raise ValueError('Invalid or duplicate pair combination')
        seen.add(legs)
        if not all(isinstance(row.get(k), (int, float)) and math.isfinite(row[k]) for k in PARAMS):
            raise ValueError('Invalid scientific parameters')
        if row['gamma'] <= 0 or row['z_std'] <= 0 or not (0 < row['half_life'] < 500) or row['adfstat'] >= -6:
            raise ValueError('Pair fails the agreed getpara filters')
    return snapshot


def publish(root, snapshot):
    root = Path(root)
    validate_snapshot(snapshot)
    release_id = digest(snapshot)
    release = root / 'releases' / release_id
    release.mkdir(parents=True, exist_ok=True)
    atomic(release / 'snapshot.json', snapshot)
    atomic(release / 'schedule.json', [snapshot])
    pointer = dict(schema=1, release_id=release_id, published_utc=dt.datetime.now(UTC).isoformat())
    # One pointer is the commit point; readers never see a half-published release.
    atomic(root / 'current.json', pointer)
    return pointer


def load_release(root, now=None):
    root = Path(root)
    pointer = json.loads((root / 'current.json').read_text())
    ident = pointer['release_id']
    if len(ident) != 64 or any(c not in '0123456789abcdef' for c in ident):
        raise ValueError('Invalid release identifier')
    path = root / 'releases' / ident
    snapshot = json.loads((path / 'snapshot.json').read_text())
    if digest(snapshot) != ident or json.loads((path / 'schedule.json').read_text()) != [snapshot]:
        raise ValueError('Release checksum/schedule mismatch')
    validate_snapshot(snapshot, now)
    return pointer, snapshot, path / 'schedule.json'


def merge_combinations(old, new, retired=(), blocked_coins=()):
    """Admit only disjoint relationships; rejected new pairs are not queued."""
    result, seen = [], set()
    retired = {tuple(x) for x in retired}
    for row in old:
        key = canonical(row)
        if key in seen or key in retired:
            continue
        seen.add(key)
        result.append(row)
    occupied = set(blocked_coins) | {p for row in result for p in canonical(row)}
    for row in new:
        key = canonical(row)
        if key in seen or key in retired or occupied.intersection(key):
            continue
        result.append(row)
        seen.add(key)
        occupied.update(key)
    return result
