#!/usr/bin/env python3
"""Standalone public Binance delisting scanner; strategy owns all trade actions.

Run: python delist_watch.py run --userdir user_data (5-minute loop)
Scan once: python delist_watch.py scan --userdir user_data
No API credentials or Freqtrade import are required.
"""
from __future__ import annotations

import argparse
import calendar
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

STATE = Path('user_data/v8/state/delist_watch_state.json')
ALERTS = STATE.with_name('delist_alerts.json')
LIVE_DELIST = STATE.with_name('live_delisting.json')

UA = {
    'User-Agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
                   '(KHTML, like Gecko) Chrome/124.0 Safari/537.36'),
    'Accept': 'application/json,text/plain,*/*',
}
FAPI = 'https://fapi.binance.com/fapi/v1/exchangeInfo'
CMS_LIST = 'https://www.binance.com/bapi/apex/v1/public/apex/cms/article/list/query'
CMS_DETAIL = 'https://www.binance.com/bapi/composite/v1/public/cms/article/detail/query'
DELIST_CATALOG = 161
PLACEHOLDER_DELIVERY = 4133404800000


def get_json(url: str, timeout: int = 25):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def iso(ms: int | None) -> str | None:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(ms / 1000)) if ms else None


# --------------------------------------------------------------------------- announcements
def _collect_text(node, out: list[str]) -> None:
    """Binance CMS bodies are a rich-text node tree, not HTML."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == 'text' and isinstance(value, str):
                out.append(value)
            else:
                _collect_text(value, out)
    elif isinstance(node, list):
        for value in node:
            _collect_text(value, out)


def list_delisting_articles(pages: int = 2) -> list[dict]:
    articles: list[dict] = []
    for page in range(1, pages + 1):
        data = get_json(f'{CMS_LIST}?type=1&pageNo={page}&pageSize=50&catalogId={DELIST_CATALOG}')
        catalogs = (data.get('data') or {}).get('catalogs') or []
        if not catalogs or not catalogs[0].get('articles'):
            break
        articles += catalogs[0]['articles']
        if len(catalogs[0]['articles']) < 50:
            break
    return articles


def article_body(code: str) -> str:
    data = get_json(f'{CMS_DETAIL}?articleCode={urllib.parse.quote(code)}')
    payload = (data.get('data') or {}).get('body')
    texts: list[str] = []
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            texts = [payload]
    if isinstance(payload, (dict, list)):
        _collect_text(payload, texts)
    return re.sub(r'\s+', ' ', ' '.join(texts))


def parse_notice(article: dict) -> dict | None:
    """Turn one 'Binance Futures Will Delist ...' article into an actionable record."""
    title = article.get('title') or ''
    if not (re.search(r'Futures', title, re.I) and re.search(r'Delist', title, re.I)):
        return None
    # COIN-M-only notices cannot affect this USDT-M account. Mixed notices continue.
    if 'COIN-M' in title and not re.search(r'USD.?-M|USDT|USDC', title):
        return None
    body = article_body(article['code'])
    # Quotes seen on USDⓈ-M. Only the USDT ones matter for this project; the rest are
    # still captured so a notice is never silently dropped because of its quote asset.
    symbols = sorted(set(re.findall(r'\b([A-Z0-9]{2,15}(?:USDT|USDC|BUSD|FDUSD))\b', body)))
    # Binance stamps are UTC. calendar.timegm (not time.mktime) keeps that true off-UTC hosts.
    stop_ts = sorted({calendar.timegm(time.strptime(s, '%Y-%m-%d %H:%M')) for s in
                      re.findall(r'(20\d\d-\d\d-\d\d \d\d:\d\d)', body)})
    effective = re.search(r'\((20\d\d-\d\d-\d\d)', title)
    halt_ms = int(stop_ts[0] * 1000) if stop_ts else None
    settle_ms = int(stop_ts[-1] * 1000) if stop_ts else None
    return {
        'title': title,
        'code': article['code'],
        'published_utc': iso(article.get('releaseDate')),
        'published_ms': article.get('releaseDate'),
        'effective_date': effective.group(1) if effective else None,
        'symbols': symbols,
        # Binance prints two stamps: trading halt, then settlement. Earliest = halt.
        'timeline_utc': [time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t)) for t in stop_ts],
        'halt_ms': halt_ms,
        'settle_ms': settle_ms,
        'halt_utc': iso(halt_ms),
        'settle_utc': iso(settle_ms),
    }


# --------------------------------------------------------------------------- exchange status
def exchange_status() -> dict:
    info = get_json(FAPI)
    perps, settling = {}, []
    for sym in info['symbols']:
        if sym.get('quoteAsset') != 'USDT' or sym.get('contractType') != 'PERPETUAL':
            continue
        perps[sym['symbol']] = sym['status']
        if sym['status'] != 'TRADING':
            settling.append({
                'symbol': sym['symbol'],
                'status': sym['status'],
                'delivery_utc': iso(sym.get('deliveryDate')),
                'contract_type': sym.get('contractType'),
            })
    return {'perps': perps, 'non_trading': settling}


# --------------------------------------------------------------------------- state
def atomic_json(path, value):
    import os
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {'seen_articles': [], 'perps': {}, 'non_trading': []}


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(STATE, state)


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def normalize_to_freqtrade_pair(raw_symbol: str) -> str | None:
    """Convert a raw exchange symbol like 'BTCUSDT' or 'IPUSDC' to freqtrade futures pair notation.

    Returns None if no known quote asset suffix is found.
    """
    m = re.fullmatch(r'([A-Z0-9]{1,20})(USDT|USDC|BUSD|FDUSD)', raw_symbol)
    if not m:
        return None
    base, quote = m.group(1), m.group(2)
    return f'{base}/{quote}:{quote}'


# --------------------------------------------------------------------------- commands
def cmd_scan(args) -> int:
    state = load_state()
    seen = set(state.get('seen_articles', []))
    watch = {p.strip() for p in (args.watch or '').split(',') if p.strip()}

    print(f'[{now_utc()}] scanning CMS catalog {DELIST_CATALOG} + exchangeInfo ...')
    plan = []

    source_errors = []
    # --- announcements (lead time lives here)
    try:
        articles = list_delisting_articles(pages=args.pages)
    except Exception as exc:                                  # noqa: BLE001
        print(f'  !! announcement source unavailable: {type(exc).__name__}: {exc}')
        source_errors.append('CMS list: ' + str(exc))
        articles = []
    notices: list[dict] = []
    new_notices = []
    cached = state.get('parsed_notices', {})
    for art in articles:
        try:
            notice = cached.get(art['code']) or parse_notice(art)
            if notice is not None and (not notice['symbols'] or not notice['halt_ms']):
                raise ValueError('Unparseable futures delisting: ' + art['code'])
        except Exception as exc:
            source_errors.append('CMS detail: ' + str(exc))
            continue
        if notice is None:
            seen.add(art.get('code'))      # non-futures delisting: remember, do not alert
            continue
        cached[art['code']] = notice
        notices.append(notice)
        if art.get('code') not in seen:
            new_notices.append(notice)
            seen.add(art['code'])

    for n in sorted(new_notices, key=lambda x: -(x['published_ms'] or 0)):
        hit = sorted(set(n['symbols']) & watch) if watch else []
        print(f"  + {n['published_utc']}  {n['title'][:86]}")
        print(f"      symbols={n['symbols'][:12]}{' ...' if len(n['symbols']) > 12 else ''}")
        print(f"      halt={n['halt_utc']}  settle={n['settle_utc']}  effective={n['effective_date']}")
        if hit:
            print(f"      *** 命中持仓清单: {hit}")
            plan.append({'pair': hit, 'action': 'block_new_entries', 'until': n['settle_utc'],
                         'then': 'exit_both_legs_immediately', 'deadline_utc': n['halt_utc'],
                         'reason': 'binance_futures_delisting_announced'})

    # --- status backstop
    try:
        snap = exchange_status()
    except Exception as exc:                                  # noqa: BLE001
        print(f'  !! exchangeInfo unavailable: {type(exc).__name__}: {exc}')
        source_errors.append('exchangeInfo: ' + str(exc))
        snap = None
    if snap:
        prev = state.get('perps') or {}
        flipped = [p for p, st in snap['perps'].items()
                   if st != 'TRADING' and prev.get(p) == 'TRADING']
        if flipped:
            print(f'  !! TRADING -> non-TRADING 翻转 {len(flipped)} 个（零提前量，属兜底）: {flipped[:12]}')
            plan.append({'pair': flipped, 'action': 'reconcile_after_settlement',
                         'reason': 'exchange_status_flipped_no_lead_time'})
        state['perps'] = snap['perps']
        state['non_trading'] = snap['non_trading']
        print(f"  exchangeInfo: {len(snap['perps'])} 个 USDT 合约，非 TRADING {len(snap['non_trading'])} 个")

    state['parsed_notices'] = cached
    state['source_errors'] = source_errors
    state['seen_articles'] = sorted(seen)
    state['last_scan_utc'] = now_utc()
    save_state(state)

    atomic_json(ALERTS, {
        'generated_utc': now_utc(),
        'new_notices': new_notices,
        'actions': plan,
        'note': ('公告中位提前 4.16 天但存在 <1 天的个例；exchangeInfo 无提前量。'
                 '本文件只产生计划，不执行平仓。'),
    })

    # --- live_delisting.json: rolling table read by the strategy
    # Regenerate from ALL parsed notices so historical announcements remain visible
    # even after they have been marked as seen.
    live_delistings: dict[str, dict] = (json.loads(LIVE_DELIST.read_text()).get('delistings', {}) if LIVE_DELIST.exists() else {})
    now_ms = int(time.time() * 1000)
    for n in cached.values():
        for raw in n.get('symbols', []):
            pair = normalize_to_freqtrade_pair(raw)
            if not pair or not pair.endswith('/USDT:USDT'):
                continue
            live_delistings[pair] = {
                'halt_ms': n.get('halt_ms'),
                'settle_ms': n.get('settle_ms'),
                'halt_utc': n.get('halt_utc'),
                'settle_utc': n.get('settle_utc'),
                'announced_utc': n.get('published_utc'),
                'announced_ms': n.get('published_ms'),
                'source': 'binance_cms_delisting',
                'title': n.get('title'),
            }
    # Persist bans across unavailable sources and the CMS page moving forward.
    if snap:
        for record in snap['non_trading']:
            if record['status'] != 'SETTLING':
                continue
            pair = normalize_to_freqtrade_pair(record['symbol'])
            if pair:
                live_delistings.setdefault(pair, dict(source='exchangeInfo_SETTLING', detected_utc=now_utc()))
    atomic_json(LIVE_DELIST, {
        'generated_utc': now_utc(), 'sources_ok': not source_errors,
        'source_errors': source_errors, 'delistings': live_delistings,
    })

    print(f'  -> {ALERTS}  新增公告 {len(new_notices)} 条，待处理动作 {len(plan)} 条')
    print(f'  -> {LIVE_DELIST}  滚动退市表 {len(live_delistings)} 条（本次解析 {len(notices)} 条，保留既有禁入记录）')
    return 1 if source_errors else 0


def cmd_blacklist(args) -> int:
    """Pairs that must not be entered right now: announced but not yet settled."""
    if not ALERTS.exists():
        print('no alerts file; run `scan` first')
        return 1
    data = json.loads(ALERTS.read_text())
    now = now_utc()
    pairs: list[str] = []
    for action in data.get('actions', []):
        if action.get('action') == 'block_new_entries' and (action.get('until') or '') > now:
            pairs += action.get('pair', [])
    print('\n'.join(sorted(set(pairs))))
    return 0


def main():
    global STATE, ALERTS, LIVE_DELIST
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', choices=['scan', 'run', 'blacklist'], default='run')
    parser.add_argument('--userdir', type=Path, default=Path('user_data'))
    parser.add_argument('--state-dir', type=Path)
    parser.add_argument('--interval', type=int, default=300)
    parser.add_argument('--pages', type=int, default=2)
    parser.add_argument('--watch')
    args = parser.parse_args()
    if args.interval < 1 or args.pages < 1:
        parser.error('interval and pages must be positive')
    state = args.state_dir or args.userdir/'v8/state'
    state.mkdir(parents=True, exist_ok=True)
    STATE = state/'delist_watch_state.json'
    ALERTS = state/'delist_alerts.json'
    LIVE_DELIST = state/'live_delisting.json'
    if args.command == 'blacklist':
        return cmd_blacklist(args)
    while True:
        try:
            code = cmd_scan(args)
            if code == 0:
                (state/'monitor_error.json').unlink(missing_ok=True)
        except Exception as exc:
            # Retain the last rolling bans; never replace an error with a healthy empty table.
            atomic_json(state/'monitor_error.json', dict(at=now_utc(), error=f'{type(exc).__name__}: {exc}'))
            print(f'Monitor failed: {exc}', file=sys.stderr, flush=True)
            code = 1
        if args.command == 'scan':
            return code
        time.sleep(args.interval)


if __name__ == '__main__':
    raise SystemExit(main())
