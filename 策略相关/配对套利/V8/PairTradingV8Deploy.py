"""V8 Binance USDT-M cross pair strategy: capacity K and shared trading lifecycle.

PairTradingV8Deploy owns admission, signal/budget planning, paired fills/exits,
N-day permanent retirement, delisting bans and persisted deployment health/state.
Its production constructor accepts dry_run/live only and installs no research hooks.

PairTradingV8Aligned (in its own file) reuses that lifecycle with historical publications,
announcements, frozen market precision, settlement candles and native replay adapters.
Each leg retains its entry parameters. Old admitted relationships keep their original
coefficients; new relationships sharing an occupied or banned coin are rejected.

N14 / 1x / K18 deployment defaults can be overridden in configuration. V7 is archived;
persisted trade/state schemas and the original N/leverage search remain compatible.
V8 acceptance: research/maintenance/v8_capacity_20261011/.
"""


import builtins
import datetime as dt
import hashlib
import os
import tempfile
from copy import deepcopy
from datetime import timedelta
import json
import logging
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
from freqtrade.persistence import Trade
from freqtrade.strategy import CategoricalParameter, IntParameter
from freqtrade.strategy.interface import IStrategy

HERE = Path(__file__).resolve().parent
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


def configure_deployment(config):
    """Supply production defaults without creating releases or replacing explicit settings."""
    fixed = dict(n_days=14, leverage_choice=1, max_pairs=18)
    fixed.update(config.get('atr_fixed_parameters', {}))
    config['atr_fixed_parameters'] = fixed
    root = Path(config.get('user_data_dir', 'user_data')) / 'v8'
    mode = getattr(config.get('runmode'), 'value', config.get('runmode'))
    config.setdefault('atr_release_root', config.get('atr_parameters', str(root / 'parameters.json')))
    config.setdefault('atr_state_path', str(root / 'state' / ('live.json' if mode == 'live' else 'dryrun.json')))
    config.setdefault('atr_live_delisting', str(root / 'state/live_delisting.json'))
    if not config.get('atr_schedule'):
        try:
            _, _, schedule = load_release(config['atr_release_root'])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError('V8 needs a valid parameter release at '
                             + str(config['atr_release_root'])
                             + '; publish getpara results before starting') from exc
        config['atr_schedule'] = str(schedule)


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
    if set(snapshot) == {'asof', 'selected'}:
        snapshot = dict(snapshot, complete=True, exchange='binance', margin_mode='cross',
                        lookback_start=(dt.datetime.fromisoformat(snapshot['asof'])-dt.timedelta(days=90)).isoformat())
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
        if not all(not isinstance(row.get(k), bool) and isinstance(row.get(k), (int, float)) and math.isfinite(row[k]) for k in PARAMS):
            raise ValueError('Invalid scientific parameters')
        if row['gamma'] <= 0 or row['z_std'] <= 0 or not (0 < row['half_life'] < 500) or row['adfstat'] >= -6:
            raise ValueError('Pair fails the agreed getpara filters')
    return snapshot


def publish(root, snapshot):
    root = Path(root)
    validate_snapshot(snapshot)
    if root.suffix == '.json':
        payload = parameter_payload(snapshot)
        history = root.with_name(root.stem + '_history')
        stamp = dt.datetime.fromisoformat(payload['asof']).astimezone(UTC).strftime('%Y%m%dT%H%M%SZ')
        atomic(history / (stamp + '_' + digest(payload)[:12] + '.json'), payload)
        atomic(root, payload)
        return dict(release_id=digest(payload))
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
    if root.suffix == '.json':
        snapshot = json.loads(root.read_text())
        validate_snapshot(snapshot, now)
        return dict(release_id=digest(snapshot)), snapshot, root
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


def parameter_payload(snapshot):
    """The public getpara file contains only the cutoff and fitted selections."""
    validate_snapshot(snapshot)
    selected = [{name: row[name] for name in ('pair_A', 'pair_B', *PARAMS)}
                for row in snapshot['selected']]
    occupied = set()
    for row in selected:
        if occupied.intersection(canonical(row)):
            raise ValueError('A getpara publication must contain disjoint combinations')
        occupied.update(canonical(row))
    return dict(asof=snapshot['asof'], selected=selected)


def parameter_path(config):
    return Path(config.get('atr_parameters',
                str(Path(config.get('user_data_dir', 'user_data')) / 'v8/parameters.json')))


def read_schedule(path):
    """Read one compact file, archived compact files, or a legacy research schedule."""
    path = Path(path)
    files = sorted(path.glob('*.json')) if path.is_dir() else [path]
    result = {}
    for file in files:
        data = json.loads(file.read_text())
        for snapshot in data if isinstance(data, list) else [data]:
            validate_snapshot(snapshot)
            at = dt.datetime.fromisoformat(snapshot['asof'])
            if at in result and result[at] != snapshot:
                raise ValueError('Conflicting publications have the same cutoff; select a file explicitly')
            result[at] = snapshot
    if not result:
        raise ValueError('No parameter publications found at ' + str(path))
    return [result[at] for at in sorted(result)]


def default_config(pairs):
    """Public Binance cross paper configuration, with no authentication material."""
    return dict(
        dry_run=True, dry_run_wallet=1000, trading_mode='futures', margin_mode='cross',
        timeframe='5m', stake_currency='USDT', stake_amount='unlimited', max_open_trades=36,
        tradable_balance_ratio=.99, fiat_display_currency='USD',
        cancel_open_orders_on_exit=False,
        unfilledtimeout=dict(entry=10, exit=10, exit_timeout_count=0, unit='minutes'),
        entry_pricing=dict(price_side='other', use_order_book=True, order_book_top=1,
                           check_depth_of_market=dict(enabled=False, bids_to_ask_delta=1)),
        exit_pricing=dict(price_side='other', use_order_book=True, order_book_top=1),
        exchange=dict(name='binance', key='', secret='', ccxt_config={}, ccxt_async_config={},
                      pair_whitelist=sorted(pairs), pair_blacklist=[]),
        pairlists=[dict(method='StaticPairList')],
        api_server=dict(enabled=False, listen_ip_address='127.0.0.1', listen_port=8080,
                        username='', password=''),
        telegram=dict(enabled=False, token='', chat_id=''), internals=dict(process_throttle_secs=5))


log = logging.getLogger(__name__)


class PairTradingV8Deploy(IStrategy):
    """Shared trading lifecycle; production inputs and persistent deployment state."""


    # --- freqtrade contract (inherited from the OKX-era base + ATR450) ----------
    INTERFACE_VERSION = 3
    can_short = True
    startup_candle_count = 900  # >= 72 complete hourly candles before the first signal
    # Reconsider retirement/publications even within the same candle.
    timeframe = "5m"
    process_only_new_candles = False
    stoploss = -100             # no stop loss at all: the n-day timeout is the exit rule
    trailing_stop = False
    use_custom_stoploss = False
    minimal_roi = {}
    use_exit_signal = True
    exit_profit_only = False
    order_types = {
        "entry": "market",
        "exit": "market",
        "emergency_exit": "market",
        "force_entry": "market",
        "force_exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
        "stoploss_on_exchange_interval": 60,
        "stoploss_on_exchange_market_ratio": 0.99,
    }
    # --- pair gate -------------------------------------------------------------
    Zscore_entry = 1      # enter when |z| exceeds this
    Zscore_exit = 0.8     # reversion exit
    Pair_profit_target = 0.06
    Zscore_entry1 = 1.1 * Zscore_entry
    # Declared by pair_trading_run_V6 and still declared by the frozen research packages.
    # Nothing in this file reads them; they are kept so runtime.json consumers that look
    # them up do not turn into AttributeError.
    LEVERAGE = 1.0
    Zscore_stop = 3
    can_trade_usdt = False
    min_consistency_score = 0.002
    zscore_mean_window = 10

    # --- holding timeout, leverage and manually configured combination capacity --
    n_days = CategoricalParameter([7, 10, 14, 21, 30], default=14, space="atr")
    leverage_choice = CategoricalParameter([1, 2, 3], default=1, space="atr")
    # Manual K deliberately preserves the existing N/leverage search (optimize=False).
    # Fixed configuration may pin any positive integer; this domain is not searched.
    max_pairs = IntParameter(1, 40, default=18, space="atr", optimize=False, load=False)

    # --- acceptance-test fixture only (delist_smoke.py) -----------------------
    # Two disjoint allow-lists for config["atr_overrides"]. A performance run must never
    # set config["atr_overrides"]: it exists so delist_smoke.py can pin a deterministic
    # settlement scenario, and a bad name is a hard error rather than a silent mutation.
    #   OVERRIDABLE            -> plain scalars, set directly on the instance
    #   OVERRIDABLE_PARAMETERS -> CategoricalParameter objects, `.value` is pinned
    OVERRIDABLE = ("Zscore_entry", "Zscore_entry1", "Zscore_exit", "Pair_profit_target",
                   "zscore_mean_window", "min_consistency_score")
    OVERRIDABLE_PARAMETERS = ("n_days", "leverage_choice", "max_pairs")
    # Accepted-and-ignored in config["atr_fixed_parameters"]. These pinned the ATR arm,
    # which no longer exists. Listed explicitly so a stale config keeps loading while a
    # genuinely unknown name still raises.
    RETIRED_PARAMETERS = ("atr_period", "atr_multiple", "hard_loss_pct", "timeout_days")


    def __init__(self, config):
        self._validate_mode(config)
        configure_capacity(config)
        self._initialize_inputs(config)
        # IStrategy.__init__, previously reached through pair_trading_run_V6. That parent
        # demanded an even-length, duplicate-free whitelist because it traded in pairs;
        # this class keeps its own candidate table, so the config is passed through as is.
        super().__init__(config)
        self.config = config
        # Pin deployment defaults on this instance; an offline instance must keep
        # its declared search space even after another bot has been constructed.
        for name in self.OVERRIDABLE_PARAMETERS:
            setattr(self, name, deepcopy(getattr(self, name)))
        schedule_path = self._schedule_path(config)
        self.schedule = read_schedule(schedule_path)
        self.schedule.sort(key=lambda s: s["asof"])
        for name, value in config.get("atr_fixed_parameters", {}).items():
            # atr_period / atr_multiple / hard_loss_pct / timeout_days pinned parameters
            # that only the (now removed) ATR arm read. A config written before this
            # refactor still lists them, and the deployment config on the server does too;
            # ignoring exactly that set keeps it loadable. Any other unknown name is still
            # a hard error -- silent parameter mismatch is how this project gets burned.
            if name in self.RETIRED_PARAMETERS:
                log.warning("Ignoring retired atr_fixed_parameters entry %s=%r "
                            "(ATR arm removed; nothing reads it)", name, value)
                continue
            parameter = getattr(self, name)
            parameter.default = value
            parameter.value = value
            parameter.optimize = False
            parameter.load = False
        self._last_time = None
        # --- live delisting table (server-side rolling signal) ---
        # Read by the strategy in live/dry-run. In backtest it is normally empty
        # unless atr_live_delisting_test is set for integration verification.
        self.live_delist_path = Path(config.get(
            "atr_live_delisting", str(HERE.parent.parent / "research/binance_cross_v1/progress/live_delisting.json")))
        self._live_delists = {}
        self._force_live_delisting = bool(config.get("atr_live_delisting_test", False))
        # Test-fixture only: deterministic overrides for the acceptance smoke
        # (delist_smoke.py). Deliberately an allow-list, so a config typo cannot silently
        # change an arbitrary strategy attribute. Never set in a performance run.
        for name, value in (config.get("atr_overrides") or {}).items():
            if name in self.OVERRIDABLE:
                setattr(self, name, value)
            elif name in self.OVERRIDABLE_PARAMETERS:
                parameter = getattr(self, name)
                parameter.default = value
                parameter.value = value
                parameter.optimize = False
                parameter.load = False
            else:
                raise ValueError("atr_overrides only supports "
                                 + ", ".join(self.OVERRIDABLE + self.OVERRIDABLE_PARAMETERS))
        self.config['max_open_trades'] = max(self.config['max_open_trades'], 2 * self._pair_capacity())
        self._reset()
        if not self._offline or self._force_live_delisting:
            self._load_live_delists()


    def _validate_mode(self, config):
        mode = getattr(config.get('runmode'), 'value', config.get('runmode'))
        if mode not in ('dry_run', 'live'):
            raise ValueError('Deployment class requires dry_run/live; use research classes for backtests')
        if (config.get('exchange', {}).get('name'), config.get('trading_mode'), config.get('margin_mode')) != ('binance', 'futures', 'cross'):
            raise ValueError('V8 requires Binance USDT perpetual futures in cross mode')
        configure_deployment(config)

    def _initialize_inputs(self, config):
        # Deployment never installs historical exchange/custom-data hooks.
        self._offline = False
        self._settlements = {}
        self._state_path = Path(config['atr_state_path'])
        self._engine_delisting = {}

    def _schedule_path(self, config):
        return Path(config['atr_schedule'])

    def _settlement_exit(self, trade, current_time):
        return None

    def _settlement_engine_mismatches(self):
        return []

    def _reset(self):
        self._pool = []
        self._retired_store = set()
        self._banned_coins = set()
        self._state_error = self._delist_error = self._release_error = None
        self._release_id = self._latest_asof = None
        self._bad_trades = []
        self._last_state = None
        self._health_at = 0
        self._loop_time = None
        self._loop_ready = False
        self._round_trade_ids = set()
        self._round_entry_intents = set()
        self._signal_owner = {}
        self.tradable_pairs = {}
        self.stake_allocations = {}
        self.entry_pair_keys = set()
        self._profit_exit = set()
        self._open = []
        self._groups = {}
        initial = float(self.config.get("dry_run_wallet", 1000))
        self._metrics = dict(initial=initial, peak=initial, max_drawdown=0.0, last_equity=initial,
                             detached_legs=0, atr_exits=0, hard_exits=0, timeout_exits=0, samples=0,
                             delisting_settlements=0,
                             settlement_frames_injected=len(getattr(self, "_injected_frames", {})),
                             settlement_engine_checked=len(getattr(self, "_engine_delisting", {}) or {}),
                             settlement_engine_mismatches=self._settlement_engine_mismatches())
        # This block belongs to initialisation only. It used to be duplicated at the tail of
        # the (now removed) _detach_pair_trades, which wiped the whole metrics dict every time
        # a pair was decoupled. Every later _plan() then raised KeyError while the backtest
        # still finished, so the corruption only surfaced in runtime.json.
        self._metrics["parameters"] = {name: int(getattr(self, name).value) for name in
            ("n_days", "leverage_choice", "max_pairs")}
        if self.config.get("atr_diagnostic", False):
            self._metrics["continuation_events"] = {}
        builtins._atr450_metrics = self._metrics
        self._plans = {}
        self._pending_exit = {}
        self._metrics.update(model_version="v8-fixed-capacity", partial_entry_failures=0,
            retired_combinations=0, entry_plan_rejections=0, max_pair_budget_fraction=0.0,
            # Why a pair did not enter, counted per reason. The strategy can fail to open
            # *any* trade for a whole run with no error anywhere (a mismatched whitelist /
            # parameter set does exactly that), and this is the only place that says which
            # gate rejected it. Surfaced through runtime.json -> equity.
            plan_rejections={})
        self._shadow_book = {}
        self._shadow_closed_at = {}
        self._metrics.update(shadow_events=[], shadow_opened=0, shadow_promoted=0,
            shadow_timeout_retired=0, shadow_active=0)


    @staticmethod
    def _pair_key_from_tag(tag):
        if not isinstance(tag, str):
            return None
        parts = tag.split("_", 2)
        if len(parts) != 3 or parts[0] != "entry" or parts[1] not in ("long", "short"):
            return None
        legs = parts[2].split("_")
        if len(legs) != 2 or not all("/" in leg for leg in legs) or legs[0] == legs[1]:
            return None
        return parts[2]


    def _trade_params(self, trade):
        return trade.get_custom_data("atr450_params", default=None)


    def populate_exit_trend(self, dataframe, metadata):
        dataframe["exit_long"] = pd.Series(0, index=dataframe.index, dtype="Int8")
        dataframe["exit_short"] = pd.Series(0, index=dataframe.index, dtype="Int8")
        return dataframe


    def _base_order_filled(self, pair, trade, order, current_time, **kwargs):
        key = self._pair_key_from_tag(trade.enter_tag)
        if not key:
            return
        if order.ft_order_side == trade.entry_side:
            params = self.tradable_pairs.get(key)
            if params:
                trade.set_custom_data("atr450_params", dict(params))
                trade.set_custom_data("atr450_original_stake", float(trade.stake_amount))
                mates = self._groups.get(key, [])
                opened = min([trade.open_date_utc, *[t.open_date_utc for t in mates]])
                trade.set_custom_data("atr450_opened", opened.isoformat())
        elif order.ft_order_side == trade.exit_side:
            if getattr(order, "safe_amount", 0) < trade.amount:
                return
            reason = getattr(order, "ft_order_tag", None) or trade.exit_reason or ""
            if reason == "pair_profit":
                self._profit_exit.add(key)
            if reason.startswith("delist_settlement"):
                # Settlement is not an ATR/timeout/stop exit; count it separately so the
                # arm-level exit mix stays interpretable.
                self._metrics["delisting_settlements"] += 1
                return
            field = {"atr_hard_loss": "hard_exits", "atr_timeout": "timeout_exits", "timeout_immediate": "timeout_exits", "trailing_stop_loss": "atr_exits", "stop_loss": "atr_exits", "atr_trailing_exit": "atr_exits"}.get(reason)
            detached = trade.get_custom_data("atr450_detached", default=None)
            if field == "atr_exits" and detached and detached.get("binding") == "hard":
                field = "hard_exits"
            if field and trade.get_custom_data("atr450_detached", default=None):
                self._metrics[field] += 1

    def _base_custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        # B6: settlement outranks everything -- detached legs (immediate/ATR arms),
        # z-reversion and pair profit. The contract stops existing at this bar.
        settlement = self._settlement_exit(trade, current_time)
        if settlement:
            return settlement
        # A detached leg (n-day timeout reached, or a live delisting ban) leaves now.
        # This strategy has no stop loss: the n-day timeout IS the exit rule.
        if trade.get_custom_data("atr450_detached", default=None):
            return "timeout_immediate"
        params = self._trade_params(trade)
        if not params:
            return None
        key = params["pair_A"] + "_" + params["pair_B"]
        if key in self._profit_exit:
            return "pair_already_exit"
        a, b = self._row(params["pair_A"], current_time), self._row(params["pair_B"], current_time)
        if a is None or b is None:
            return None
        z = (a["close"] - params["gamma"] * b["close"] - params["c"] - params["z_mean"]) / params["z_std"]
        if abs(z) < self.Zscore_exit:
            return "pair_z_reversion"
        mates = self._groups.get(key, [])
        total = sum(t.stake_amount for t in mates)
        raw_profit = 0.0
        for mate in mates:
            price = a["close"] if mate.pair == params["pair_A"] else b["close"]
            ratio = (1 - price / mate.open_rate) if mate.is_short else (price / mate.open_rate - 1)
            raw_profit += ratio * mate.leverage * mate.stake_amount
        if total and raw_profit / total > self.Pair_profit_target and current_time > trade.open_date_utc + timedelta(days=1):
            return "pair_profit"
        return None


    @property
    def _exchange(self):
        return self.dp._exchange

    def _minimum(self, pair, price, leverage):
        return max(5.0, float(self._exchange.get_min_pair_stake_amount(pair, price, -.05, leverage) or 0))


    def _pair_capacity(self):
        return positive_count(self.max_pairs.value, 'max_pairs')

    def _capacity_keys(self, opened, current_time):
        # Pending entry orders and remaining exit legs still have an open Trade.
        used = {self._pair_key_from_tag(t.enter_tag) or ('invalid', t.id) for t in opened}
        used.update(k for k, p in self._plans.items()
            if p['valid'] and p['at'] == current_time)
        return used

    def _entry_priority(self, key):
        row = self.tradable_pairs[key]
        return (row['half_life'], row['adfstat'], canonical(row), key)

    def _plan(self, key, current_time):
        if not self._loop_ready or self._loop_time is None:
            return dict(valid=False, leverage=1, stakes={}, sides={})
        # Reserve in statistical rank order, regardless of the engine's coin traversal.
        # Invalid/minimum-size plans reserve nothing, so lower-ranked signals can proceed.
        if key in self.entry_pair_keys:
            for candidate in sorted(self.entry_pair_keys, key=self._entry_priority):
                self._build_plan(candidate, self._loop_time)
        return self._build_plan(key, self._loop_time)

    def _build_real_plan(self, key, current_time):
        if not self._loop_ready or self._loop_time is None:
            return dict(valid=False, leverage=1, stakes={}, sides={})
        current_time = self._loop_time
        cached = self._plans.get(key)
        if cached is not None and cached["at"] == current_time:
            return cached
        plan = dict(at=current_time, valid=False, filled=set(), stakes={}, sides={}, why="")
        self._plans[key] = plan
        if key not in self.entry_pair_keys:
            return self._why(plan, "not_in_entry_keys")
        params = self.tradable_pairs[key]
        pairs = (params["pair_A"], params["pair_B"])
        opened = list(Trade.get_trades_proxy(is_open=True))
        if len(self._capacity_keys(opened, current_time)) >= self._pair_capacity():
            return self._why(plan, 'no_pair_capacity')
        reserved_slots = sum(len(p["stakes"])-len(p["filled"]) for k,p in self._plans.items()
            if k != key and p["valid"] and p["at"] == current_time)
        if self.config["max_open_trades"]-len(opened)-reserved_slots < 2:
            return self._why(plan, "no_free_slots")
        if any(t.pair in pairs for t in opened):
            return self._why(plan, "pair_already_open")
        previous = [self._row(p,current_time) for p in pairs]
        current = [self._row(p,current_time,at_open=True) for p in pairs]
        if any(row is None for row in previous+current):
            return self._why(plan, "missing_row")
        if any(row.get("volume",1) <= 0 for row in previous):
            return self._why(plan, "zero_volume")
        prices = [r["open"] for r in current]
        z=(previous[0]["close"]-params["gamma"]*previous[1]["close"]-params["c"]-params["z_mean"])/params["z_std"]
        if z > self.Zscore_entry:
            sides=("short","long")
        elif z < -self.Zscore_entry:
            sides=("long","short")
        else:
            return self._why(plan, "no_signal")
        fee = float(self.config.get("fee",.0005))
        reserved_cash = sum(p["stakes"][pair]*(1+p["leverage"]*fee)
            for k,p in self._plans.items() if k != key and p["valid"] and p["at"] == current_time
            for pair in p["stakes"] if pair not in p["filled"])
        cash = max(0,self.wallets.get_available_stake_amount()-reserved_cash)
        lev = float(self.leverage_choice.value)
        requested = self.stake_allocations.get(key,0)
        denom = prices[0]+params["gamma"]*prices[1]
        if requested <= 0 or denom <= 0 or not all(math.isfinite(p) and p>0 for p in prices):
            return self._why(plan, f"bad_budget requested={requested} denom={denom} prices={prices}")
        for pair, share in zip(pairs,(prices[0]/denom,params["gamma"]*prices[1]/denom)):
            lev=min(lev,float(self._exchange.get_max_leverage(pair,requested*share)))
        if lev < 1:
            return self._why(plan, "lev_below_1")
        budget=min(requested,cash/(1+lev*fee)*.999)
        target_a=budget*lev/denom
        quantities=(self._quantity(pairs[0],target_a),self._quantity(pairs[1],target_a*params["gamma"]))
        stakes=[q*price/lev for q,price in zip(quantities,prices)]
        for pair, price, amount in zip(pairs,prices,stakes):
            if amount < self._minimum(pair,price,lev) or amount > self._exchange.get_max_pair_stake_amount(pair,price,lev):
                return self._why(plan, (f"stake_out_of_bounds {pair} amount={amount} "
                                        f"min={self._minimum(pair,price,lev)} prices={prices} lev={lev}"))
        if sum(stakes)*(1+lev*fee) > cash:
            return self._why(plan, f"over_cash {sum(stakes)*(1+lev*fee)}>{cash}")
        plan["why"] = "valid"
        plan.update(valid=True,leverage=lev,stakes=dict(zip(pairs,stakes)),sides=dict(zip(pairs,sides)))
        equity=self._metrics["last_equity"]
        if equity>0:
            self._metrics["max_pair_budget_fraction"]=max(self._metrics["max_pair_budget_fraction"],sum(stakes)/equity)
        return plan

    def _why(self, plan, reason):
        """Record the first gate that refused this plan, in the plan and in the metrics."""
        plan["why"] = reason
        bucket = reason.split(" ")[0]
        counters = self._metrics.setdefault("plan_rejections", {})
        counters[bucket] = counters.get(bucket, 0) + 1
        return plan

    def leverage(self,pair,current_time,entry_tag=None,max_leverage=3,**kwargs):
        key=self._pair_key_from_tag(entry_tag)
        plan=self._plan(key,current_time)
        if plan["valid"] and max_leverage < plan["leverage"]:
            plan["valid"]=False
        return plan.get("leverage",1) if plan["valid"] else 1

    def custom_stake_amount(self,pair,current_time,current_rate,proposed_stake,min_stake,max_stake,
                           leverage,entry_tag,side,**kwargs):
        key=self._pair_key_from_tag(entry_tag)
        plan=self._plan(key,current_time)
        amount=plan["stakes"].get(pair,0)
        if not plan["valid"] or plan["sides"].get(pair)!=side or plan.get("leverage")!=leverage:
            return 0
        if amount < (min_stake or 0) or amount > max_stake:
            plan["valid"]=False
            self._why(plan, "stake_out_of_wallet_bounds")
            self._metrics["entry_plan_rejections"] += 1
            return 0
        return amount

    def bot_start(self, **kwargs):
        # Load before the parent saves its startup state, so a restart cannot
        # erase a paper clock. Offline epochs remain entirely memory-local.
        if not self._offline and self._state_path.exists():
            import json
            saved = json.loads(self._state_path.read_text())
            for key, paper in saved.get('shadow_positions', {}).items():
                self._validate_shadow(key, paper)
            self._shadow_book = saved.get('shadow_positions', {})
        if self._offline or self._settlements:
            raise RuntimeError('Offline settlement hooks must not enter deployment')
        if self._state_path.exists():
            state = json.loads(self._state_path.read_text())
            if state.get('schema') != 1:
                raise ValueError('Invalid lifecycle state; restore before starting')
            self._pool = state['combinations']
            self._retired_store = {tuple(p) for p in state['retired']}
            self._banned_coins = set(state['banned_coins'])
            # A rejected relationship belongs to a consumed publication. Restarting
            # must not reconsider it after the old relationship releases its coins.
            self._release_id = state['release_id']
        # Database custom records survive restarts, including an exit between JSON writes.
        for trade in Trade.get_trades_proxy():
            if trade.get_custom_data('atr450_detached', default=None):
                try:
                    _, params, _ = self._position_metadata(trade)
                except (KeyError, TypeError, ValueError):
                    continue  # Open records are reported individually in the loop.
                self._retired_store.add(canonical(params))
        self._reload_release()
        self.config['exchange']['pair_whitelist'] = sorted({p for row in self._pool
            if canonical(row) not in self._retired_store and not self._delisting_leg(canonical(row), None)
            for p in canonical(row)})
        self._save_state()

    def _reload_release(self):
        try:
            pointer, snapshot, _ = load_release(self.config['atr_release_root'])
            if pointer['release_id'] != self._release_id:
                occupied = {t.pair for t in Trade.get_trades_proxy(is_open=True)} | self._banned_coins
                self._pool = merge_combinations(self._pool, snapshot['selected'], self._retired_store, occupied)
                self._release_id = pointer['release_id']
                log.info('Loaded deployment release %s; %s admitted combinations', self._release_id[:12], len(self._pool))
            self._latest_asof = snapshot['asof']
            self._release_error = None
        except Exception as exc:
            self._release_error = f'{type(exc).__name__}: {exc}'
            log.error('Release update rejected; retaining loaded positions/parameters: %s', self._release_error)
            if self._release_id is None:
                raise

    def _save_state(self):
        if self._offline:
            self._state_error = None
            return
        state = dict(schema=1, combinations=self._pool, retired=sorted(self._retired_store),
            banned_coins=sorted(self._banned_coins), release_id=self._release_id,
            shadow_positions=deepcopy(self._shadow_book))
        if state != self._last_state:
            try:
                atomic(self._state_path, state)
            except Exception as exc:
                self._state_error=f'Shadow state persistence failed: {exc}'
                self.entry_pair_keys.clear()
                self._loop_ready=False
                raise
            self._last_state = state
        self._state_error = None

    def _load_live_delists(self):
        try:
            data = json.loads(self.live_delist_path.read_text())
            if not isinstance(data.get('delistings'), dict):
                raise ValueError('Missing delistings map')
            generated = pd.Timestamp(data['generated_utc'])
            if generated.tzinfo is None:
                raise ValueError('Delisting timestamp has no timezone')
            self._live_delists.update(data['delistings'])
            self._banned_coins.update(data['delistings'])
            age = (pd.Timestamp.now(tz='UTC') - generated).total_seconds()
            if age < -60 or age > 1800 or not data.get('sources_ok', False):
                raise ValueError('Delisting monitor stale or source failed')
            self._delist_error = None
        except Exception as exc:
            # Retain confirmed bans and continue exits. Never replace with an empty map.
            self._delist_error = f'{type(exc).__name__}: {exc}'

    def _delisting_leg(self, pairs, when, buffer_min=None):
        return next((p for p in pairs if p in self._banned_coins), None)

    def _row(self, pair, when, at_open=False):
        # Native dry/live frames contain completed candles. Require the most recent
        # completed candle even when the bot wakes at 12:05:17 rather than 12:05:00.
        frame = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe)
        if frame is None or frame.empty:
            return None
        expected = pd.Timestamp(when).floor('5min') - pd.Timedelta(minutes=5)
        rows = frame.loc[frame.date == expected]
        if rows.empty:
            return None
        row = rows.iloc[-1]
        values = {name: float(row[name]) for name in ('open', 'high', 'low', 'close', 'volume')}
        if not all(np.isfinite(v) for v in values.values()) or values['close'] <= 0:
            return None
        if at_open:
            # Live orders use the exchange's rate; completed close is a sizing estimate.
            return {'open': values['close']}
        return values

    def _quantity(self, pair, amount):
        return self._exchange.amount_to_contract_precision(pair, amount)

    def _position_metadata(self, trade):
        """Validate only the identity/coefficients needed to manage an existing leg."""
        key = self._pair_key_from_tag(trade.enter_tag)
        if not key:
            raise ValueError('missing_or_invalid_entry_tag')
        params = self._trade_params(trade)
        if not isinstance(params, dict) or not params:
            raise ValueError('missing_frozen_pair_parameters')
        if pair_key(params) != key or trade.pair not in canonical(params):
            raise ValueError('pair_metadata_mismatch')
        for name in ('gamma', 'c', 'z_mean', 'z_std'):
            value = params.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('invalid_frozen_coefficient_' + name)
        if params['gamma'] <= 0 or params['z_std'] <= 0:
            raise ValueError('nonpositive_gamma_or_z_std')
        opened = pd.Timestamp(trade.get_custom_data('atr450_opened', default=trade.open_date_utc.isoformat()))
        if pd.isna(opened) or opened.tzinfo is None or opened > pd.Timestamp(trade.open_date_utc):
            raise ValueError('invalid_pair_opened_timestamp')
        return key, params, opened

    def _retire_group(self, key, params, trades, current_time, reason):
        self._retired_store.add(canonical(params))
        for trade in trades:
            if not trade.get_custom_data('atr450_detached', default=None):
                trade.set_custom_data('atr450_detached', dict(at=current_time.isoformat(),
                    extreme=trade.open_rate, stop=None))
        self._pending_exit[key] = (current_time, reason)

    def _real_bot_loop_start(self, current_time, **kwargs):
        self._loop_ready = False  # Native callback errors must never leave old entry permission active.
        self.entry_pair_keys = set()
        self._signal_owner = {}
        self._loop_time = current_time
        self._plans = {}
        self._round_trade_ids = set()
        self._round_entry_intents = set()
        self._reload_release()
        self._load_live_delists()
        self._open = list(Trade.get_trades_proxy(is_open=True))
        self._groups = {}
        self._bad_trades = []
        metadata = {}
        for trade in self._open:
            try:
                key, params, opened = self._position_metadata(trade)
            except (KeyError, TypeError, ValueError) as exc:
                self._bad_trades.append(dict(id=trade.id, pair=trade.pair, reason=str(exc)))
                continue  # Suspend entries, but keep managing every valid relationship.
            metadata[trade.id] = (params, opened)
            self._groups.setdefault(key, []).append(trade)
        self._profit_exit.intersection_update(self._groups)
        self._pending_exit = {k:v for k,v in self._pending_exit.items() if k in self._groups}
        for key, trades in self._groups.items():
            params = metadata[trades[0].id][0]
            opened = min(metadata[t.id][1] for t in trades)
            banned = self._delisting_leg(canonical(params), current_time)
            timeout = pd.Timestamp(current_time) >= opened + pd.Timedelta(days=int(self.n_days.value))
            detached = any(t.get_custom_data('atr450_detached', default=None) for t in trades)
            if banned or timeout or detached:
                self._retire_group(key, params, trades, current_time,
                    'delisting_announced' if banned else 'timeout_immediate')
            elif len(trades) != 2:
                self._pending_exit[key] = (current_time, 'pair_incomplete_entry')
        self._retired_store.update(canonical(p) for p in self._pool if self._delisting_leg(canonical(p), current_time))
        self._pool = merge_combinations(self._pool, [], self._retired_store)
        self.tradable_pairs = {pair_key(p):p for p in self._pool if not self._delisting_leg(canonical(p), current_time)}
        for key, trades in self._groups.items():
            self.tradable_pairs[key] = self._trade_params(trades[0])
        occupied = {t.pair for t in self._open}
        allowed = set(self.dp.current_whitelist())
        candidates = {}
        for params in self._pool:
            key = pair_key(params)
            legs = canonical(params)
            if self._delisting_leg(legs, current_time) or canonical(params) in self._retired_store:
                continue
            if key in self._groups:
                if key not in self._pending_exit:
                    candidates[key] = params
                continue
            if occupied.intersection(legs) or not set(legs).issubset(allowed):
                continue
            if any(self._row(p, current_time) is None for p in legs):
                continue
            candidates[key] = params
        # Each slot owns a share of account capital even when it is currently empty.
        budget = .98 * self.wallets.get_total('USDT') / self._pair_capacity()
        self.stake_allocations = {key:budget for key in candidates}
        self.entry_pair_keys = set(candidates).difference(self._groups)
        # Signal ownership prevents two simultaneous combinations from sharing a coin.
        # Conflicting new relationships were already rejected at release admission.
        used = set(occupied)
        for key, params in candidates.items():
            legs = canonical(params)
            if key not in self.entry_pair_keys or used.intersection(legs):
                continue
            a, b = self._row(params['pair_A'], current_time), self._row(params['pair_B'], current_time)
            z = (a['close']-params['gamma']*b['close']-params['c']-params['z_mean'])/params['z_std']
            if abs(z) <= self.Zscore_entry:
                continue
            self._signal_owner[key] = z
            used.update(legs)
        self.entry_pair_keys.intersection_update(self._signal_owner)
        # StaticPairList reads this shared config on the next native refresh.
        # Held trades are independently added by Freqtrade's active-whitelist path.
        self.config['exchange']['pair_whitelist'] = sorted({p for row in self._pool
            if not self._delisting_leg(canonical(row), current_time) for p in canonical(row)})
        try:
            self._save_state()
        except Exception as exc:
            self._state_error = f'Lifecycle persistence failed: {exc}'
        if self._bad_trades:
            fault = f'{len(self._bad_trades)} open trades have invalid pair metadata; see bad_trades'
            self._state_error = '; '.join(filter(None, (self._state_error, fault)))
        if self._state_error or self._delist_error or self._release_error:
            self.entry_pair_keys.clear()
        self._last_time = current_time
        self._loop_ready = True
        start_marker = self._state_path.with_name(self._state_path.stem + '_start.json')
        if not start_marker.exists() and not (self._state_error or self._delist_error or self._release_error):
            atomic(start_marker, dict(started_utc=current_time.isoformat(), mode=self.dp.runmode.value))
        if time.monotonic() - self._health_at >= 30:
            atomic(self._state_path.with_name(self._state_path.stem + '.health.json'), dict(
                at=current_time.isoformat(), release_id=self._release_id, getpara_asof=self._latest_asof,
                combinations=len(self._pool), open_legs=len(self._open), signal_combinations=len(self.entry_pair_keys),
                retired=len(self._retired_store), banned_coins=sorted(self._banned_coins),
                state_error=self._state_error, delist_error=self._delist_error, release_error=self._release_error,
                bad_trades=self._bad_trades,
                plan_rejections=self._metrics.get('plan_rejections', {}),
                max_pairs=self._pair_capacity(), open_combinations=len(self._groups),
                pair_budget=budget, dynamic_capacity='fixed_slots'))
            self._health_at = time.monotonic()

    def populate_indicators(self, dataframe, metadata):
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = None
        if dataframe.empty or not self._loop_ready:
            return dataframe
        for key in self.entry_pair_keys:
            params = self.tradable_pairs[key]
            pair = metadata['pair']
            if pair not in canonical(params):
                continue
            if pd.Timestamp(dataframe.iloc[-1].date) != pd.Timestamp(self._loop_time).floor('5min')-pd.Timedelta(minutes=5):
                continue
            short = (self._signal_owner[key] > 0) == (pair == params['pair_A'])
            side = 'short' if short else 'long'
            dataframe.loc[dataframe.index[-1], 'enter_' + side] = 1
            dataframe.loc[dataframe.index[-1], 'enter_tag'] = 'entry_' + side + '_' + key
        return dataframe

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force, side, current_time=None, **kwargs):
        key = self._pair_key_from_tag(kwargs.get('entry_tag'))
        plan = self._plan(key, current_time)
        if not plan['valid'] or key not in self.entry_pair_keys or plan['sides'].get(pair) != side:
            return False
        for trade in Trade.get_trades_proxy(is_open=True):
            if trade.pair not in plan['stakes']:
                continue
            # Live market orders may be committed while their exchange status is
            # still open. The fill callback then arrives after the mate is checked.
            # Recognize our own approved submission in this loop without waiting
            # for that callback; never adopt an old/foreign position or duplicate a leg.
            same_relation = self._pair_key_from_tag(trade.enter_tag) == key
            pending_this_round = ((key, trade.pair) in self._round_entry_intents and
                pd.Timestamp(trade.open_date_utc) >= pd.Timestamp(self._loop_time))
            if trade.pair == pair or not same_relation or not (
                    trade.id in self._round_trade_ids or pending_this_round):
                return False
        # Market price may have moved since sizing: don't approve a larger allocation.
        stake = amount * rate / plan['leverage']
        approved = self._minimum(pair, rate, plan['leverage']) <= stake <= plan['stakes'][pair] * 1.01
        if approved:
            self._round_entry_intents.add((key, pair))
        return approved

    def _real_order_filled(self, pair, trade, order, current_time, **kwargs):
        self._base_order_filled(pair, trade, order, current_time, **kwargs)
        key = self._pair_key_from_tag(trade.enter_tag)
        if order.ft_order_side == trade.entry_side:
            self._round_trade_ids.add(trade.id)
            plan = self._plans.get(key)
            if plan and plan['valid']:
                plan['filled'].add(pair)
        elif order.ft_order_side == trade.exit_side and getattr(order, 'safe_amount', 0) >= trade.amount:
            self._pending_exit.setdefault(key, (current_time, 'pair_leg_closed'))

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        # A known exchange ban applies even when this leg's pair identity is damaged.
        if self._delisting_leg((trade.pair,), current_time):
            return 'delisting_announced'
        try:
            key, params, opened = self._position_metadata(trade)
        except (KeyError, TypeError, ValueError):
            if trade.get_custom_data('atr450_detached', default=None):
                return 'timeout_immediate'
            return None  # Unknown relationship: report it rather than guess its mate.
        banned = self._delisting_leg(canonical(params), current_time)
        # Exit management must survive a failed/partial bot_loop_start callback.
        if banned or pd.Timestamp(current_time) >= opened + pd.Timedelta(days=int(self.n_days.value)):
            reason = 'delisting_announced' if banned else 'timeout_immediate'
            self._retire_group(key, params, self._groups.get(key, [trade]), current_time, reason)
            return reason
        if key in self._pending_exit:
            return self._pending_exit[key][1]
        return self._base_custom_exit(pair, trade, current_time, current_rate, current_profit, **kwargs)

    @staticmethod
    def _validate_shadow(key, paper):
        params = paper['params']
        started = pd.Timestamp(paper['opened'])
        if key != pair_key(params) or started.tzinfo is None or pd.isna(started):
            raise ValueError('Invalid shadow relationship identity or clock')
        values = [paper['entry_A'], paper['entry_B'], params['gamma'], params['z_std']]
        if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>0 for v in values):
            raise ValueError('Invalid shadow prices or scientific parameters')
        if not all(isinstance(params[n],(int,float)) and not isinstance(params[n],bool) and math.isfinite(params[n])
                   for n in ('c','z_mean')):
            raise ValueError('Invalid shadow coefficients')
        if not isinstance(paper['short_A'],bool):
            raise ValueError('Invalid shadow direction')

    def _shadow_event(self, key, event, when, **values):
        self._metrics['shadow_events'].append(dict(key=key,event=event,at=when.isoformat(),**values))
        self._metrics['shadow_active'] = len(self._shadow_book)

    def _shadow_fill_price(self,pair,current_time):
        # A virtual market fill uses the currently observed ticker, not a future
        # candle close. The offline subclass supplies the native bar's open.
        quote=self.dp.ticker(pair)
        price=quote.get('last') if quote else None
        return float(price) if isinstance(price,(int,float)) and math.isfinite(price) and price>0 else None

    def _shadow_open(self, key, current_time):
        if key in self._shadow_book or self._shadow_closed_at.get(key) == current_time:
            return
        if key in self._groups or key not in self.entry_pair_keys:
            return
        params = self.tradable_pairs[key]
        if canonical(params) in self._retired_store or self._delisting_leg(canonical(params), current_time):
            return
        a,b = [self._shadow_fill_price(params[p],current_time) for p in ('pair_A','pair_B')]
        if a is None or b is None:
            return
        if not all(math.isfinite(r) and r>0 for r in (a,b)):
            return
        paper=dict(params=deepcopy(params),opened=current_time.isoformat(),
            entry_A=a,entry_B=b,short_A=self._signal_owner[key]>0)
        self._shadow_book[key] = paper
        self._metrics['shadow_opened'] += 1
        self._shadow_event(key,'open',current_time,paper=deepcopy(paper),reason='K_capacity')
        self._save_state()

    def _build_plan(self, key, current_time):
        plan = self._build_real_plan(key, current_time)
        if plan.get('why') == 'no_pair_capacity' and self._loop_ready:
            self._shadow_open(key, self._loop_time)
        return plan

    def _shadow_close(self, key, current_time, reason, retire=False):
        paper = self._shadow_book.pop(key)
        params = paper['params']
        prices = [self._shadow_fill_price(params[p],current_time) for p in ('pair_A','pair_B')]
        if retire:
            fresh = canonical(params) not in self._retired_store
            self._retired_store.add(canonical(params))
            if fresh and reason == 'timeout_immediate':
                self._metrics['shadow_timeout_retired'] += 1
            # Rare partial real fills still need the deployed exit manager.
            actual = [t for t in Trade.get_trades_proxy(is_open=True)
                if self._pair_key_from_tag(t.enter_tag) == key]
            if actual:
                self._retire_group(key,params,actual,current_time,reason)
            self._pool = merge_combinations(self._pool,[],self._retired_store)
        self._shadow_closed_at[key] = current_time
        self.entry_pair_keys.discard(key)
        self._signal_owner.pop(key,None)
        self._shadow_event(key,'retire' if retire else 'close',current_time,
            reason=reason,opened=paper['opened'],
            exit_A=prices[0],exit_B=prices[1])

    def _advance_shadows(self, current_time):
        admitted = {pair_key(r) for r in self._pool}
        for key,paper in list(self._shadow_book.items()):
            params = paper['params']
            opened = pd.Timestamp(paper['opened'])
            if opened > pd.Timestamp(current_time):
                raise ValueError('Shadow clock cannot be in the future')
            banned = self._delisting_leg(canonical(params),current_time)
            timeout = pd.Timestamp(current_time) >= opened + pd.Timedelta(days=int(self.n_days.value))
            if banned or timeout or canonical(params) in self._retired_store:
                self._shadow_close(key,current_time,'delisting_announced' if banned else 'timeout_immediate',True)
                continue
            if key not in admitted:
                raise ValueError('Non-retired shadow relationship lost admission')
            if key in self._groups:
                continue  # Keep the original clock during recovery of a partial real entry.
            a,b = [self._row(params[p],current_time) for p in ('pair_A','pair_B')]
            if a is None or b is None:
                continue
            z=(a['close']-params['gamma']*b['close']-params['c']-params['z_mean'])/params['z_std']
            sign=-1 if paper['short_A'] else 1
            pnl=sign*((a['close']-paper['entry_A'])-params['gamma']*(b['close']-paper['entry_B']))
            ratio=pnl/(paper['entry_A']+params['gamma']*paper['entry_B'])
            if abs(z)<self.Zscore_exit:
                self._shadow_close(key,current_time,'pair_z_reversion')
            elif ratio>self.Pair_profit_target and pd.Timestamp(current_time)>opened+pd.Timedelta(days=1):
                self._shadow_close(key,current_time,'pair_profit')
        self._metrics.update(shadow_active=len(self._shadow_book),
            retired_combinations=len(self._retired_store),admitted_combinations=len(self._pool))

    @staticmethod
    def _inherit_shadow_clock(trade, origin):
        opened = pd.Timestamp(origin)
        if pd.isna(opened) or opened.tzinfo is None or opened > pd.Timestamp(trade.open_date_utc):
            raise ValueError('Invalid inherited shadow clock')
        trade.set_custom_data('atr450_opened', origin)
        trade.set_custom_data('atr450_shadow_origin', origin)

    @staticmethod
    def _shadow_entry_filled(trade, origin):
        # A persisted Trade can exist before its order fills. A completed native
        # order, or the marker written by its fill callback, is durable evidence.
        orders = [o for o in getattr(trade, 'orders', ()) if o.ft_order_side == trade.entry_side]
        if any(o.ft_is_open for o in orders) or trade.amount <= 0:
            return False
        return (trade.get_custom_data('atr450_shadow_origin', default=None) == origin
                or any(not o.ft_is_open and (o.filled or 0) > 0 for o in orders))

    def _reconcile_shadow_promotions(self, current_time):
        if not self._shadow_book:
            return
        actual = list(Trade.get_trades_proxy(is_open=True))
        for key, paper in list(self._shadow_book.items()):
            legs = [t for t in actual if self._pair_key_from_tag(t.enter_tag) == key]
            params = paper['params']
            if len(legs) != 2 or {t.pair for t in legs} != set(canonical(params)):
                continue
            try:
                if any(self._position_metadata(t)[1] != params for t in legs):
                    continue  # Never adopt a different relationship's coefficients.
            except (KeyError, TypeError, ValueError):
                continue  # The base loop reports damaged metadata and manages valid legs.
            # Real entry follows the current signal; it need not keep the virtual
            # direction after a price gap. The real legs must still oppose each other.
            if len({t.is_short for t in legs}) != 2 or any(
                    not self._shadow_entry_filled(t, paper['opened']) for t in legs):
                continue
            for trade in legs:
                self._inherit_shadow_clock(trade, paper['opened'])
            if not self._offline:
                # Commit both real N clocks before removing their JSON recovery
                # record. A crash between these writes can safely retry promotion.
                Trade.commit()
            self._shadow_book.pop(key)
            self._metrics['shadow_promoted'] += 1
            self._shadow_event(key, 'promote', current_time, inherited_opened=paper['opened'])

    def bot_loop_start(self,current_time,**kwargs):
        try:
            # Reconcile before the base N-timeout checks, including legacy overlap
            # recovered from disk. Per-loop entry plans are deliberately irrelevant.
            self._reconcile_shadow_promotions(current_time)
            self._real_bot_loop_start(current_time,**kwargs)
            base_fault = self._state_error
            self._shadow_closed_at={k:v for k,v in self._shadow_closed_at.items() if v==current_time}
            self._advance_shadows(current_time)
            if not self._offline:
                self.config['exchange']['pair_whitelist']=sorted({coin for row in self._pool for coin in canonical(row)})
            self._save_state()
            if base_fault:
                self._state_error = base_fault
        except Exception as exc:
            self._state_error=f'Shadow lifecycle failed: {exc}'
            self.entry_pair_keys.clear()
            self._loop_ready=False
            raise

    def order_filled(self,pair,trade,order,current_time,**kwargs):
        origin=trade.get_custom_data('atr450_shadow_origin', default=None)
        self._real_order_filled(pair,trade,order,current_time,**kwargs)
        key=self._pair_key_from_tag(trade.enter_tag)
        paper=self._shadow_book.get(key)
        if order.ft_order_side!=trade.entry_side:
            return
        if not paper and not origin:
            return
        # Repeated native fill callbacks must not overwrite an inherited N clock
        # after the shadow record has already been removed.
        self._inherit_shadow_clock(trade, origin or paper['opened'])
        self._reconcile_shadow_promotions(current_time)
        self._save_state()
