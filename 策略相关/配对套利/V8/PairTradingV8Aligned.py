"""V8 offline adapter, including the unified virtual-position lifecycle.

No live files, wall clock, or server state enter a simulation. Parameter publications
and delisting detections are explicit historical events; candles remain native 5m.
"""
import builtins
import copy
import math
import datetime as dt
import hashlib
import inspect
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from freqtrade.persistence import Trade, LocalTrade
from freqtrade.exchange.exchange_utils import amount_to_contract_precision
from freqtrade.strategy.interface import StrategyResultValidator

from PairTradingV8Deploy import PairTradingV8Deploy, canonical, pair_key, merge_combinations, parameter_path

STUDY = Path(__file__).resolve().parent.parent.parent / 'research/binance_cross_v1'
if not STUDY.is_dir():
    STUDY = Path(__file__).resolve().parent
# Offline engine adapters are local to this module; deployment never installs them.

def _install_custom_data_index():
    from freqtrade.persistence.custom_data import CustomDataWrapper
    if getattr(CustomDataWrapper, "_atr450_index_installed", False):
        return
    original = CustomDataWrapper.get_custom_data
    state = {"records": None, "length": 0, "by_trade": {}}

    def indexed(*, trade_id, key=None):
        if CustomDataWrapper.use_db:
            return original(trade_id=trade_id, key=key)
        records = CustomDataWrapper.custom_data
        if records is not state["records"] or len(records) < state["length"]:
            state.update(records=records, length=0, by_trade={})
        for record in records[state["length"]:]:
            state["by_trade"].setdefault(record.ft_trade_id, []).append(record)
        state["length"] = len(records)
        found = state["by_trade"].get(trade_id, ())
        if key is not None:
            found = [record for record in found if record.cd_key.casefold() == key.casefold()]
        # Do not cache the deserialized value: mutations without set_custom_data
        # must have the same behavior as the original JSON-backed implementation.
        return [CustomDataWrapper._convert_custom_data(record) for record in found]

    CustomDataWrapper.get_custom_data = staticmethod(indexed)
    CustomDataWrapper._atr450_index_installed = True
    CustomDataWrapper._atr450_original_get = original


def _install_frozen_markets(path):
    from freqtrade.exchange import Exchange
    from freqtrade.util.datetime_helpers import dt_ts
    if getattr(Exchange,"_atr450_frozen_path",None)==str(path):
        return
    payload=json.loads(Path(path).read_text())
    if payload.get("exchange") != "binance" or payload.get("margin_mode") != "cross":
        raise RuntimeError("This study requires frozen Binance cross metadata")

    def check(self):
        mode=self._config.get("runmode")
        value=getattr(mode,"value",mode)
        if value not in ("backtest","hyperopt"):
            raise RuntimeError("Frozen market metadata is offline research only")
        if (self._config.get("exchange",{}).get("name"),self._config.get("margin_mode")) != ("binance","cross"):
            raise RuntimeError("Binance cross metadata cannot be applied to another exchange or margin mode")

    def tiers(self):
        check(self)
        self._leverage_tiers=copy.deepcopy(payload["leverage_tiers"])

    def markets(self,force=False,*,load_leverage_tiers=True):
        check(self)
        data=copy.deepcopy(payload["markets"])
        currencies=copy.deepcopy(payload["currencies"])
        self._markets=data
        for api in (self._api,self._api_async):
            api.precisionMode=payload["precision_mode"]
            api.set_markets(data,currencies)
        self._last_markets_refresh=dt_ts()
        if load_leverage_tiers: tiers(self)

    Exchange.reload_markets=markets
    Exchange.fill_leverage_tiers=tiers
    Exchange._atr450_frozen_path=str(path)


ENGINE_SHA256=None
SUPPORTED_GENERATOR_SHA256='abf3d1a360ab88f1f8d5a36a7d9ef3cfd6314eb8ea402b3a5de8ae9f11f303fc'


def _install_signal_bridge():
    global ENGINE_SHA256
    from freqtrade.optimize.backtesting import Backtesting, LONG_IDX, SHORT_IDX, ENTER_TAG_IDX, ELONG_IDX, ESHORT_IDX, EXIT_TAG_IDX
    if getattr(Backtesting,'_deployment_replay_installed',False):
        ENGINE_SHA256=Backtesting._deployment_replay_engine_sha
        return
    original=Backtesting.time_pair_generator
    source=inspect.getsource(original)
    # Pin the internal integration point we actually inspected for this project.
    if 'self.strategy.bot_loop_start' not in source or 'yield current_time_det, pair, row, is_last_row, trade_dir' not in source:
        raise RuntimeError('Freqtrade generator changed; review the stateful signal bridge')
    ENGINE_SHA256=hashlib.sha256(source.encode()).hexdigest()
    if ENGINE_SHA256 != SUPPORTED_GENERATOR_SHA256:
        raise RuntimeError('Pinned Freqtrade generator hash mismatch; review replay bridge before running')
    def generator(self,*args,**kwargs):
        aligned=getattr(self.strategy,'deployment_aligned',False)
        if aligned and self.config.get('timeframe_detail'):
            raise RuntimeError('Aligned replay does not support intrabar detail mode')
        for when,pair,row,last,direction in original(self,*args,**kwargs):
            if aligned:
                direction,tag=self.strategy.replay_signal(pair,when)
                row=list(row)
                row[LONG_IDX]=int(direction=='long')
                row[SHORT_IDX]=int(direction=='short')
                row[ENTER_TAG_IDX]=tag
                if hasattr(self.strategy,'replay_exit_signal'):
                    long_exit,short_exit,exit_tag=self.strategy.replay_exit_signal(pair,when)
                    row[ELONG_IDX]=long_exit;row[ESHORT_IDX]=short_exit;row[EXIT_TAG_IDX]=exit_tag
            yield when,pair,row,last,direction
    Backtesting.time_pair_generator=generator
    Backtesting._deployment_replay_installed=True
    Backtesting._deployment_replay_engine_sha=ENGINE_SHA256


class _MemoryState:
    """Make the deployment observability paths inert without patching any globals."""
    stem='offline'
    def with_name(self,name): return self
    def exists(self): return True


class PairTradingV8Aligned(PairTradingV8Deploy):
    deployment_aligned = True

    def __init__(self, config):
        super().__init__(config)
        self._publication_times = []
        previous = None
        compact = any(set(snapshot) == {'asof', 'selected'} for snapshot in self.schedule)
        self._compact_parameters = compact
        delay = pd.Timedelta(minutes=float(config.get('atr_publication_delay_minutes', 15 if compact else 0)))
        if delay < pd.Timedelta(0):
            raise ValueError('Publication delay cannot be negative')
        for snapshot in self.schedule:
            asof = pd.Timestamp(snapshot['asof'])
            published = pd.Timestamp(snapshot.get('published_at', asof+delay))
            if asof.tzinfo is None or published.tzinfo is None or published < asof:
                raise ValueError('Publication must be timezone-aware and not before the getpara cutoff')
            if previous is not None and published < previous:
                raise ValueError('Historical publications must be ordered')
            self._publication_times.append(published)
            previous = published
        notice_path = config.get('atr_historical_delistings')
        self._notices = []
        self._notice_metadata = {}
        if notice_path:
            path = Path(notice_path)
            data = json.loads(path.read_text())
            self._notice_metadata = dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                         coverage=data.get('coverage'))
            scan_delay = pd.Timedelta(minutes=float(config.get('atr_delist_scan_delay_minutes', 5)))
            if scan_delay < pd.Timedelta(0):
                raise ValueError('Scan delay cannot be negative')
            for pair, record in data['delistings'].items():
                published = pd.Timestamp(record['announced_utc'])
                if published.tzinfo is None:
                    raise ValueError('Announcement timestamp must have a timezone')
                # Explicit detection overrides are useful for recorded scanner outages.
                detected = pd.Timestamp(record.get('detected_at', published+scan_delay)).ceil('5min')
                if detected < published:
                    raise ValueError('Delisting detection cannot precede its announcement')
                self._notices.append((detected,pair))
            self._notices.sort()

    def _validate_mode(self, config):
        mode = getattr(config.get('runmode'), 'value', config.get('runmode'))
        if mode not in ('backtest', 'hyperopt'):
            raise ValueError('Aligned adapter is offline only; use PairTradingV8Deploy on the server')
        if (config.get('exchange', {}).get('name'), config.get('trading_mode'), config.get('margin_mode')) != ('binance', 'futures', 'cross'):
            raise ValueError('V8 requires Binance USDT perpetual futures in cross mode')
        if config.get('timeframe','5m') != '5m':
            raise ValueError('Aligned replay is specified for 5m candles')
        if config.get('timeframe_detail'):
            raise ValueError('Aligned replay currently supports native 5m candles, without detail timeframe')

    def _initialize_inputs(self, config):
        self._offline = True
        self._state_path = _MemoryState()
        _install_custom_data_index()
        snapshot = Path(config.get('atr_exchange_snapshot', str(STUDY / 'exchange_snapshot.json')))
        self._market_snapshot = snapshot
        if config.get('atr_exchange_snapshot') and not snapshot.exists():
            raise ValueError('Configured frozen exchange snapshot does not exist')
        if snapshot.exists():
            _install_frozen_markets(snapshot)
            self._precision_mode = json.loads(snapshot.read_text())['precision_mode']
        # --- B1/B2/B6: frozen settlement table (delisting_adapter.py) ---
        # A contract settlement is an exchange-side event: the leg is closed at the
        # settlement price and the pair can never be traded again. Freqtrade cannot
        # represent this in backtesting (see delist_hook.py), so it is an explicit
        # frozen input here rather than something inferred from the engine.
        self.settlement_price_field = config.get("atr_settlement_price_field", "price_mark_open")
        self.settlements_path = Path(config.get(
            "atr_settlements", str(STUDY / "progress/delisting_adapter/settlements.json")))
        self._settlements = {}
        if self._offline and self.settlements_path.exists():
            payload = json.loads(self.settlements_path.read_text())
            field = self.settlement_price_field
            for pair, record in payload.get("settlements", {}).items():
                price = record.get(field)
                self._settlements[pair] = dict(
                    bar_ms=int(record["bar_ms"]), settle_utc=record.get("settle_utc"),
                    price=None if price is None else float(price),
                    tick=float(record["price_tick"]) if record.get("price_tick") else None,
                    resolved=price is not None)
        self._raw = {}
        self._injected_frames = {}

    def _schedule_path(self, config):
        return Path(config['atr_schedule']) if config.get('atr_schedule') else parameter_path(config)

    def _reset(self):
        super()._reset()
        self._publication_cursor = 0
        self._notice_cursor = 0
        if hasattr(self,'_engine_signal_hash'):
            self._metrics['engine_signal_hook_sha256']=self._engine_signal_hash
        self._metrics.update(model_version='v8-deployment-aligned',publication_events=[],
                             partial_entry_failures=0,signal_bridge_calls=0)
        self._metrics['model_version']='v8-shadow-deployment-aligned'

    def _install_adapters(self):
        _install_signal_bridge()
        self._engine_signal_hash=ENGINE_SHA256
        snapshot = getattr(self, '_market_snapshot', STUDY/'exchange_snapshot.json')
        if snapshot.exists():
            _install_frozen_markets(snapshot)

    def __setstate__(self,state):
        # Joblib may unpickle a strategy into a fresh worker before generator dispatch.
        # Install there, not halfway through its first unwrapped backtest generator.
        self.__dict__.update(state)
        self._install_adapters()

    def bot_start(self, **kwargs):
        if self._compact_parameters and (not self.config.get('atr_historical_delistings') or not self.settlements_path.exists()):
            raise ValueError('Aligned replay requires frozen historical delisting announcements and a settlement table; do not use the live monitor file')
        self._install_adapters()
        self._engine_delisting = self._probe_engine_settlement()
        self._metrics.update(settlement_engine_checked=len(self._engine_delisting),
            settlement_engine_mismatches=self._settlement_engine_mismatches(),
            engine_signal_hook_sha256=self._engine_signal_hash)

    def _reload_release(self):
        when = pd.Timestamp(self._loop_time)
        while self._publication_cursor < len(self.schedule) and self._publication_times[self._publication_cursor] <= when:
            snapshot = self.schedule[self._publication_cursor]
            old = {canonical(p) for p in self._pool}
            occupied = {t.pair for t in Trade.get_trades_proxy(is_open=True)} | self._banned_coins
            self._pool = merge_combinations(self._pool,snapshot['selected'],self._retired_store,occupied)
            admitted = [pair_key(p) for p in self._pool if canonical(p) not in old]
            rejected = [pair_key(p) for p in snapshot['selected'] if canonical(p) not in {canonical(q) for q in self._pool}]
            self._metrics['publication_events'].append(dict(at=when.isoformat(),asof=snapshot['asof'],
                published_at=self._publication_times[self._publication_cursor].isoformat(),admitted=admitted,
                skipped=rejected,retained_combinations=len(self._pool)))
            self._release_id = str(self._publication_cursor)
            self._latest_asof = snapshot['asof']
            self._publication_cursor += 1
        self._release_error = None

    def _load_live_delists(self):
        # Called in the parent constructor only for live/test fixtures. No current file.
        if self._loop_time is None:
            return
        when = pd.Timestamp(self._loop_time)
        while self._notice_cursor < len(self._notices) and self._notices[self._notice_cursor][0] <= when:
            _,pair = self._notices[self._notice_cursor]
            self._banned_coins.add(pair)
            self._notice_cursor += 1
        # Unknown historical announcement timestamps fall back to settlement, never to
        # a guessed earlier date. The settlement-price proxy remains separately audited.
        for pair,record in self._settlements.items():
            if record['bar_ms'] <= when.value//1_000_000:
                self._banned_coins.add(pair)
        self._delist_error = None

    def _save_state(self):
        # State is simulated in memory, isolated per epoch; never overwrite live JSON.
        self._state_error = None

    def _probe_engine_settlement(self):
        """B3/B4: cross-check the frozen table against the engine's own view.

        Freqtrade's ``check_delisting_time`` is gated to live/dry-run and would be ``None``
        here; ``delist_hook`` lifts that gate using this same table, so ``hooked`` proves
        the pathway works but is not independent evidence. The independent signal is
        ``_check_delisting_futures``, which reads ``info.deliveryDate`` from the frozen
        exchange snapshot. The OKX-era ``active`` flag is deliberately never consulted:
        it is a capture-time flag and is False for all 135 settled contracts.
        """
        exchange = getattr(self.dp, "_exchange", None)
        probe = {}
        metadata = getattr(exchange, "_check_delisting_futures", None)
        for pair, record in self._settlements.items():
            entry = {"table_ms": record["bar_ms"], "hooked": None, "metadata": None}
            if exchange is not None:
                try:
                    value = self.dp.check_delisting(pair)
                    entry["hooked"] = None if value is None else int(value.timestamp() * 1000)
                except Exception:  # noqa: BLE001 - probe only
                    pass
                if callable(metadata):
                    try:
                        value = metadata(pair)
                        entry["metadata"] = None if value is None else int(value.timestamp() * 1000)
                    except Exception:  # noqa: BLE001 - probe only
                        pass
            probe[pair] = entry
        return probe

    def _settlement_engine_mismatches(self):
        probe = getattr(self, "_engine_delisting", None) or {}
        bad = []
        for pair, entry in probe.items():
            for key in ("hooked", "metadata"):
                value = entry.get(key)
                if value is not None and int(value) != int(entry["table_ms"]):
                    bad.append({"pair": pair, "source": key, "engine_ms": int(value),
                                "table_ms": int(entry["table_ms"])})
        return bad

    def _settled_leg(self, pairs, when):
        """Which of `pairs` has already settled at `when` (or None)."""
        stamp_ms = pd.Timestamp(when).value // 1_000_000
        for leg in pairs:
            record = self._settlements.get(leg)
            if record is not None and record["bar_ms"] <= stamp_ms:
                return leg
        return None

    def _settlement_exit(self, trade, current_time):
        """Return a settlement exit reason for this leg, if any.

        The settling leg exits at the settlement price (its injected final candle's
        open). Its mate is still trading, so it exits at the same bar's open -- the two
        legs must never be separated across a settlement.
        """
        if not self._settlements:
            return None
        params = self._trade_params(trade)
        legs = [trade.pair]
        if params:
            legs.extend((params["pair_A"], params["pair_B"]))
        stamp_ms = pd.Timestamp(current_time).value // 1_000_000
        for leg in dict.fromkeys(legs):
            record = self._settlements.get(leg)
            if record is not None and record["bar_ms"] <= stamp_ms:
                return "delist_settlement_self" if leg == trade.pair else "delist_settlement_mate"
        return None

    def _settlement_frame(self, pair, frame):
        """Replace the pair's final candle with its settlement candle.

        Without this the engine stops calling the strategy for a settled pair
        (``backtesting.py:1474`` returns None -> ``continue``), and the position is
        carried at a frozen price until ``handle_left_open`` force-exits it at the last
        available bar's open -- i.e. ~5 minutes before settlement, potentially months
        before the end of the test window.
        """
        record = self._settlements.get(pair)
        if record is None or frame is None or frame.empty:
            return frame
        bar = pd.Timestamp(record["bar_ms"], unit="ms", tz="UTC")
        needs = ("date", "open", "high", "low", "close", "volume")
        if any(column not in frame.columns for column in needs):
            raise RuntimeError("Settlement injection needs an OHLCV frame for " + pair)
        if bar <= frame["date"].iloc[0]:
            return frame
        if record["price"] is None:
            # Only fatal if the settlement could actually be reached inside the window.
            if bar <= frame["date"].iloc[-1] + pd.Timedelta(days=1):
                raise RuntimeError(
                    f"no settlement price for {pair} settling {record['settle_utc']} inside the loaded window; "
                    f"rebuild progress/delisting_adapter/settlements.json")
            return frame
        kept = frame.loc[frame["date"] < bar]
        if kept.empty:
            return frame
        price = float(record["price"])
        row = kept.iloc[[-1]].copy()
        row["date"] = bar
        row["close"] = price
        # Fire the settlement *at the open* of this candle: backtesting fills a
        # CUSTOM_EXIT at ``row[OPEN_IDX]`` (``backtesting.py:592``). The low/high band is
        # widened by one price tick because freqtrade rounds every order price to the
        # market tick before testing the fill (``_exit_trade`` -> ``price_to_precision``,
        # then ``low <= rate <= high``). With ``low == high == <raw archive price>`` the
        # order is created but can never fill, which leaves the leg open for the rest of
        # the run instead of closing it at settlement -- exactly the failure this adapter
        # exists to prevent. The band is one tick wide, so it cannot plausibly reach a real
        # stop or liquidation level.
        tick = float(record.get("tick") or price * 1e-9)
        row["open"] = price
        row["low"] = price - tick
        row["high"] = price + tick
        row["volume"] = 0.0
        injected = pd.concat([kept, row], ignore_index=True)
        self._injected_frames[pair] = record["settle_utc"]
        return injected

    @property
    def _bar_age_ns(self):
        """One timeframe in nanoseconds: the tolerance for '_row has no data here'."""
        limit = self.__dict__.get("_bar_age_ns_cache")
        if limit is None:
            limit = int(pd.Timedelta(self.timeframe or "5m").value)
            self.__dict__["_bar_age_ns_cache"] = limit
        return limit

    def _row(self, pair, when, at_open=False):
        raw = self._raw.get(pair)
        if raw is None:
            return None
        ts = int(pd.Timestamp(when).value)
        index = int(np.searchsorted(raw["time"], ts, side="left"))
        index -= 1
        if index < 0:
            return None
        # B5: `index` is the last bar strictly before `ts`. If that bar is more than one
        # timeframe behind, the pair has no data at `ts` -- it settled, or the series has
        # a gap. Returning the last available bar instead would freeze its price for the
        # remainder of the run and make every `is None` guard below unreachable (which is
        # exactly what used to happen for delisted contracts).
        if ts - int(raw["time"][index]) > self._bar_age_ns:
            return None
        row = {name: float(values[index]) for name, values in raw.items() if name != "time"}
        if at_open:
            # Deployment sizes with the last completed close, never a future open.
            return {"open": row["close"]}
        return row

    def _quantity(self,pair,amount):
        if not hasattr(self._exchange,"markets"):
            return self._exchange.amount_to_contract_precision(pair,amount)
        market=self._exchange.markets[pair]
        # Hyperopt workers deliberately have no live ccxt API client.
        mode = getattr(self, '_precision_mode', None)
        if mode is None:
            mode = self._exchange.precisionMode
        return amount_to_contract_precision(amount,market["precision"]["amount"],
            mode,market.get("contractSize") or 1)

    def populate_indicators(self,dataframe,metadata):
        names=('open','high','low','close','volume')
        self._raw[metadata['pair']]={'time':pd.DatetimeIndex(dataframe.date).as_unit('ns').asi8,
            **{k:dataframe[k].to_numpy(copy=True) for k in names}}
        return dataframe

    def advise_all_indicators(self, data):
        # The native per-epoch path otherwise leaves the previous analyzed
        # frames alive until reset_backtest, after new indicator copies exist.
        # Each optimizer call supplies an owned fresh pickle load; no original
        # files or shared input datasets are changed by this in-place pass.
        self._raw.clear()
        self.dp.clear_cache()
        gc.collect()
        # B6: give every settled pair its final settlement candle *before* indicators are
        # computed and before StrategyResultValidator snapshots the input frame, so the
        # engine keeps calling the strategy on the settlement bar (see _settlement_frame).
        # Idempotent: re-running on an already injected frame returns the same frame.
        for pair, frame in list(data.items()):
            updated = self._settlement_frame(pair, frame)
            if updated is not frame:
                data[pair] = updated
        result = {}
        for pair, frame in data.items():
            validator = StrategyResultValidator(frame, warn_only=not self.disable_dataframe_checks)
            result[pair] = self.advise_indicators(frame, {"pair": pair})
            validator.assert_df(result[pair])
        return result

    def populate_entry_trend(self,dataframe,metadata):
        # Engine preprocessing must contain no static future candidate/tag decisions.
        dataframe['enter_long']=pd.Series(0,index=dataframe.index,dtype='Int8')
        dataframe['enter_short']=pd.Series(0,index=dataframe.index,dtype='Int8')
        dataframe['enter_tag']=None
        return dataframe

    def _real_bot_loop_start(self,current_time,**kwargs):
        if self._last_time is None or current_time < self._last_time:
            self._reset()
        # Simulated side-effect boundary: reuse the deployment loop, but suppress ONLY
        # its JSON heartbeat/start-marker output and its mutable live whitelist.
        previously_detached={t.id for t in Trade.get_trades_proxy(is_open=True)
                             if t.get_custom_data('atr450_detached',default=None)}
        saved_whitelist=self.config['exchange']['pair_whitelist']
        self._state_path=_MemoryState()
        self._health_at=float('inf')
        try:
            super()._real_bot_loop_start(current_time,**kwargs)
        finally:
            self.config['exchange']['pair_whitelist']=saved_whitelist
        self._metrics['detached_legs']+=sum(t.id not in previously_detached and
            bool(t.get_custom_data('atr450_detached',default=None)) for t in self._open)
        self._metrics['retired_combinations']=len(self._retired_store)
        self._metrics['admitted_combinations']=len(self._pool)
        self._metrics['historical_notice_input']=self._notice_metadata
        self._metrics['partial_entry_failures'] += sum(len(ts)!=2 and not any(t.get_custom_data('atr450_detached',default=None) for t in ts)
                                                     for ts in self._groups.values())
        # Last completed bar only. Funding/fees match LocalTrade's own calculation.
        equity=self._metrics['initial']+float(LocalTrade.bt_total_profit)
        for trade in self._open:
            row=self._row(trade.pair,current_time)
            if row:
                equity += trade.calc_profit(row['close'])
        peak=max(self._metrics['peak'],equity)
        self._metrics.update(peak=peak,last_equity=equity,samples=self._metrics['samples']+1,
            max_drawdown=max(self._metrics['max_drawdown'],(peak-equity)/peak if peak>0 else 1))
        builtins._atr450_metrics=self._metrics

    def replay_signal(self,pair,current_time):
        self._metrics['signal_bridge_calls']+=1
        for key in self.entry_pair_keys:
            params=self.tradable_pairs[key]
            if pair not in canonical(params):
                continue
            short=(self._signal_owner[key]>0)==(pair==params['pair_A'])
            side='short' if short else 'long'
            return side,'entry_'+side+'_'+key
        return None,None

    def _real_order_filled(self,pair,trade,order,current_time,**kwargs):
        super()._real_order_filled(pair,trade,order,current_time,**kwargs)
        reason=getattr(order,'ft_order_tag',None) or trade.exit_reason or ''
        if order.ft_order_side==trade.exit_side and reason=='liquidation':
            # Intrabar liquidation is learned after the bar; don't close its mate at
            # this bar's open using that future information.
            key=self._pair_key_from_tag(trade.enter_tag)
            self._pending_exit[key]=(current_time+dt.timedelta(minutes=5),'pair_leg_closed')

    def custom_exit(self,pair,trade,current_time,current_rate,current_profit,**kwargs):
        settlement=self._settlement_exit(trade,current_time)
        if settlement:
            return settlement
        pending=self._pending_exit.get(self._pair_key_from_tag(trade.enter_tag))
        if pending and current_time<pending[0]:
            return self._base_custom_exit(pair,trade,current_time,current_rate,current_profit,**kwargs)
        return super().custom_exit(pair,trade,current_time,current_rate,current_profit,**kwargs)

    def _shadow_fill_price(self,pair,current_time):
        raw=self._raw.get(pair)
        if raw is None:
            return None
        stamp=int(pd.Timestamp(current_time).value)
        index=int(np.searchsorted(raw['time'],stamp,side='left'))
        if index>=len(raw['time']) or int(raw['time'][index])!=stamp:
            return None
        price=float(raw['open'][index])
        return price if math.isfinite(price) and price>0 else None
