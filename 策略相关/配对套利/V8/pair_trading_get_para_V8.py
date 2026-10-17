"""Compute one causal V8 publication through Freqtrade's backtesting command."""
import itertools
import logging
from decimal import Decimal, ROUND_CEILING
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from freqtrade.configuration import TimeRange
from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType
from freqtrade.strategy.interface import IStrategy
from statsmodels.tsa.stattools import adfuller

from PairTradingV8Deploy import (
    atomic, canonical, default_config, parameter_path, publish, positive_count,
)

logger = logging.getLogger(__name__)


def order_feasibility(pair, price, markets, config):
    """Reject only impossible orders; pair allocation is still checked at actual entry.

    A fixed 27.5-USDT coin cutoff would incorrectly discard asymmetric pairs and
    pairs trading with fewer active allocations. Use the whole portfolio as an
    upper bound, and publish the real minimum required at the causal last price.
    """
    market = markets.get(pair)
    if market is None:
        return None, 'missing_exchange_order_filters'
    filters = {f['filterType']: f for f in market['filters']}
    lot = filters.get('LOT_SIZE', {})
    p = Decimal(str(price))
    step = Decimal(lot.get('stepSize', '0'))
    minimum = Decimal(lot.get('minQty', '0'))
    maximum = Decimal(lot.get('maxQty', '0'))
    notional = Decimal(filters.get('MIN_NOTIONAL', {}).get('notional', '0'))
    if step <= 0 or maximum <= 0 or notional <= 0:
        return None, 'invalid_exchange_order_filters'
    amount = (max(minimum, notional/p)/step).to_integral_value(rounding=ROUND_CEILING)*step
    required = amount*p
    upper = Decimal(str(config.get('dry_run_wallet', 1000))) * Decimal(str(config.get('tradable_balance_ratio', .99))) * 3
    if amount > maximum or required*Decimal('1.05') > upper:
        return None, 'minimum_order_exceeds_portfolio_capacity'
    return dict(last_completed_price=float(p), step_size=float(step), min_qty=float(minimum),
                min_notional=float(notional), rounded_min_qty=float(amount),
                rounded_min_notional=float(required), portfolio_upper_notional=float(upper),
                scope='Coin feasibility upper bound; actual pair budget/rounding checked at entry'), None



def select_combinations(records, max_pairs=40):
    """Same gates and stable ordering as the local build_snapshots pipeline."""
    positive_count(max_pairs, 'getpara_max_pairs')
    valid = [row for row in records if
             all(not isinstance(row.get(k), bool) and isinstance(row.get(k), (int, float))
                 and np.isfinite(row[k]) for k in ('gamma', 'c', 'z_mean', 'z_std', 'half_life', 'adfstat'))
             and row['adfstat'] < -6 and 0 < row['half_life'] < 500
             and row['gamma'] > 0 and row['z_std'] > 0]
    result, occupied = [], set()
    for row in sorted(valid, key=lambda r: (r['half_life'], r['adfstat'], canonical(r), r['pair_A'], r['pair_B'])):
        if occupied.intersection(canonical(row)):
            continue
        result.append(row)
        occupied.update(canonical(row))
        if len(result) == max_pairs:
            break
    return result


class pair_trading_get_para_V8(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True
    timeframe = '5m'
    startup_candle_count = 0
    process_only_new_candles = False
    minimal_roi = {'0': 1}
    stoploss = -.99

    def __init__(self, config):
        super().__init__(config)
        self._parameters_written = False

    def _cutoff(self):
        timerange = TimeRange.parse_timerange(self.config.get('timerange'))
        if timerange.stopts == 0:
            raise ValueError('getpara requires an explicit --timerange end (the data cutoff)')
        end = pd.Timestamp(timerange.stopts, unit='s', tz='UTC')
        if end != end.floor('5min') or end > pd.Timestamp.now(tz='UTC').floor('5min'):
            raise ValueError('getpara cutoff must be a completed UTC 5-minute boundary')
        start = end-pd.Timedelta(days=90)
        if timerange.startts and pd.Timestamp(timerange.startts, unit='s', tz='UTC') != start:
            raise ValueError('getpara --timerange must cover exactly the preceding 90 days')
        return start, end

    def _read_frames(self, pairs, start, end):
        timerange = TimeRange('date', 'date', int(start.timestamp()), int(end.timestamp()))
        return {pair: load_pair_history(pair=pair, timeframe='5m',
                    datadir=Path(self.config['datadir']), timerange=timerange,
                    fill_up_missing=False, drop_incomplete=False, startup_candles=0,
                    data_format=self.config.get('dataformat_ohlcv', 'feather'),
                    candle_type=CandleType.FUTURES) for pair in pairs}

    def calculate(self, frames, cutoff, markets=None):
        """Pure calculation used by both the native entry and synthetic acceptance."""
        end = pd.Timestamp(cutoff)
        if end.tzinfo is None or end > pd.Timestamp.now(tz='UTC').floor('5min') or end != end.floor('5min'):
            raise ValueError('Invalid getpara data cutoff')
        grid = pd.date_range(end-pd.Timedelta(days=90), end, freq='5min', inclusive='left').as_unit('ns')
        valid, excluded = {}, {}
        for pair, frame in sorted(frames.items()):
            if not pair.endswith('/USDT:USDT'):
                excluded[pair] = 'not_usdt_perpetual'
                continue
            if frame is None or not {'date', 'close'}.issubset(frame.columns):
                excluded[pair] = 'missing_history'
                continue
            frame = frame.loc[(frame.date >= grid[0]) & (frame.date < end), ['date', 'close']].sort_values('date')
            prices = frame.close.to_numpy()
            if len(frame) != len(grid) or not pd.DatetimeIndex(frame.date).as_unit('ns').equals(grid):
                excluded[pair] = 'incomplete_90_day_history'
            elif not np.isfinite(prices).all() or (prices <= 0).any() or np.std(prices) == 0:
                excluded[pair] = 'invalid_price'
            else:
                if markets is not None:
                    _, reason = order_feasibility(pair, prices[-1], markets, self.config)
                    if reason:
                        excluded[pair] = reason
                        continue
                valid[pair] = frame.close.reset_index(drop=True)
        if len(valid) < 2:
            raise ValueError('Fewer than two coins have complete valid 90-day history; old parameters retained: ' + str(excluded))
        records = []
        for a, b in itertools.permutations(valid, 2):
            try:
                c, gamma, residual, zm, zs, crossings, consistency, half, corr = self.fn_ecm(
                    pd.DataFrame(dict(y=valid[a], x=valid[b])), 'y', 'x')
                pvalue, adf = self.adf_test_on_residuals(residual)
            except (ValueError, np.linalg.LinAlgError) as exc:
                logger.warning('Invalid fit %s / %s: %s', a, b, exc)
                continue
            records.append(self.convert_numpy_types(dict(pair_A=a, pair_B=b, gamma=gamma, c=c,
                           z_mean=zm, z_std=zs, half_life=half, adfstat=adf)))
        selected = select_combinations(records, self.config.get('getpara_max_pairs', 40))
        if not selected:
            raise ValueError('No qualified combinations; old parameters retained')
        logger.info('getpara cutoff=%s eligible=%s selected=%s exclusions=%s', end.isoformat(), len(valid), len(selected), excluded)
        return dict(asof=end.isoformat(), selected=selected)

    def populate_indicators(self, dataframe, metadata):
        mode = getattr(self.dp.runmode, 'value', self.dp.runmode)
        if mode != 'backtest':
            raise ValueError('getpara runs only through historical backtesting, never live/dry_run')
        if (self.config.get('exchange', {}).get('name'), self.config.get('trading_mode'), self.config.get('margin_mode')) != ('binance', 'futures', 'cross'):
            raise ValueError('getpara requires Binance futures cross inputs')
        if self.config.get('timeframe', '5m') != '5m':
            raise ValueError('getpara requires 5m candles')
        if self._parameters_written:
            return dataframe
        start, end = self._cutoff()
        frames = self._read_frames(self.dp.current_whitelist(), start, end)
        markets = {pair: (self.dp.market(pair) or {}).get('info') for pair in frames}
        snapshot = self.calculate(frames, end, markets)
        publish(parameter_path(self.config), snapshot)
        config_path = Path(self.config.get('getpara_run_config',
                           str(Path(self.config.get('user_data_dir', 'user_data'))/'config_v8.json')))
        if not config_path.exists():
            config = default_config({p for row in snapshot['selected'] for p in canonical(row)})
            for name in ('dry_run_wallet', 'tradable_balance_ratio', 'atr_fixed_parameters', 'dataformat_ohlcv'):
                if name in self.config:
                    config[name] = self.config[name]
            if self.config.get('atr_parameters'):
                config['atr_parameters'] = str(parameter_path(self.config))
            atomic(config_path, config)
        self._parameters_written = True
        logger.info('V8 parameters written to %s; paper configuration: %s (existing configuration preserved)',
                    parameter_path(self.config), config_path)
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        dataframe['enter_long'] = dataframe['enter_short'] = 0
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        dataframe['exit_long'] = dataframe['exit_short'] = 0
        return dataframe


    @staticmethod
    def align_prices(df_y, df_x):
        """同一时刻的价格才能参与回归；不填充缺失的行情。"""
        for frame in (df_y, df_x):
            if not {"date", "close"}.issubset(frame.columns):
                raise ValueError("行情缺少 date/close")
            if frame["date"].isna().any() or frame["date"].duplicated().any():
                raise ValueError("行情存在缺失或重复时间戳")
        merged = (
            df_y[["date", "close"]]
            .merge(
                df_x[["date", "close"]],
                on="date",
                how="inner",
                suffixes=("_y", "_x"),
                validate="one_to_one",
            )
            .sort_values("date")
            .rename(columns={"close_y": "y", "close_x": "x"})
        )
        valid = np.isfinite(merged[["y", "x"]]).all(axis=1)
        valid &= (merged[["y", "x"]] > 0).all(axis=1)
        return merged.loc[valid, ["y", "x"]].reset_index(drop=True)


    def fn_ecm(self, df: pd.DataFrame, y_col: str, x_col: str):
        y = df[y_col]
        x = df[x_col]

        if (
            not isinstance(y, pd.Series)
            or not isinstance(x, pd.Series)
            or not y.index.equals(x.index)
        ):
            raise ValueError("回归输入必须为具有相同时间索引的 Series")

        if y.isnull().any() or x.isnull().any():
            temp_df = pd.concat([y, x], axis=1).dropna()
            y = temp_df[y_col]
            x = temp_df[x_col]
        corr = y.corr(x)

        long_run_ols = sm.OLS(y, sm.add_constant(x))
        long_run_fit = long_run_ols.fit()
        c, gamma = long_run_fit.params

        # 残差恢复到原始索引，保留缺失行情的位置。
        z = pd.Series(index=df.index, dtype=float)
        z.update(long_run_fit.resid)

        z_mean = z.mean()
        z_std = z.std()

        # 由残差变化对滞后残差的回归系数估计半衰期。
        z_nonan = z.dropna()
        if len(z_nonan) > 1:
            z_lag = z_nonan.shift(1).dropna()
            z_ret = z_nonan.diff().dropna()
            z_lag = z_lag.loc[z_ret.index]
            X = sm.add_constant(z_lag)
            model = sm.OLS(z_ret, X)
            res = model.fit()
            if len(res.params) > 1 and res.params.iloc[1] != 0:
                half_life = -np.log(2) / res.params.iloc[1]
            else:
                half_life = np.nan
        else:
            half_life = np.nan

        # 根据穿越零点的间隔计算一致性分数。
        z_sign = np.sign(z)
        z_cross_zero_count = ((z_sign.shift(1) * z_sign) < 0).sum()
        crossings = (z_sign.shift(1) * z_sign) < 0

        crossing_indices = crossings[crossings].index

        consistency_score = 0.0

        # 至少三个穿越点才能计算多个间隔及其标准差。
        if len(crossing_indices) > 2:
            # 间隔以 K 线数量为单位。
            intervals = pd.Series(crossing_indices).diff().dropna()

            interval_mean = intervals.mean()
            interval_std = intervals.std()

            if interval_mean > 0 and interval_std > 0:
                # 以间隔均值与标准差的乘积倒数计分，乘以 1000 便于比较。
                consistency_score = 1000 / (interval_mean * interval_std)

        return c, gamma, z, z_mean, z_std, z_cross_zero_count, consistency_score, half_life, corr


    def convert_numpy_types(self, obj):
        if isinstance(obj, dict):
            return {k: self.convert_numpy_types(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.convert_numpy_types(v) for v in obj]
        if isinstance(obj, np.generic):
            obj = obj.item()
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj


    def adf_test_on_residuals(self, z: pd.Series) -> tuple[float, float]:
        """
        对给定的残差序列执行增强迪基-福勒（ADF）检验。
        返回 pvalue 和 adfstat
        """
        z_cleaned = z.dropna()

        if len(z_cleaned) < 20:
            return 1.0, 0.0

        adfstat, pvalue, usedlag, nobs, crit_values = adfuller(z_cleaned, maxlag=1, autolag=None)

        return pvalue, adfstat
