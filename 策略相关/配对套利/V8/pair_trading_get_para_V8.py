import itertools
import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from freqtrade.strategy.interface import IStrategy
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


class pair_trading_get_para_V6(IStrategy):
    can_short = True
    timeframe = "5m"
    startup_candle_count = 10
    process_only_new_candles = False
    minimal_roi = {"0": 1}
    stoploss = -0.99
    use_custom_stoploss = False  # 不启用动态止损

    def __init__(self, config):
        super().__init__(config)
        self.whitelist = config.get("exchange", {}).get("pair_whitelist", [])
        self.all_counters = len(self.whitelist) * (len(self.whitelist) - 1)
        self.counter = 0

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        if self.dp.runmode.value != "backtest":
            raise ValueError("参数计算只允许在历史数据模式运行，禁止覆盖运行参数")
        if not self.whitelist:
            raise ValueError("参数计算白名单不能为空")
        current_pair = metadata["pair"]
        dataframe["zero"] = 0

        if current_pair != self.whitelist[-1]:
            return dataframe

        all_new_params = {}
        log_close_prices = {}

        all_dfs = {
            pair: self.dp.get_pair_dataframe(pair, self.timeframe) for pair in self.whitelist
        }

        for pair, df in all_dfs.items():
            if df is not None and not df.empty:
                log_close_prices[f"{pair}_log_close"] = df["close"].iloc[-1]

        for pair_y, pair_x in itertools.permutations(self.whitelist, 2):
            try:
                df_y = all_dfs[pair_y]
                df_x = all_dfs[pair_x]

                if df_y is None or df_x is None:
                    continue
                df_merged = self.align_prices(df_y, df_x)

                if len(df_merged) < 300:
                    continue

                (
                    c,
                    gamma,
                    z,
                    z_mean,
                    z_std,
                    z_cross_zero_count,
                    consistency_score,
                    half_life,
                    corr,
                ) = self.fn_ecm(df_merged, "y", "x")
                self.counter += 1
                logger.info(f"已分析 {self.counter}/{self.all_counters} ")

                pvalue, adfstat = self.adf_test_on_residuals(z)

                pair_key = f"{pair_y}_{pair_x}"

                all_new_params[pair_key] = {
                    "pvalue": pvalue,
                    "adfstat": adfstat,
                    "gamma": gamma,
                    "c": c,
                    "z_mean": z_mean,
                    "z_std": z_std,
                    "z_cross_zero_count": z_cross_zero_count,
                    "consistency_score": consistency_score,
                    "half_life": half_life,
                    "corr": corr,
                    "regression_y": pair_y,
                    "regression_x": pair_x,
                }
            except Exception as e:
                logger.error(f"分析配对 {pair_y}/{pair_x} 时出错: {e}")
        self.overwrite_run_config_params(all_new_params, log_close_prices)

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

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

    def overwrite_run_config_params(self, new_params: dict, log_close_prices: dict):
        """
        负责读取 run 策略的配置文件，用新计算出的参数，完整地覆盖旧的参数部分，然后写回。
        """
        if not new_params:
            raise ValueError("参数结果为空，保留原参数文件")
        run_config_path = (
            Path(self.config.get("user_data_dir", "user_data"))
            / self.config.get("pair_parameters_file", "config/future_run_paraV6.json")
        )

        logger.info(f"准备将 {len(new_params)} 个配对的新参数写入到: {run_config_path}")

        try:
            pvalue_dict = {}
            gamma_dict = {}
            c_dict = {}
            z_mean_dict = {}
            z_std_dict = {}
            z_cross_zero_count_dict = {}
            consistency_score_dict = {}
            half_life_dict = {}
            corr_dict = {}
            adfstat_dict = {}

            for key, params in new_params.items():
                y = params["regression_y"]
                x = params["regression_x"]

                pvalue_dict[f"{y}_{x}_pvalue"] = params["pvalue"]
                adfstat_dict[f"{y}_{x}_adfstat"] = params["adfstat"]
                gamma_dict[f"{y}_{x}_gamma"] = params["gamma"]
                c_dict[f"{y}_{x}_c"] = params["c"]
                z_mean_dict[f"{y}_{x}_z_mean"] = params["z_mean"]
                z_std_dict[f"{y}_{x}_z_std"] = params["z_std"]
                z_cross_zero_count_dict[f"{y}_{x}_z_cross_zero_count"] = params[
                    "z_cross_zero_count"
                ]
                consistency_score_dict[f"{y}_{x}_consistency_score"] = params["consistency_score"]
                half_life_dict[f"{y}_{x}_half_life"] = params["half_life"]
                corr_dict[f"{y}_{x}_corr"] = params["corr"]

            run_config_data = {}
            if os.path.exists(run_config_path):
                with open(run_config_path, "r", encoding="utf-8") as f:
                    run_config_data = json.load(f)
            else:
                logger.warning(f"目标配置文件 {run_config_path} 不存在，将创建一个新的。")

            run_config_data["pvalue_dict"] = pvalue_dict
            run_config_data["adfstat_dict"] = adfstat_dict
            run_config_data["gamma_dict"] = gamma_dict
            run_config_data["c_dict"] = c_dict
            run_config_data["z_mean_dict"] = z_mean_dict
            run_config_data["z_std_dict"] = z_std_dict
            run_config_data["z_cross_zero_count_dict"] = z_cross_zero_count_dict
            run_config_data["consistency_score_dict"] = consistency_score_dict
            run_config_data["half_life_dict"] = half_life_dict
            run_config_data["corr_dict"] = corr_dict
            run_config_data["log_close_dict"] = log_close_prices

            payload = json.dumps(
                self.convert_numpy_types(run_config_data),
                ensure_ascii=False,
                indent=4,
                allow_nan=False,
            )
            run_config_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=run_config_path.parent, delete=False
                ) as f:
                    temporary = f.name
                    f.write(payload)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temporary, run_config_path)
            finally:
                if temporary and os.path.exists(temporary):
                    os.unlink(temporary)

            logger.info(f"成功覆写 run 策略的配置文件！")

        except Exception as e:
            logger.exception("覆写配置文件失败，原文件保持不变")
            raise
