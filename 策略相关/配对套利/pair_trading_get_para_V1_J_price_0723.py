# --- Import Freqtrade Libraries ---
import itertools

from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import IntParameter, DecimalParameter
import pandas as pd
import numpy as np
import talib as ta
from datetime import timedelta
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
import json
import os
import logging

logger = logging.getLogger(__name__)


class pair_trading_get_para_V1_J_price_0723(IStrategy):
    can_short = True
    timeframe = '5m'
    startup_candle_count = 0
    process_only_new_candles = False
    # 交易参数
    minimal_roi = {"0": 1}
    stoploss = -0.99999
    use_custom_stoploss = False  # 启用动态止损

    def __init__(self, config):
        super().__init__(config)
        # 交易对白名单
        self.whitelist = config.get('exchange', {}).get('pair_whitelist', [])

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        current_pair = metadata['pair']
        dataframe['zero'] = 0

        if current_pair != self.whitelist[-1]:
            return dataframe

        all_new_params = {}
        log_close_prices = {}

        all_dfs = {pair: self.dp.get_pair_dataframe(pair,self.timeframe) for pair in self.whitelist}

        for pair, df in all_dfs.items():
            if df is not None and not df.empty:
                log_close_prices[f'{pair}_log_close'] = df['close'].iloc[-1]

        for pair_y,pair_x in itertools.permutations(self.whitelist, 2):
            try:
                df_y = all_dfs[pair_y]
                df_x = all_dfs[pair_x]

                df_merged = pd.DataFrame(index=df_y.index)
                df_merged['y'] = df_y['close']
                df_merged['x'] = df_x['close']
                df_merged.dropna(inplace=True)

                if len(df_merged) < 1000:
                    continue

                c, gamma, z, z_mean, z_std, z_cross_zero_count, consistency_score, half_life, corr = self.fn_ecm(df_merged,'y','x')

                pvalue, adfstat = self.adf_test_on_residuals(z)

                # pair_key_sorted = '_'.join(sorted((pair_y,pair_x)))
                pair_key = f"{pair_y}_{pair_x}"

                all_new_params[pair_key] = {
                    'pvalue': pvalue,
                    'adfstat': adfstat,
                    'gamma': gamma,
                    'c': c,
                    'z_mean': z_mean,
                    'z_std': z_std,
                    'z_cross_zero_count': z_cross_zero_count,
                    'consistency_score': consistency_score,
                    'half_life': half_life,
                    'corr': corr,
                    'regression_y':pair_y,
                    'regression_x':pair_x,
                }
            except Exception as e:
                logger.error(f"分析配对 {pair_y}/{pair_x} 时出错: {e}")
        self.overwrite_run_config_params(all_new_params,log_close_prices)


        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        return dataframe



    def fn_ecm(self, df: pd.DataFrame, y_col: str, x_col: str):
        y = df[y_col]
        x = df[x_col]

        # 验证输入
        assert isinstance(y, pd.Series), 'y 必须是 pd.Series 类型'
        assert isinstance(x, pd.Series), 'x 必须是 pd.Series 类型'
        assert y.index.equals(x.index), 'y 和 x 必须有相同的时间索引'

        # 空值处理
        if y.isnull().any() or x.isnull().any():
            temp_df = pd.concat([y, x], axis=1).dropna()
            y = temp_df[y_col]
            x = temp_df[x_col]

        # 皮尔森相关系数
        corr = y.corr(x)

        # 协整回归
        long_run_ols = sm.OLS(y, sm.add_constant(x))
        long_run_fit = long_run_ols.fit()
        c, gamma = long_run_fit.params

        # 残差（只取有效部分，避免稀疏）
        z = long_run_fit.resid

        # 使用 z-score 标准化残差（只对有效部分）
        z_std = z.std()
        z_mean = z.mean()
        z_score = (z - z_mean) / z_std
        #z_std = z_score.std()
        #z_mean = z_score.mean()

        # --- 半衰期计算 ---
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

        # --- 穿越一致性分数计算 ---
        crossings = (np.sign(z).shift(1) * np.sign(z)) < 0
        crossing_indices = crossings[crossings].index

        consistency_score = 0.0
        if len(crossing_indices) > 2:
            intervals = pd.Series(crossing_indices).diff().dropna()
            interval_mean = intervals.mean()
            interval_std = intervals.std()
            if interval_mean > 0 and interval_std > 0:
                consistency_score = 1000 / (interval_mean * interval_std)

        # --- 只统计z分数穿越1和-1的次数 ---
        z_score_lag = z_score.shift(1)
        crossing_1 = ((z_score_lag < 1) & (z_score >= 1)).sum() + ((z_score_lag > 1) & (z_score <= 1)).sum()
        crossing_minus1 = ((z_score_lag > -1) & (z_score <= -1)).sum() + ((z_score_lag < -1) & (z_score >= -1)).sum()
        z_cross_zero_count = crossing_1 + crossing_minus1

        # 日志输出
        if half_life < 1440 and half_life > 0:
            logger.info(f"Half-life for {y_col} vs {x_col}: {half_life}")
            logger.info(f"corr {y_col} vs {x_col}: {corr}")

        return c, gamma, z_score, z_mean, z_std, z_cross_zero_count, consistency_score, half_life, corr


    def convert_numpy_types(self, obj):
        if isinstance(obj, dict):
            return {k: self.convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif pd.isna(obj): # 处理Pandas的NaN
            return None
        else:
            return obj

    def adf_test_on_residuals(self, z: pd.Series) -> tuple[float, float]:
        """
        对给定的残差序列执行增强迪基-福勒（ADF）检验。
        返回 pvalue 和 adfstat
        """
        z_cleaned = z.dropna()

        if len(z_cleaned) < 20:
            # 如果数据点太少，返回一个明确表示测试未成功运行的值
            return 1.0,0.0

        # adfuller返回多个值，我们捕获adfstat和pvalue
        adfstat, pvalue, usedlag, nobs, crit_values = adfuller(z_cleaned, maxlag=1, autolag=None)
        pvalue = pvalue
        adfstat = adfstat

        return pvalue, adfstat

    def overwrite_run_config_params(self, new_params: dict, log_close_prices:dict):
        """
        负责读取 run 策略的配置文件，用新计算出的参数，完整地覆盖旧的参数部分，然后写回。
        """
        run_config_path = 'user_data/config/future_pair_pairing_run_0723.json'

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
                y = params['regression_y']
                x = params['regression_x']

                pvalue_dict[f"{y}_{x}_pvalue"] = params['pvalue']
                adfstat_dict[f"{y}_{x}_adfstat"] = params['adfstat']
                gamma_dict[f"{y}_{x}_gamma"] = params['gamma']
                c_dict[f"{y}_{x}_c"] = params['c']
                z_mean_dict[f"{y}_{x}_z_mean"] = params['z_mean']
                z_std_dict[f"{y}_{x}_z_std"] = params['z_std']
                z_cross_zero_count_dict[f"{y}_{x}_z_cross_zero_count"] = params['z_cross_zero_count']
                consistency_score_dict[f"{y}_{x}_consistency_score"] = params['consistency_score']
                half_life_dict[f"{y}_{x}_half_life"] = params['half_life']
                corr_dict[f"{y}_{x}_corr"] = params['corr']

            run_config_data = {}
            if os.path.exists(run_config_path):
                with open(run_config_path, 'r', encoding='utf-8') as f:
                    run_config_data = json.load(f)
            else:
                logger.warning(f"目标配置文件 {run_config_path} 不存在，将创建一个新的。")

            run_config_data['pvalue_dict'] = pvalue_dict
            run_config_data['adfstat_dict'] = adfstat_dict
            run_config_data['gamma_dict'] = gamma_dict
            run_config_data['c_dict'] = c_dict
            run_config_data['z_mean_dict'] = z_mean_dict
            run_config_data['z_std_dict'] = z_std_dict
            run_config_data['z_cross_zero_count_dict'] = z_cross_zero_count_dict
            run_config_data['consistency_score_dict'] = consistency_score_dict
            run_config_data['half_life_dict'] = half_life_dict
            run_config_data['corr_dict'] = corr_dict
            run_config_data['log_close_dict'] = log_close_prices

            with open(run_config_path, 'w', encoding='utf-8') as f:
                json.dump(self.convert_numpy_types(run_config_data), f, ensure_ascii=False,
                          indent=4)

            logger.info(f"成功覆写 run 策略的配置文件！")

        except Exception as e:
            logger.error(f"覆写配置文件时出错: {e}")

















