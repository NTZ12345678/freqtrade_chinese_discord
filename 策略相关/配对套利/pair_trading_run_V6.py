# --- Import Freqtrade Libraries ---
from typing import Tuple, Optional, Union
import numpy as np
from freqtrade.strategy.interface import IStrategy
from datetime import datetime, timedelta, timezone
from freqtrade.persistence import Trade
from datetime import datetime, timedelta
from freqtrade.exchange import timeframe_to_prev_date
from datetime import timedelta, timezone
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class pair_trading_run_V6(IStrategy):
    can_short = True
    startup_candle_count = 3
    process_only_new_candles = True
    # 交易参数
    stoploss = -100
    trailing_stop = False
    use_custom_stoploss = False  # 启用动态止损
    Zscore_entry = 1  # Z-score入场阈值
    Zscore_entry1 = 1.1 * Zscore_entry  # Z-score入场阈值1.1倍
    Zscore_exit = 0.8  # Z-score出场阈值
    Zscore_stop = 3  # Z-score止损阈值 (亏损平仓)
    can_trade_usdt = False
    min_consistency_score = 0.002
    zscore_mean_window = 10  # Z-score均值计算窗口

    order_types = {
        'entry': 'market',
        'exit': 'market',
        'emergency_exit': 'market',
        'force_entry': 'market',
        'force_exit': "market",
        'stoploss': 'market',
        'stoploss_on_exchange': False,
        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_market_ratio': 0.99
    }

    def __init__(self, config):
        super().__init__(config)
        self.free_usdt = 0
        # self.free_usdt = 0.5 * self.free_usdt  # 假设有60%的USDT可用
        self.dry_run_wallet = config.get('dry_run_wallet', 5000) if hasattr(config, 'get') else 5000
        self.whitelist = config.get('exchange', {}).get('pair_whitelist', [])
        self.pvalue_dict = config.get('pvalue_dict', {})
        self.gamma_dict = config.get('gamma_dict', {})
        self.c_dict = config.get('c_dict', {})
        self.z_mean_dict = config.get('z_mean_dict', {})
        self.z_std_dict = config.get('z_std_dict', {})
        self.adfstat_dict = config.get('adfstat_dict', {})
        self.consistency_score_dict = config.get('consistency_score_dict', {})
        self.half_life_dict = config.get('half_life_dict', {})
        #配对字典相关
        self.all_candidate_pairs = {}
        self.tradable_pairs = {}
        self.stake_allocations = {}
        self.pair_states = {}
        ### 修改：加一个定时器，更新配对池
        self.last_update_time = datetime(1945, 8, 15, tzinfo=timezone.utc)
        self.update_interval = timedelta(days=1)
        self.candidates_initialized = False

        # 控制日志打印频率
        self._pair_profit_log_counter = 0


        # 白名单过滤
        self.use_whitelist =  True

    def _is_pair_active(self, pair: str) -> bool:
        try:
            market = self.dp.market(pair)
        except Exception:
            return False
        if not market:
            return False
        return bool(market.get('active', True))

    def _remove_invalid_tradable_pairs(self) -> None:
        if not self.tradable_pairs:
            return

        invalid_keys = []
        for pair_key, params in self.tradable_pairs.items():
            pair_A = params.get('pair_A')
            pair_B = params.get('pair_B')
            if not pair_A or not pair_B:
                invalid_keys.append(pair_key)
                continue
            if not self._is_pair_active(pair_A) or not self._is_pair_active(pair_B):
                invalid_keys.append(pair_key)

        if not invalid_keys:
            return

        for pair_key in invalid_keys:
            self.tradable_pairs.pop(pair_key, None)
            self.stake_allocations.pop(pair_key, None)
            self.pair_states.pop(pair_key, None)

        logger.warning(f"检测到下架/失效交易对，已剔除配对: {invalid_keys}")

    def _initialize_all_candidates(self) -> list:


        whitelist = self.whitelist

        # ============================
        # 1️⃣ 可选白名单过滤
        # ============================
        if self.use_whitelist:
            whitelist_pairs = set()

            for i in range(0, len(whitelist), 2):
                if i + 1 >= len(whitelist):
                    break

                a = whitelist[i]
                b = whitelist[i + 1]

                whitelist_pairs.add(f"{a}_{b}")

            #logger.info(f"启用白名单过滤，配对: {whitelist_pairs}")

            pvalue_filtered = {
                k: v for k, v in self.pvalue_dict.items()
                if k.replace('_pvalue', '') in whitelist_pairs
            }

        else:
            logger.info("未启用白名单过滤，使用全部配对")
            pvalue_filtered = {k: v for k, v in self.pvalue_dict.items() if v <= 0.01}

        logger.info(f"白名单步骤后配对数量: {len(pvalue_filtered )}")



        logger.info(f"预加载配对数量: {len(pvalue_filtered)}")
        candidates = []
        for p_key, pvalue in pvalue_filtered.items():
            try:
                a_key, b_key = p_key.replace('_pvalue', '').split('_')
                score_key = f"{a_key}_{b_key}_consistency_score"
                consistency_score = self.consistency_score_dict.get(score_key)
                # a_price = self.log_close_dict.get(f"{a_key}_log_close")

                adf_key = f"{a_key}_{b_key}_adfstat"
                adfstat = self.adfstat_dict.get(adf_key)

                # ADF 统计量需要小于临界值 (更负) 才能认为序列平稳。
                # 暂定 小于 -6.0 
                if adfstat is None or adfstat > -6.0:
                    continue
                logger.info(f"预加载配对 {a_key}_{b_key} 通过ADF检验，ADF统计量: {adfstat:.4f}")
                # if consistency_score is None or consistency_score < self.min_consistency_score:
                #     continue
                if not self._is_pair_active(a_key) or not self._is_pair_active(b_key):
                    continue
                logger.info(f"预加载配对 {a_key}_{b_key} 通过活跃性检查")
                logger.info(f"预加载配对 {a_key}_{b_key} 开始获取数据...")

                a_df = self.dp.get_pair_dataframe(pair=a_key, timeframe=self.timeframe)
                logger.info(f"预加载配对 {a_key}_{b_key} 获取数据完成，数据点数量: {len(a_df)}")
                if a_df.empty:
                    logger.warning(f"预加载配对 {a_key}_{b_key} 时，无法获取 {a_key} 的数据，跳过该配对。")
                    continue
                logger.info(f"预加载配对 {a_key}_{b_key} 成功获取数据，数据点数量: {len(a_df)}")
                if self.dp.runmode.value in ('live', 'dry_run'):
                    a_price = a_df['close'].iloc[-1]
                else:
                    a_price = a_df['close'].iloc[0]
                logger.info(f"预加载配对 {a_key}_{b_key} 获取价格: {a_key}={a_price:.4f}")
                b_df = self.dp.get_pair_dataframe(pair=b_key, timeframe=self.timeframe)
                if b_df.empty:
                    #logger.warning(f"预加载配对 {a_key}_{b_key} 时，无法获取 {b_key} 的数据，跳过该配对。")
                    continue
                if self.dp.runmode.value in ('live', 'dry_run'):
                    b_price = b_df['close'].iloc[-1]
                else:
                    b_price = b_df['close'].iloc[0]
                logger.info(f"预加载配对 {a_key}_{b_key} 获取价格: {b_key}={b_price:.4f}")
                gamma = self.gamma_dict.get(f"{a_key}_{b_key}_gamma")
                if a_price is None or b_price is None or gamma is None:
                    logger.warning(f"预加载配对 {a_key}_{b_key} 时，缺少价格或gamma数据，跳过该配对。")
                    continue
                logger.info(f"预加载配对: {a_key} - {b_key}, p-value: {pvalue}, gamma: {gamma}")

                # m_val, n_val = (a_price + gamma * b_price), 0
                # if m_val > 0: n_val = self.free_usdt / m_val
                # a_amount, b_amount = n_val * a_price, n_val * gamma * b_price
                # if a_amount < 5 / 20 or b_amount < 5 / 20 : continue

                # 1. 定义金融常量
                MIN_STAKE_USDT = 5.0  # 交易所要求的最小订单名义价值

                # 2. 使用资金中性法 (Dollar Neutrality) 计算目标仓位价值
                #    公式: Stake_B = Budget / (gamma + 1), Stake_A = Budget - Stake_B
                if (gamma + 1) <= 0:
                    continue  # 避免除以零或负数
                #logger.info(f"预加载配对 {a_key}_{b_key} 计算目标仓位: free_usdt={self.free_usdt:.2f}, gamma={gamma:.4f}")
                target_stake_B = self.free_usdt / (gamma + 1)
                target_stake_A = self.free_usdt - target_stake_B

                # 3. 检查仓位价值是否满足最小要求 (与杠杆无关！)
                if target_stake_A < MIN_STAKE_USDT or target_stake_B < MIN_STAKE_USDT:
                    continue
                #logger.info(f"预加载配对 {a_key}_{b_key} 计算目标仓位: Stake_A=${target_stake_A:.2f}, Stake_B=${target_stake_B:.2f}")
                half_key = f"{a_key}_{b_key}_half_life"
                half_life = self.half_life_dict.get(half_key)
                #logger.info(f"预加载配对 {a_key}_{b_key} 计算半衰期: {half_life} 天")
                if half_life is None or half_life > 500:
                    continue
                logger.info(f"预加载配对 {a_key}_{b_key} 通过所有筛选条件，半衰期: {half_life:.2f} 天")
                ### 修改
                candidates.append({
                    'raw_pair_key': f"{a_key}_{b_key}",
                    'pair_A': a_key,
                    'pair_B': b_key,
                    # 'consistency_score': consistency_score,
                    'half_life': half_life,
                    'adfstat': adfstat,
                })
            except Exception as e:
                logger.error(f"预加载数据时出错: {e} for key {p_key}")

        ### 修改
        logger.info(f"预加载所有候选配对完成，共计 {len(candidates)} 个配对。")
        logger.info(f"所有配对示例: {candidates}")
        return sorted(candidates, key=lambda x: x['half_life'], reverse=False)
        # return candidates

    def _update_tradable_pairs(self):
        # logger.warning(f"触发动态配对池更新...")


        new_tradable_pairs = {}
        locked_currencies = set()



        # 2. 贪婪选择，填充剩余名额
        max_allowed_pairs = (self.config.get('max_open_trades', 1)) / 2
        # logger.info(f"当前最大可交易配对数量: {max_allowed_pairs}")
        # logger.info(f"初始化筛选的配对数量: {len(new_tradable_pairs)}")

        for candidate in self.all_candidate_pairs:
            #logger.info(f"检查候选配对: {candidate['raw_pair_key']} ")

            if len(new_tradable_pairs) >= max_allowed_pairs:
                break

            pair_A_base = candidate['pair_A'].split('/')[0]
            pair_B_base = candidate['pair_B'].split('/')[0]
            if not self._is_pair_active(candidate['pair_A']) or not self._is_pair_active(candidate['pair_B']):
                continue

            if pair_A_base in locked_currencies or pair_B_base in locked_currencies:
                continue

            pair_key_sorted = '_'.join((candidate['pair_A'], candidate['pair_B']))
            if pair_key_sorted not in new_tradable_pairs:
                new_tradable_pairs[pair_key_sorted] = {
                    'pair_A': candidate['pair_A'], 'pair_B': candidate['pair_B'],
                    'gamma': self.gamma_dict.get(f"{candidate['raw_pair_key']}_gamma"),
                    'half_life': candidate.get('half_life'),'adfstat': candidate.get('adfstat'),
                    'c': self.c_dict.get(f"{candidate['raw_pair_key']}_c"),
                    'z_mean': self.z_mean_dict.get(f"{candidate['raw_pair_key']}_z_mean"),
                    'z_std': self.z_std_dict.get(f"{candidate['raw_pair_key']}_z_std"),
                }
                if self.use_whitelist ==  True:
                    locked_currencies.add(pair_A_base)
                    locked_currencies.add(pair_B_base)
                #logger.info(f"动态选入新配对 {pair_key_sorted} (一致性分数: {candidate['consistency_score']:.2f})")
        #logger.info(f"动态配对池更新完成。选入配对数量: {len(new_tradable_pairs)}")

        # 3. 原子化地更新 tradable_pairs 和资金分配
        self.tradable_pairs = new_tradable_pairs
        if len(self.tradable_pairs) > 0:
            capital_per_pair = self.free_usdt / len(self.tradable_pairs)
            #capital_per_pair = self.free_usdt / 6 
            self.stake_allocations = {key: capital_per_pair for key in self.tradable_pairs}
        else:
            self.stake_allocations = {}
        
        #logger.info(f"金额分配: {self.stake_allocations}")
        #logger.info(f"动态配对池更新完成。当前可交易配对数量: {len(self.tradable_pairs)}")
        # for pair_key, pair_info in self.tradable_pairs.items():
        #     logger.info(f"  - {pair_key}: {pair_info}")
        if self.use_whitelist ==  False:
            logger.info(f"动态配对池更新完成。当前可交易配对:一共{len(self.tradable_pairs)}对 {list(self.tradable_pairs.keys())} ")


        # # ============================
        # # 4️⃣ 白名单过滤（强制配对）
        # # ============================
        # whitelist = self.config.get("pair_whitelist", [])

        # filtered_pairs = {}

        # # 每两个组成一对 (A-B, C-D)
        # for i in range(0, len(whitelist), 2):
        #     if i + 1 >= len(whitelist):
        #         break

        #     pair_A = whitelist[i]
        #     pair_B = whitelist[i + 1]

        #     pair_key = f"{pair_A}_{pair_B}"

        #     if pair_key in self.tradable_pairs:
        #         filtered_pairs[pair_key] = self.tradable_pairs[pair_key]
        #     else:
        #         logger.warning(f"白名单配对未找到: {pair_key}")

        # # 覆盖原结果
        # self.tradable_pairs = filtered_pairs

        # # 重新分配资金
        # if len(self.tradable_pairs) > 0:
        #     capital_per_pair = self.free_usdt / len(self.tradable_pairs)
        #     self.stake_allocations = {
        #         key: capital_per_pair for key in self.tradable_pairs
        #     }
        # else:
        #     self.stake_allocations = {}

        # logger.info(f"白名单过滤后剩余配对: {list(self.tradable_pairs.keys())}")

    def bot_start(self, **kwargs) -> None:
        # bot_start 运行时 DataProvider 尚未刷新首轮 OHLCV，候选池延后到 bot_loop_start 初始化。
        self.free_usdt = 1 * self.wallets.get_total('USDT')




    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        # 更新配对信息
        # 更新配对利润统计
        self._remove_invalid_tradable_pairs()

        pair_open_profit, pair_closed_profit, pair_closed_pair_count, total_asset = self. get_pair_profit_stats(current_time)
        self.free_usdt = total_asset
        if (not self.candidates_initialized) or current_time >= self.last_update_time + self.update_interval:
            self.free_usdt = 1 * self.wallets.get_total('USDT')
            self.all_candidate_pairs = self._initialize_all_candidates()
            self.candidates_initialized = True
            # logger.info(f"log____all_candidate_pairs，{self.all_candidate_pairs[:10]}")
            self._update_tradable_pairs()
            self.last_update_time = current_time
        
        # 更新配对利润统计
        self. get_pair_profit_stats(current_time)



    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:



        # 调试看frequi用
        dataframe['zero'] = 0
        dataframe['2'] = 2
        dataframe['-2'] = -2

        for pair in self.whitelist:
            if not self._is_pair_active(pair):
                continue
            pair_df = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe)
            if pair_df.empty:
                continue
            dataframe[f'{pair}_log_close'] = pair_df['close']

        ### 修改记录: 遍历tradable_pairs
        #logger.info(f"指标函数里的加载配对数量{len(self.tradable_pairs)}")
        for pair_key_str, params in self.tradable_pairs.items():
            pair_A = params['pair_A']  # 回归方程中的Y
            pair_B = params['pair_B']  # 回归方程中的X
            # logger.info(f"计算配对 {pair_A}_{pair_B} 的 Z-score 指标...")
            # # 打印date
            # logger.info(f"当前dataframe的date: {dataframe['date'].iloc[-5]}")

            if f'{pair_A}_log_close' in dataframe.columns and f'{pair_B}_log_close' in dataframe.columns:
                gamma = params.get('gamma')
                c = params.get('c')
                z_mean = params.get('z_mean')
                z_std = params.get('z_std')

                if all(p is not None for p in [gamma, c, z_mean, z_std]) and z_std != 0:
                    y_series = dataframe[f'{pair_A}_log_close']
                    x_series = dataframe[f'{pair_B}_log_close']
                    log_y_series =y_series
                    #logger.info(f"log_y_series for {pair_A}: {log_y_series.iloc[-5:].values}")
                    log_x_series = x_series
                    #logger.info(f"log_x_series for {pair_B}: {log_x_series.iloc[-5:].values}")
                    z_value = self.zvalue(log_y_series, log_x_series, gamma, c)
                    dataframe[f'Zvalue'] = z_value
                    #logger.info(f"计算配对 {pair_A}_{pair_B} 的 Z-value... Z-value for {pair_A}_{pair_B}: {z_value.iloc[-5:].values}")
                    #logger.info(f"计算---------------------------- {pair_A}_{pair_B} 的 Z-value: {z_value.iloc[-5]:.4f}")
                    z_score = (z_value - z_mean) / z_std
                    dataframe[f'{pair_A}_{pair_B}_Zscore'] = z_score
                    #logger.info(f"计算配对 {pair_A}_{pair_B} 的 Z-score... Z-score for {pair_A}_{pair_B}: {z_score.iloc[-20:].values}")
                    dataframe[f'{pair_A}_{pair_B}_Zscore_mean'] = dataframe[f'{pair_A}_{pair_B}_Zscore'].rolling(window=self.zscore_mean_window).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        current_pair = metadata['pair']
        # logger.info(f"--- [populate_entry_trend] for {current_pair} at {dataframe.iloc[-1]['date']} ---")

        for pair_key, params in self.tradable_pairs.items():
            pair_A = params['pair_A']
            pair_B = params['pair_B']
            zscore_col_name = f'{pair_A}_{pair_B}_Zscore'
            zscore_col_name_mean = f'{pair_A}_{pair_B}_Zscore_mean'

            if zscore_col_name in dataframe.columns and (current_pair == pair_A or current_pair == pair_B):
                zscore = dataframe[zscore_col_name]
                # zscore_mean = dataframe[zscore_col_name_mean]
                #zscore_diff = zscore - zscore.shift(1)
                # short_cond = (zscore > self.Zscore_entry) & (zscore_diff < 0)
                # long_cond = (zscore < -self.Zscore_entry) & (zscore_diff > 0)
                # short_cond = (zscore > self.Zscore_entry) & (zscore <= 1.5)
                # long_cond = (zscore < -self.Zscore_entry) & (zscore >= -1.5)
                # short_cond = (zscore > self.Zscore_entry) & (zscore.diff() < 0)
                # long_cond = (zscore < -self.Zscore_entry) & (zscore.diff() > 0)
                # short_cond = (zscore > self.Zscore_entry) &  (zscore < self.Zscore_entry1)
                # long_cond = (zscore < -self.Zscore_entry) &  (zscore > -self.Zscore_entry1)
                short_cond = zscore > self.Zscore_entry
                long_cond = zscore < -self.Zscore_entry
                # short_cond = (zscore > (1.00 * zscore_mean))  &  (zscore > 0.5)
                # long_cond = (zscore < (-1.00 * zscore_mean)) &  (zscore < -0.5)
                short_tag = f'entry_short_{pair_A}_{pair_B}'
                long_tag = f'entry_long_{pair_A}_{pair_B}'
                if current_pair == pair_A:
                    dataframe.loc[short_cond, ['enter_short', 'enter_tag']] = (1, short_tag)
                    dataframe.loc[long_cond, ['enter_long', 'enter_tag']] = (1, long_tag)
                if current_pair == pair_B:
                    dataframe.loc[short_cond, ['enter_long', 'enter_tag']] = (1, long_tag)
                    dataframe.loc[long_cond, ['enter_short', 'enter_tag']] = (1, short_tag)

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        current_pair = metadata['pair']
        for pair_key, params in self.tradable_pairs.items():
            pair_A = params['pair_A']
            pair_B = params['pair_B']
            zscore_col_name = f'{pair_A}_{pair_B}_Zscore'
            if zscore_col_name in dataframe.columns and (current_pair == pair_A or current_pair == pair_B):

                zscore = dataframe[zscore_col_name]
                exit_cond = (abs(zscore) < self.Zscore_exit)
                exit_tag = f'exit_{pair_A}_{pair_B}_get_profit'
                dataframe.loc[exit_cond, ['exit_short', 'exit_tag']] = (1, exit_tag)
                dataframe.loc[exit_cond, ['exit_long', 'exit_tag']] = (1, exit_tag)

                # 2. 止损平仓条件 (Stop Loss): 当价差持续发散时触发
                # Z-score的绝对值大于我们的止损阈值
                #sl_cond = (abs(zscore) > self.Zscore_stop)
                #exit_tag1 = f'exit_{pair_A}_{pair_B}_stoploss'
                #dataframe.loc[sl_cond, ['exit_short', 'exit_tag']] = (1, exit_tag1)
                #dataframe.loc[sl_cond, ['exit_long', 'exit_tag']] = (1, exit_tag1)

        return dataframe

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):


        pair_mates = self.get_pair_mates_from_stake_allocations(pair)
        if pair_mates is None:
            return None
        pair_A, pair_B = pair_mates
        #logger.info(f"检查配对 {pair_A}_{pair_B} 的平仓条件: 当前时间: {current_time}, 交易开仓时间: {trade.open_date_utc}, 当前币的利润为{current_profit}")
        open_pairs_profit = self.get_pair_profit_ratio_of_wallet(pair_A, pair_B)
        #logger.info(f"检查配对 {pair_A}_{pair_B} 的平仓条件: 当前配对利润占比: {open_pairs_profit}, 当前币的利润为{current_profit}")
        exit_time = trade.open_date_utc + timedelta(days=1)
        if open_pairs_profit > 0.01 and current_time > exit_time:
            #logger.info(f"配对 {pair_A}_{pair_B} 达到盈利目标，准备平仓。当前配对利润占比: {open_pairs_profit},当前币的利润为{current_profit}")
            return f"Pair_profit_get_profit_{pair_A}_{pair_B}"
        pair_key = f"{pair_A}_{pair_B}"
        state = self.pair_states.get(pair_key)
        if not state:
            return None
        if state.get('one_pair_is_already_exit') == 1:
            #logger.info(f"配对 {pair_A}_{pair_B} 达到盈利目标，准备平仓。当前配对利润占比: {open_pairs_profit},当前币的利润为{current_profit}")
            return f"Pair_already_exit_{pair_A}_{pair_B}"


        return None


    def _calculate_precise_stakes(
        self,
        pair_key: str,
        budget: float,
        pair_A_price: float,
        pair_B_price: float
    ) -> dict:
        """
        使用数量对冲法 (Unit Neutral) 计算仓位大小
        目标: qty_B = gamma * qty_A
        """

        params = self.tradable_pairs.get(pair_key)
        if not params:
            return {'stake_A': 0, 'stake_B': 0}

        gamma = params.get('gamma')
        pair_A = params['pair_A']
        pair_B = params['pair_B']

        # ===============================
        # 1. 基础校验
        # ===============================
        if gamma is None or gamma <= 0:
            logger.error(f"Gamma 值无效: {gamma}")
            return {'stake_A': 0, 'stake_B': 0}

        if pair_A_price <= 0 or pair_B_price <= 0:
            logger.error(f"价格无效: P_A={pair_A_price}, P_B={pair_B_price}")
            return {'stake_A': 0, 'stake_B': 0}

        # ===============================
        # 2. 数量对冲核心公式
        # qty_B = gamma * qty_A
        # budget = qty_A * P_A + qty_B * P_B
        # ===============================
        denom = pair_A_price + gamma * pair_B_price
        if denom <= 0:
            logger.error(f"分母异常 denom={denom}")
            return {'stake_A': 0, 'stake_B': 0}

        target_qty_A = budget / denom
        target_qty_B = gamma * target_qty_A

        # ===============================
        # 3. 获取交易所精度
        # ===============================
        market_A = self.dp.market(pair_A)
        market_B = self.dp.market(pair_B)

        amount_precision_A = market_A.get('precision', {}).get('amount')
        amount_precision_B = market_B.get('precision', {}).get('amount')

        if amount_precision_A is None or amount_precision_B is None:
            logger.error(f"无法获取 {pair_A} 或 {pair_B} 精度")
            return {'stake_A': 0, 'stake_B': 0}

        # ===============================
        # 4. 向下取整（避免超预算）
        # ===============================
        final_qty_B = np.floor(target_qty_B / amount_precision_B) * amount_precision_B
        final_qty_A = np.floor(target_qty_A / amount_precision_A) * amount_precision_A

        # Freqtrade保护
        final_qty_A = round(final_qty_A, 8)
        final_qty_B = round(final_qty_B, 8)

        # ===============================
        # 5. 重新计算资金（防止精度误差）
        # ===============================
        final_stake_A = final_qty_A * pair_A_price
        final_stake_B = final_qty_B * pair_B_price
        total_stake = final_stake_A + final_stake_B

        # ===============================
        # 6. 二次预算保护（极重要）
        # ===============================
        if total_stake > budget:
            scale = budget / total_stake

            final_qty_A = np.floor((final_qty_A * scale) / amount_precision_A) * amount_precision_A
            final_qty_B = np.floor((final_qty_B * scale) / amount_precision_B) * amount_precision_B

            final_qty_A = round(final_qty_A, 8)
            final_qty_B = round(final_qty_B, 8)

            final_stake_A = final_qty_A * pair_A_price
            final_stake_B = final_qty_B * pair_B_price
            total_stake = final_stake_A + final_stake_B

        # ===============================
        # 7. 最小下单限制
        # ===============================
        min_stake_limit = 5.0
        if final_stake_A < min_stake_limit or final_stake_B < min_stake_limit:
            # logger.warning(
            #     f"仓位过小，跳过 {pair_key} | "
            #     f"A=${final_stake_A:.2f}, B=${final_stake_B:.2f}"
            # )

            logger.info(
                f"[数量对冲] {pair_key} | "
                f"budget=${budget:.2f}, gamma={gamma:.4f} | "
                f"{pair_A}: qty={final_qty_A}, price={pair_A_price} | "
                f"{pair_B}: qty={final_qty_B}, price={pair_B_price} | "
                f"A=${final_stake_A:.2f}, B=${final_stake_B:.2f}, total=${total_stake:.2f}"
            )

            realized_gamma = final_qty_B / final_qty_A if final_qty_A > 0 else 0
            # logger.info(
            #     f"[Gamma验证] 目标={gamma:.4f}, 实际={realized_gamma:.4f}, "
            #     f"误差={(realized_gamma / gamma - 1):.4%}"
            # )

            return {'stake_A': 0, 'stake_B': 0}

        # ===============================
        # 8. 日志（核心调试信息）
        # ===============================
        # logger.info(
        #     f"[数量对冲] {pair_key} | "
        #     f"budget=${budget:.2f}, gamma={gamma:.4f} | "
        #     f"{pair_A}: qty={final_qty_A}, price={pair_A_price} | "
        #     f"{pair_B}: qty={final_qty_B}, price={pair_B_price} | "
        #     f"A=${final_stake_A:.2f}, B=${final_stake_B:.2f}, total=${total_stake:.2f}"
        # )

        # gamma验证（关键）
        if final_qty_A > 0:
            realized_gamma = final_qty_B / final_qty_A
            # logger.info(
            #     f"[Gamma验证] 目标={gamma:.4f}, 实际={realized_gamma:.4f}, "
            #     f"误差={(realized_gamma / gamma - 1):.4%}"
            # )

        return {
            'stake_A': final_stake_A,
            'stake_B': final_stake_B
        }


    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            leverage: float, entry_tag: str, side: str,
                            **kwargs) -> float:

        pair_key = '_'.join(entry_tag.split('_')[2:])
        if pair_key not in self.tradable_pairs or pair_key not in self.stake_allocations:
            return 0.0

        # 获取配对状态，如果不存在则为空字典
        state = self.pair_states.get(pair_key)

        if not state:
            budget = self.stake_allocations[pair_key]
            #logger.info(f"计算配对 {pair_key} 的仓位，预算: ${budget:.2f}")
            params = self.tradable_pairs[pair_key]
            pair_A = params['pair_A']
            pair_B = params['pair_B']
            if not self._is_pair_active(pair_A) or not self._is_pair_active(pair_B):
                self.tradable_pairs.pop(pair_key, None)
                self.stake_allocations.pop(pair_key, None)
                self.pair_states.pop(pair_key, None)
                return 0.0

            dataframe_A = self.dp.get_pair_dataframe(pair=pair_A, timeframe=self.timeframe)
            dataframe_B = self.dp.get_pair_dataframe(pair=pair_B, timeframe=self.timeframe)

            if dataframe_A.empty or dataframe_B.empty:
                return 0.0

            pair_A_price = dataframe_A.iloc[-1]['close']
            pair_B_price = dataframe_B.iloc[-1]['close']
            #logger.info(f"开始计算每对金额")
            stakes = self._calculate_precise_stakes(pair_key, budget, pair_A_price, pair_B_price)

            self.pair_states[pair_key] = {
                'stakes_calculated': True,
                'stake_A': stakes['stake_A'],
                'stake_B': stakes['stake_B'],
                'leg_A_opened': False,
                'leg_B_opened': False,
                'one_pair_is_already_exit': 0,
                'entry_tag': entry_tag
            }

            state = self.pair_states.get(pair_key)

        pair_A = self.tradable_pairs[pair_key]['pair_A']
        stake_to_return = state['stake_A'] if pair == pair_A else state['stake_B']
        # logger.info(f"当前配对状态: {pair_key}: {self.pair_states[pair_key]}")
        return stake_to_return

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, side: str, **kwargs) -> bool:
        #logger.info(f"确认买入 {pair} ({side})，订单类型: {order_type}, 数量: {amount}, 价格: {rate}, 时间有效性: {time_in_force}")
        if not self._is_pair_active(pair):
            return False

        pair_mates = self.get_pair_mates_from_stake_allocations(pair)
        if pair_mates is None:
            return False
        pair_A, pair_B = pair_mates
        pair_key = f"{pair_A}_{pair_B}"
        state = self.pair_states.get(pair_key)
        if not state:
            return False

        if pair == pair_A and state['leg_A_opened'] == False:
            state['leg_A_opened'] = True
            #logger.info(f"配对 {pair_key} 的 A腿 开仓成功，状态更新为: {state}")
        if pair == pair_B and state['leg_B_opened'] == False:
            state['leg_B_opened'] = True
            #logger.info(f"配对 {pair_key} 的 B腿 开仓成功，状态更新为: {state}")
        
        #logger.info(f"交易 {pair} ({side}) 开仓成功，当前配对状态: {state}")
        #logger.info(f"----------------------------------------买入日志结束---------------------------------------")
        return True



    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           exit_tag: str = None, **kwargs) -> bool:
        #logger.info(f"----------------------------------------卖出日志开始---------------------------------------")
        enter_tag = trade.enter_tag
        # logger.info(f"确认买入{pair} ({side})")
        try:
            pair_key = '_'.join(enter_tag.split('_')[2:])
        except IndexError:
            # logger.warning(f"无法从 entry_tag '{enter_tag}' 中解析出配对key。")
            return False

        state = self.pair_states.get(pair_key)
        if not state:
            return False
        params = self.tradable_pairs[pair_key]
        pair_A = params['pair_A']
        pair_B = params['pair_B']

        if pair == pair_A:
            if state['leg_A_opened'] == True and state['leg_B_opened'] == True and exit_reason == f"Pair_profit_get_profit_{pair_A}_{pair_B}":
                state['one_pair_is_already_exit'] = 1
                #logger.info(f"-----------------------触发one_pair_is_already_exit置1---------------------")
            if state['leg_A_opened'] == True and state['leg_B_opened'] == False and state['one_pair_is_already_exit'] == 1 and exit_reason == f"Pair_already_exit_{pair_A}_{pair_B}":
                #logger.info(f"-----------------------触发one_pair_is_already_exit置0---------------------")
                state['one_pair_is_already_exit'] = 0
            state['leg_A_opened'] = False
        if pair == pair_B:
            if state['leg_A_opened'] == True and state['leg_B_opened'] == True and exit_reason == f"Pair_profit_get_profit_{pair_A}_{pair_B}":
                state['one_pair_is_already_exit'] = 1
                #logger.info(f"-----------------------触发one_pair_is_already_exit置1---------------------")
            if state['leg_A_opened'] == False and state['leg_B_opened'] == True and state['one_pair_is_already_exit'] == 1 and exit_reason == f"Pair_already_exit_{pair_A}_{pair_B}":
                #logger.info(f"-----------------------触发one_pair_is_already_exit置0---------------------")
                state['one_pair_is_already_exit'] = 0
            state['leg_B_opened'] = False

        # 计算浮动收益率
        if trade.is_short:
            profit_ratio = 1 - (rate / trade.open_rate)
        else:
            profit_ratio = (rate/ trade.open_rate) - 1

        profit_ratio *= trade.leverage
        profit_amount = trade.stake_amount * profit_ratio


        logger.info(f"交易 {trade.id} ({pair}) 平仓，平仓原因: {exit_reason}, 平仓结束，当前配对状态: {state}，平仓数量{amount}")
        #logger.info(f"------------- ---------------------------卖出日志结束---------------------------------------")

        if not state['leg_A_opened'] and not state['leg_B_opened']:
            self.pair_states.pop(pair_key,None)

        return True



    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str, side: str,
                 **kwargs) -> float:
        return 1.0



    def get_pair_mates_from_stake_allocations(
        self,
        pair: str
    ) -> Optional[Tuple[str, str]]:
        """
        输入单币 pair（如 'OGN/USDT:USDT'）
        从 self.stake_allocations 中解析并返回对应的配对 (pair_a, pair_b)

        如果未找到，返回 None
        """

        for pair_key in self.stake_allocations.keys():
            try:
                pair_a, pair_b = pair_key.split("_")
            except ValueError:
                continue

            if pair == pair_a or pair == pair_b:
                return pair_a, pair_b

        return None


    def get_pair_profit_ratio_of_wallet(self, coin_a: str, coin_b: str) -> float:
        """
        仅统计 coin_a / coin_b 相关的【开仓浮动利润】
        占总钱包 USDT 的比例
        """
        target_coins = {coin_a, coin_b}
        total_open_profit = 0.0

        open_trades = Trade.get_trades_proxy(is_open=True)

        for trade in open_trades:
        
            # 只关心包含目标币种的 trade
            if trade.pair not in target_coins :
                continue

            (df, _) = self.dp.get_analyzed_dataframe(
                pair=trade.pair,
                timeframe=self.timeframe
            )

            last_price = df.iloc[-1]["close"]

            # 计算浮动收益率
            if trade.is_short:
                profit_ratio = 1 - (last_price / trade.open_rate)
            else:
                profit_ratio = (last_price / trade.open_rate) - 1

            profit_ratio *= trade.leverage
            profit_amount = trade.stake_amount * profit_ratio

            total_open_profit += profit_amount
        #logger.info(f"配对 {coin_a}_{coin_b} 的总开仓浮动利润: {total_open_profit:.2f} USDT")
        total_usdt = self.wallets.get_total('USDT')
        #logger.info(f"当前钱包总 USDT 余额: {total_usdt:.2f} USDT")
        if total_usdt <= 0:
            return 0.0

        return total_open_profit / total_usdt



    def get_pair_profit_stats(self, current_time: datetime):
        """
        统计配对交易盈亏和关仓配对数
        - pair_open_profit: 当前开仓浮动利润
        - pair_closed_profit: 已关仓累计利润
        - pair_closed_pair_count: 已关仓配对次数（每 2 条 trade 视为 1 个配对）
        """

        pair_open_profit = {}
        pair_closed_profit = {}
        pair_closed_trade_count = {}   # 单腿关仓数
        pair_closed_pair_count = {}    # 完整配对数

        total_closed_profit = 0.0
        total_open_profit = 0.0
        total_closed_trade_count = 0


        # ============================
        # 1️⃣ 构建反向映射：单币 pair -> 配对 key
        # ============================
        pair_reverse_map = {}
        for pair_key in self.stake_allocations.keys():
            try:
                pair_a, pair_b = pair_key.split("_")
            except ValueError:
                continue
            pair_reverse_map[pair_a] = pair_key
            pair_reverse_map[pair_b] = pair_key

        # ============================
        # 2️⃣ 已关仓利润（realized） + 关仓次数
        # ============================
        closed_trades = Trade.get_trades_proxy(is_open=False)

        for trade in closed_trades:
            pair_key = pair_reverse_map.get(trade.pair)
            if pair_key is None:
                continue

            # —— 配对维度 —— #
            pair_closed_profit.setdefault(pair_key, 0.0)
            pair_closed_profit[pair_key] += trade.close_profit_abs

            pair_closed_trade_count.setdefault(pair_key, 0)
            pair_closed_trade_count[pair_key] += 1

            # —— 全币种维度 —— #
            total_closed_profit += trade.close_profit_abs
            total_closed_trade_count += 1

        # ============================
        # 3️⃣ 当前开仓利润（unrealized）
        # ============================
        open_trades = Trade.get_trades_proxy(is_open=True)

        for trade in open_trades:
            pair_key = pair_reverse_map.get(trade.pair)
            if pair_key is None:
                continue

            (df, _) = self.dp.get_analyzed_dataframe(
                pair=trade.pair,
                timeframe=self.timeframe
            )
            if df.empty:
                continue

            last_price = df.iloc[-1]["close"]

            if trade.is_short:
                profit_ratio = 1 - (last_price / trade.open_rate)
            else:
                profit_ratio = (last_price / trade.open_rate) - 1

            profit_ratio *= trade.leverage
            profit_amount = trade.stake_amount * profit_ratio

            pair_open_profit.setdefault(pair_key, 0.0)
            pair_open_profit[pair_key] += profit_amount

            
            # —— 全币种维度 —— #
            total_open_profit += profit_amount

        # ============================
        # 4️⃣ 计算完整配对次数（每 2 条 trade 为 1 个配对）
        # ============================
        for pair_key, trade_cnt in pair_closed_trade_count.items():
            pair_closed_pair_count[pair_key] = trade_cnt // 2

        # ============================
        # 5️⃣ 每 288 次调用打印一次日志
        # ============================
        self._pair_profit_log_counter += 1

        should_log = (self._pair_profit_log_counter % 288 == 0)

        if should_log:
            logger.info(f"=========={current_time} Pair Profit Summary ==========")

            for pair_key in self.stake_allocations.keys():
                open_pnl = pair_open_profit.get(pair_key, 0.0)
                closed_pnl = pair_closed_profit.get(pair_key, 0.0)
                total_pnl = open_pnl + closed_pnl

                closed_pairs = pair_closed_pair_count.get(pair_key, 0)

                logger.info(
                    f"{pair_key} | "
                    f"ClosedPairs: {closed_pairs} | "
                    f"Open: {open_pnl:.2f} USDT | "
                    f"Closed: {closed_pnl:.2f} USDT | "
                    f"Total: {total_pnl:.2f} USDT"
                )
        
        total_asset = total_open_profit + total_closed_profit + self.dry_run_wallet


        return pair_open_profit, pair_closed_profit, pair_closed_pair_count, total_asset


















    #frequi查看z分数回归情况的一些调试代码
    def zvalue(self, y, x, gamma, c):
        if gamma is None or c is None:
            logger.error("gamma或c值未设置，无法计算Z值。")
            return None

        Z = y - (gamma * x + c)
        return Z

    def Yvalue(self, y, x, gamma, c):
        if gamma is None or c is None:
            logger.error("gamma或c值未设置，无法计算Y值。")
            return None

        Y = gamma * x + c
        return Y

    def change_y(self, y, x, gamma, c):
        """
        计算差值占y值的百分比
        """
        if gamma is None or c is None:
            logger.error("gamma或c值未设置，无法计算Z值。")
            return None

        Y = gamma * x + c
        y_change = (Y - y) / y * 100  # 计算差值占Y的百分比
        return y_change
