from datetime import datetime, timedelta
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.strategy import merge_informative_pair
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from freqtrade.strategy import DecimalParameter, IntParameter
from functools import reduce
import pandas as pd
import warnings
from freqtrade.exchange import date_minus_candles

import logging

logger = logging.getLogger(__name__)


# 纯山寨币做空策略 策略风险较大，适合大牛市开始回撤时使用   讨论群 https://discord.gg/hfxRh74mK7

'''
dryrun



ROI: Closed trades
∙ 520.179 USDT (67.99%) (26.27 Σ%)
∙ 520.179 USD
ROI: All trades
∙ 454.301 USDT (57.85%) (22.94 Σ%)
∙ 454.301 USD
Total Trade Count: 1129
Bot started: 2025-08-01 09:58:11
First Trade opened: 3 months ago (2025-08-01 10:12:56)
Latest Trade opened: 34 minutes ago (2025-11-03 01:06:45)
Win / Loss: 1060 / 0
Winrate: 100.00%
Expectancy (Ratio): 0.49 (100.00)
Avg. Duration: 3 days, 23:14:35
Best Performing: ATA/USDT:USDT: 7.513 USDT (69.14%)
Trading volume: 31252.268 USDT
Profit factor: inf
Max Drawdown: 0.00% (0 USDT)
    from 2025-08-02 10:13:49 (0.554 USDT)
    to 2025-08-02 10:13:49 (0.554 USDT)


Monday (count)    USDT         USD        Profit %
----------------  -----------  ---------  ----------
2025-11-03 (16)   8.578 USDT   8.58 USD   0.34%
2025-10-27 (129)  73.535 USDT  73.54 USD  3.05%
2025-10-20 (70)   35.088 USDT  35.09 USD  1.48%
2025-10-13 (228)  97.9 USDT    97.90 USD  4.29%
2025-10-06 (119)  75.842 USDT  75.84 USD  3.44%
2025-09-29 (71)   30.198 USDT  30.20 USD  1.39%
2025-09-22 (117)  64.642 USDT  64.64 USD  3.06%
2025-09-15 (41)   16.154 USDT  16.15 USD  0.77%
2025-09-08 (17)   11.559 USDT  11.56 USD  0.56%
2025-09-01 (62)   23.449 USDT  23.45 USD  1.14%
2025-08-25 (48)   19.565 USDT  19.57 USD  0.96%
2025-08-18 (67)   25.629 USDT  25.63 USD  1.27%
2025-08-11 (33)   15.018 USDT  15.02 USD  0.75%
2025-08-04 (33)   15.559 USDT  15.56 USD  0.78%
2025-07-28 (9)    7.466 USDT   7.47 USD   0.38%
2025-07-21 (0)    0 USDT       0.00 USD   0.00%
2025-07-14 (0)    0 USDT       0.00 USD   0.00%
2025-07-07 (0)    0 USDT       0.00 USD   0.00%
2025-06-30 (0)    0 USDT       0.00 USD   0.00%
2025-06-23 (0)    0 USDT       0.00 USD   0.00%






'''


class SL_0707(IStrategy):
    can_short = True
    timeframe = '1d'  # 
    startup_candle_count = 30  # 需要至少30根K线
    process_only_new_candles = True  # 只处理新K线

    entry_step_pct = -5
    entry_stake_amount = 6
    num_entry = 300
    # 交易参数
    stoploss = -10000
    use_custom_stoploss = False  # 启用动态止损
    REAL_USE_MUL = 10
    liqutation_ratio = 0.0

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        current_pair = metadata['pair']
        whitelist = self.dp.current_whitelist()
        new_cols = {}
        for pair in whitelist:
            pair_df = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe)
            new_cols[f'{pair}_1d_per'] = (pair_df['close'] / pair_df['close'].shift(1)) - 1
        # 合并所有新列
        new_cols_df = pd.DataFrame(new_cols, index=dataframe.index)
        dataframe = pd.concat([dataframe, new_cols_df], axis=1)
        # 只保留实际存在的列
        valid_cols = [col for col in [f'{pair}_1d_per' for pair in whitelist] if col in dataframe.columns]
        current_col = f'{current_pair}_1d_per'
        if not valid_cols or current_col not in valid_cols:
            dataframe['momentum_rank'] = None
        else:
            dataframe['momentum_rank'] = dataframe[valid_cols].rank(axis=1, ascending=False)[current_col]
        dataframe['ma60'] = ta.SMA(dataframe, timeperiod=60)  # 计算60日均线
        return dataframe


    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        生成买入信号：
        - 做多排名前 2 的币种
        - 做空排名后 2 的币种
        """
        #dataframe.loc[dataframe['momentum_rank'] <= 2, 'enter_long'] = 1  # 排名前 2 的做多
                # 空头条件
        enter_short_conditions = [
            dataframe['momentum_rank'] <= 10,
            dataframe['close'] < dataframe['ma60'],
        ]
        if enter_short_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
            ] = (1, "short")
        #return dataframe
    
        #dataframe.loc[dataframe['momentum_rank'] >= dataframe['momentum_rank'].max() - 1, 'enter_short'] = 1  # 排名后 2 的做空
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['exit_long'] = 0  
        dataframe['exit_short'] = 0  
        return dataframe  

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                        current_profit: float, **kwargs):

            exit_time = trade.open_date_utc + timedelta(days=1)
            exit_time1 = trade.open_date_utc + timedelta(days=100)  
            exit_time2 = trade.open_date_utc + timedelta(days=200)  
            if current_time >= exit_time and current_profit > 0.5:
                return "profit_50%_time_exit"
            if current_time >= exit_time1 and current_profit > 0:
                return "time_exit_100day"
            #if current_time >= exit_time2 :
            #    return "time_exit_200day"

            open_trades = Trade.get_trades_proxy(is_open=True)
            all_profits = {}
            total_profit_amount = 0.0

            for open_trade in open_trades:
                profit_ratio = 0.0

                if open_trade.pair == pair:
                    profit_ratio = current_profit
                else:
                    other_pair = open_trade.pair
                    (dataframe, _) = self.dp.get_analyzed_dataframe(pair=other_pair, timeframe=self.timeframe)

                    if not dataframe.empty:
                        last_candle = dataframe.iloc[-1]
                        rate_for_other_pair = last_candle['close']

                        if open_trade.is_short:
                            # Short position profit ratio
                            profit_ratio = (1 - (rate_for_other_pair / open_trade.open_rate))
                        else:
                            # Long position profit ratio
                            profit_ratio = ((rate_for_other_pair / open_trade.open_rate) - 1)

                        profit_ratio *= open_trade.leverage

                profit_pct = profit_ratio * 100
                profit_amount = open_trade.stake_amount * profit_ratio
                total_profit_amount += profit_amount

                all_profits[open_trade.pair] = f"{profit_pct:.2f}% ({profit_amount:.2f} {open_trade.stake_currency})"

            ratio = total_profit_amount / self.wallets.get_total('USDT')
            if ratio < -1:
                logger.warning(f"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.")
                logger.warning(f"Total profit ratio is too low!!!!!!liqutation_ratio!!!!!!!!: {ratio:.2%}.")
                logger.warning(f"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.")
                return None
            self.liqutation_ratio = ratio

            if pair == open_trades[0].pair:
                #logger.info(f"--- Backtest time: {current_time} ---")
                #logger.info(f"Current profit for all open trades: {all_profits}")
                #logger.info(f"Total profit amount for all open trades: {total_profit_amount:.2f} USDT")
                logger.info(f"Total profit ratio: {ratio:.2%}")
                #logger.info(f"trade number:{len(all_profits)}")
                #logger.info("---------------------------------")

            return None
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float | None, max_stake: float,
                            leverage: float, entry_tag: str | None, side: str,
                            **kwargs) -> float:
        #logger.info(f"Calculating custom stake amount for {pair} at {current_time}: self.entry_stake_amount={self.entry_stake_amount}")
        return self.entry_stake_amount / self.REAL_USE_MUL

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, side: str, **kwargs) -> bool:
        if self.liqutation_ratio < -0.5:
            logger.warning(f"Liquidity ratio is too low ({self.liqutation_ratio}), skipping trade entry for {pair}.")
            return False
        logger.info(f"Confirming trade entry for {pair}: order_type={order_type}, amount={amount}, rate={rate}, side={side}")
        return True

    position_adjustment_enable = True

    def adjust_trade_position(self, trade: Trade, current_time: datetime,current_rate: float, current_profit: float,min_stake: float | None, max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,current_entry_profit: float, current_exit_profit: float,**kwargs
                              ) -> float | None | tuple[float | None, str | None]:
        current_profit_stake = current_profit * trade.stake_amount
        current_profit_stake_ratio = current_profit_stake /  self.wallets.get_total('USDT')
        #logger.info(f"Adjusting position for {trade.pair} at {current_time}: current_profit={current_profit},trade.stake_amount={trade.stake_amount} current_profit_stake={current_profit_stake} current_profit_stake_ratio = {current_profit_stake_ratio}")
        last_time = trade.date_last_filled_utc + timedelta(days=7)
        if self.wallets:
            free_eth = 0.95*self.wallets.get_free('USDT')
            used_eth = self.wallets.get_used('USDT')
            total_eth = self.wallets.get_total('USDT')
            ratio = self.wallets.get_free('USDT') / self.wallets.get_total('USDT')
            #logger.info(f"Free USDT: {free_eth}, Used USDT: {used_eth}, Total USDT: {total_eth}")
            if trade.nr_of_successful_entries == 1:
                self.entry_stake_amount = max(6, free_eth / self.num_entry)
                #self.entry_stake_amount = 6
                #logger.info(f"Adjusting entry stake amount to {self.entry_stake_amount} USDT based on available balance.,Free USDT: {free_eth}")
        if current_profit < self.entry_step_pct and trade.nr_of_successful_entries != self.num_entry and current_time > last_time  and  current_profit_stake_ratio > -0.05 :
            #logger.info(f"Adjusting position for {trade.pair} at {current_time}: current_profit={current_profit}, nr_of_successful_entries={trade.nr_of_successful_entries}")
            if trade.nr_of_successful_entries < self.num_entry:
                logger.info(f"Adjusting position for {trade.pair} at {current_time}: current_profit={current_profit}, nr_of_successful_entries={trade.nr_of_successful_entries}, entry_stake_amount={self.entry_stake_amount}")
            return self.entry_stake_amount  / self.REAL_USE_MUL
        
        return None

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           exit_tag: str = None, **kwargs) -> bool:
        #logger.info(f"exit-----------------------------------------------------------------------------exit")
        if self.wallets:
            free_usdt = 0.95*self.wallets.get_free('USDT')
            used_usdt = self.wallets.get_used('USDT')
            total_usdt = self.wallets.get_total('USDT')
        logger.info(f"Confirming trade exit for {pair}: order_type={order_type}, amount={amount}, rate={rate}, exit_reason={exit_reason}, free_usdt: {free_usdt}, used_usdt: {used_usdt}, total_usdt: {total_usdt}")

        return True


    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str, side: str,
                 **kwargs) -> float:
        return 20


              


    











