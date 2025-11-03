from datetime import datetime, timedelta
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.strategy import merge_informative_pair
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from freqtrade.strategy import DecimalParameter, IntParameter
from functools import reduce
from freqtrade.persistence import Trade
import pandas as pd
import numpy as np
import warnings
from freqtrade.exchange import date_minus_candles
import logging

logger = logging.getLogger(__name__)

'''
martin short long 策略
策略整体思路：
1、主流币做多马丁，非主流币做空马丁，将原来的两个策略融合在一起，资金分开管理


20251028
1、主流币做多马丁，非主流币做空马丁，将原来的两个策略融合在一起，避免极端行情某个方向爆仓，资金分开管理

20251030
1、使用dataframe['volume_mean'] > 400000辅助买入，和原版差距不大，估计400000这个阈值还需要再调整-------X
2、加仓限制次数，利润大幅降低，回撤也大幅降低-------X 
3、做空使用动态止盈（1d数据），交易笔数下降很多，做空总利润几乎一致-------X
4、做多使用动态止盈（1d数据），交易笔数下降很多，做多总利润提升不大-------X
5、做多调整为5x（使做多盈利能力和做空盈利能力一致），初始仓位由200刀降低到10刀，做多和做空的加仓调整为10天固定时间加仓，金额波动更为温和，最后回测结束时，不再有1000多天的扛单，利润下降了一些-------✔
'''


class MD_SL(IStrategy):

    def __init__(self, config):
        super().__init__(config)
        self.dry_run_wallet = config.get('dry_run_wallet', 4000) if hasattr(config, 'get') else 4000
        self.total_short_usdt = 0
        self.total_long_usdt = 0


    can_short = True
    timeframe = '1d'  #
    startup_candle_count = 60  # 需要至少60根K线
    process_only_new_candles = True  # 只处理新K线



    stoploss = -10000
    use_custom_stoploss = False  # 启用动态止损
    # short交易参数
    leverage1 = 20
    REAL_USE_MUL = 10
    liqutation_ratio = 0.0
    entry_step_pct = -5
    entry_stake_amount = 6
    num_entry = 300




    short_profit_ratio = 0

    # long自定义参数
    long_entry_step_pct = -0.1
    long_entry_stake_amount = 10
    long_DCA_STAKE_AMOUNT = 10
    long_profit_ratio = 0




    long_pair_list = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "XRP/USDT:USDT",
    "BNB/USDT:USDT",
    "SOL/USDT:USDT",
    "DOGE/USDT:USDT",
    "ADA/USDT:USDT",
    "TRX/USDT:USDT",
    "WBTC/USDT:USDT",
    "SUI/USDT:USDT",
    "XLM/USDT:USDT",
    "LINK/USDT:USDT",
    "HBAR/USDT:USDT",
    "BCH/USDT:USDT",
    "AVAX/USDT:USDT"
                    ]
    



    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        # 在每个bot循环开始时更新资产情况
        logger.info(f"当前日期:{current_time}")
        self.total_short_usdt, self.short_profit_ratio, self.total_long_usdt, self.long_profit_ratio, real_usdt,total_profit_pct  = self.get_assets_in_usdt()







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
        dataframe['ma5'] = ta.SMA(dataframe, timeperiod=5)  # 计算5日均线
        dataframe['ma60'] = ta.SMA(dataframe, timeperiod=60)  # 计算60日均线
  


        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        enter_short_conditions = [
            metadata['pair'] not in self.long_pair_list,
            dataframe['momentum_rank'] <= 20,
            dataframe['close'] < dataframe['ma60'],
        ]
        if enter_short_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
            ] = (1, "short")

        enter_long_conditions = [
            metadata['pair']  in self.long_pair_list,
            dataframe['close'] > dataframe['ma5'],
        ]

        if enter_long_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
            ] = (1, "long")

        enter_long_conditions1 = [
            metadata['pair']  in self.long_pair_list,
            dataframe['ma5'] > dataframe['ma60'],
        ]

        if enter_long_conditions1:
            dataframe.loc[
                reduce(lambda x, y: x & y, enter_long_conditions1), ["enter_long", "enter_tag"]
            ] = (1, "long1")       

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe


    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float | None, max_stake: float,
                            leverage: float, entry_tag: str | None, side: str,
                            **kwargs) -> float:
        if side == 'short':
            return self.entry_stake_amount / self.REAL_USE_MUL
        return self.long_entry_stake_amount

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, side: str, **kwargs) -> bool:
        if side == 'short':
            if self.short_profit_ratio < -30:
                return False
        return True

    position_adjustment_enable = True

    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float, current_profit: float,
                              min_stake: float | None, max_stake: float,
                              current_entry_rate: float, current_exit_rate: float, current_entry_profit: float,
                              current_exit_profit: float, **kwargs
                              ) -> float | None | tuple[float | None, str | None]:


        if  trade.trade_direction == 'short':
            last_time = trade.date_last_filled_utc + timedelta(days=10)
            if current_time > last_time:
                return self.entry_stake_amount / self.REAL_USE_MUL
        elif trade.trade_direction == 'long':
            last_time = trade.date_last_filled_utc + timedelta(days=10)
            if current_time > last_time:
                return self.long_DCA_STAKE_AMOUNT



        return None



    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        if trade.trade_direction == 'short':
            exit_time = trade.open_date_utc + timedelta(days=1)
            exit_time1 = trade.open_date_utc + timedelta(days=100)

            if current_time >= exit_time and current_profit > 0.5:
                return "profit_50%_time_exit"

            if current_time >= exit_time1 and current_profit > 0:
                return "time_exit_100day"





        elif trade.trade_direction == 'long':
            if current_profit > 0.05:
                return "long_profit_5%_exit"
            exit_time1 = trade.open_date_utc + timedelta(days=100)

            if  current_time >= exit_time1 and current_profit > 0:
                return "long_time_exit_100day"


        return None

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           exit_tag: str = None, **kwargs) -> bool:

        return True

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str, side: str,
                 **kwargs) -> float:
        if side == 'short':
            return self.leverage1
        return 5


    def get_assets_in_usdt(self):

        initial_usdt = self.dry_run_wallet

        open_profit = 0
        open_trades = Trade.get_trades_proxy(is_open=True)
        total_profit_amount = 0.0
        for open_trade in open_trades:
            profit_ratio = 0.0
            other_pair = open_trade.pair
            (dataframe, _) = self.dp.get_analyzed_dataframe(pair=other_pair, timeframe=self.timeframe)
            last_candle = dataframe.iloc[-1]
            rate_for_other_pair = last_candle['close']

            if open_trade.is_short:
                    # Short position profit ratio
                profit_ratio = (1 - (rate_for_other_pair / open_trade.open_rate))
            else:
                # Long position profit ratio
                profit_ratio = ((rate_for_other_pair / open_trade.open_rate) - 1)

            profit_ratio *= open_trade.leverage
            profit_amount = open_trade.stake_amount * profit_ratio
            total_profit_amount += profit_amount
            open_profit = total_profit_amount

        close_profit = Trade.get_total_closed_profit()
        # 获取USDT本身余额
        total_usdt = self.wallets.get_total('USDT')
        real_usdt = total_usdt + open_profit

        total_profit_pct = 100 * ((open_profit + close_profit)   / initial_usdt) 



        long_close_trade_profit = 0
        short_open_trade_profit = 0
        short_close_trade_profit = 0
        long_open_trade_profit = 0
        short_open_stake_amount = 0
        long_open_stake_amount = 0

        all_close_trades = Trade.get_trades_proxy(is_open=False)
        close_long_trades = [trade for trade in all_close_trades if trade.trade_direction == 'long']
        for trade in close_long_trades:
            long_close_trade_profit += float(trade.close_profit_abs) 
            #logger.info(f"{trade.pair} 平仓盈亏: {trade.close_profit_abs}")


        all_open_trades = Trade.get_trades_proxy(is_open=True)
        open_short_trades = [trade for trade in all_open_trades if trade.trade_direction == 'short']
        for trade in open_short_trades:
            profit_ratio = 0.0
            open_trade = trade
            other_pair = open_trade.pair
            (dataframe, _) = self.dp.get_analyzed_dataframe(pair=other_pair, timeframe=self.timeframe)
            last_candle = dataframe.iloc[-1]
            rate_for_other_pair = last_candle['close']

            if open_trade.is_short:
                    # Short position profit ratio
                profit_ratio = (1 - (rate_for_other_pair / open_trade.open_rate))
            else:
                # Long position profit ratio
                profit_ratio = ((rate_for_other_pair / open_trade.open_rate) - 1)

            profit_ratio *= open_trade.leverage
            profit_amount = open_trade.stake_amount * profit_ratio
            short_open_trade_profit += profit_amount

            short_open_stake_amount+= float(trade.stake_amount)



        close_short_trades = [trade for trade in all_close_trades if trade.trade_direction == 'short']
        for trade in close_short_trades:
            short_close_trade_profit += float(trade.close_profit_abs)

        open_long_trades = [trade for trade in all_open_trades if trade.trade_direction == 'long']
        for trade in open_long_trades:
            open_trade = trade
            profit_ratio = 0.0
            other_pair = open_trade.pair
            (dataframe, _) = self.dp.get_analyzed_dataframe(pair=other_pair, timeframe=self.timeframe)
            last_candle = dataframe.iloc[-1]
            rate_for_other_pair = last_candle['close']

            if open_trade.is_short:
                    # Short position profit ratio
                profit_ratio = (1 - (rate_for_other_pair / open_trade.open_rate))
            else:
                # Long position profit ratio
                profit_ratio = ((rate_for_other_pair / open_trade.open_rate) - 1)

            profit_ratio *= open_trade.leverage
            profit_amount = open_trade.stake_amount * profit_ratio
            long_open_trade_profit += profit_amount



            long_open_stake_amount += float(trade.stake_amount)



        total_short_usdt = 0.5 * initial_usdt + short_close_trade_profit + short_open_trade_profit
        total_long_usdt = 0.5 * initial_usdt + long_close_trade_profit + long_open_trade_profit


        

        short_profit_ratio = 100 * short_open_trade_profit /  total_short_usdt 
        short_profit_real_ratio = 100 * (short_close_trade_profit + short_open_trade_profit) /  (0.5 * initial_usdt) 
        long_profit_ratio = 100 * long_open_trade_profit /  total_long_usdt 
        long_profit_real_ratio = 100 * (long_close_trade_profit + long_open_trade_profit) /  (0.5 * initial_usdt) 





        logger.info(f"获取资产情况")
        logger.info(f"--short")
        logger.info(f"-----short开仓金额: {short_open_stake_amount}") 
        logger.info(f"-----short持仓利润: {short_open_trade_profit}") 
        logger.info(f"-----short关仓利润: {short_close_trade_profit}") 
        logger.info(f"-----short账户实际余额: {total_short_usdt:.2f} ")
        logger.info(f"-----short账户利润百分比（持仓利润除以short实际余额）: {short_profit_ratio:.2f} %")
        logger.info(f"-----short账户真实盈利百分比（所有short订单利润除以short初始资金）: {short_profit_real_ratio:.2f} %")
        logger.info(f"--long")
        logger.info(f"-----long开仓金额: {long_open_stake_amount}")
        logger.info(f"-----long持仓利润: {long_open_trade_profit}")
        logger.info(f"-----long关仓利润: {long_close_trade_profit}")
        logger.info(f"-----long账户总额: {total_long_usdt:.2f} ")
        logger.info(f"-----long账户利润百分比（持仓利润除以long实际余额）: {long_profit_ratio:.2f} %")
        logger.info(f"-----long账户真实盈利百分比（所有long订单利润除以long初始资金）: {long_profit_real_ratio:.2f} %")
        logger.info(f"--总览")  
        logger.info(f"-----当前账户实际余额(已包含持仓): {real_usdt:.2f} USDT")
        logger.info(f"-----当前持仓利润: {total_profit_amount:.2f} USDT")
        logger.info(f"-----当前账户真实盈利百分比（所有订单利润除以初始资金）: {total_profit_pct:.2f} %")


        
        return total_short_usdt, short_profit_ratio, total_long_usdt, long_profit_ratio, real_usdt, total_profit_pct









    











