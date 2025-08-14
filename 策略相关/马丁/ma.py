from datetime import datetime, timedelta
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from freqtrade.strategy import DecimalParameter, IntParameter
from functools import reduce
import warnings
from freqtrade.exchange import date_minus_candles

import logging

logger = logging.getLogger(__name__)


class ma(IStrategy):
    # 基本配置
    minimal_roi = {
        "0": 1
    }
    stoploss = -1
    timeframe = '5m'
    stoploss = -1.0  # 不设置止损，由策略自行控制
    trailing_stop = False
    # 策略自定义参数
    entry_step_pct = -0.10
    entry_stake_amount = 100
    DCA_STAKE_AMOUNT = 10

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        用于计算指标。
        """
        # 短期和长期均线
        dataframe['ma7'] = ta.MA(dataframe, timeperiod=7)
        dataframe['ma120'] = ta.MA(dataframe, timeperiod=120)

        return dataframe



    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['ma7'] > dataframe['ma120'])
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "long__buy")

        return dataframe


    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        return dataframe

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float | None, max_stake: float,
                            leverage: float, entry_tag: str | None, side: str,
                            **kwargs) -> float:
        return self.entry_stake_amount

    position_adjustment_enable = True

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float | None, max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs
                              ) -> float | None | tuple[float | None, str | None]:
        if self.wallets:
            free_usdt = self.wallets.get_free('USDT')
            if trade.nr_of_successful_entries == 1:
                self.DCA_STAKE_AMOUNT = max(10, free_usdt / 100)
                #logger.info(f"Adjusting entry stake amount to {self.entry_stake_amount} USDT based on available balance.,Free USDT: {free_usdt}")
        last_time = trade.date_last_filled_utc + timedelta(days=5)
        if current_profit < self.entry_step_pct and trade.nr_of_successful_entries != 200 and current_time >= last_time and free_usdt >= 3 * self.DCA_STAKE_AMOUNT:
            #logger.info(f"Adjusting position for {trade.pair} at {current_time}: current_profit={current_profit},open_rate={trade.open_rate}, current_rate={current_rate}, nr_of_successful_entries={trade.nr_of_successful_entries},")
            #logger.info(f"--------当前可用USDT--------------: {self.wallets.get_free('USDT')}")
            return self.DCA_STAKE_AMOUNT
        return None



    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        current_profit_stake = current_profit * trade.stake_amount
        if current_profit >= 0.01:
            logger.info(f"Exiting trade for {pair} at {current_time}: current_profit_stake={current_profit_stake},total_usdt={self.wallets.get_total('USDT')}")
            return "profit_1%"
        return None