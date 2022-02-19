import datetime
import backtrader as bt
import os
import numpy as np
import pandas


class TestStrategy(bt.Strategy):
    params = (

    )

    def log(self, txt, dt=None):
        """Logging function for this strategy"""
        dt = dt or self.datas[0].datetime.date[0]
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):

        self.macd = bt.indicators.MACD

        # self.dataclose = self.datas[0].close
        # self.volume = self.datas[0].volume

        # self.order = None
        # self.buyprice = None
        # self.buycomm = None

        # self.high_n = bt.indicators.Highest(self.data.high, period=self.params.n)
        # self.low_n = bt.indicators.Lowest(self.data.low, period=self.params.n)

        # self.rsv = 100 * bt.DivByZero(
        #     self.data_close - self.low_n, 
        #     self.high_n - self.low_n,
        #     zero=None
        # )
        # self.K = bt.indicators.EMA(self.rsv, period=self.params.m)
        # self.D = bt.indicators.EMA(self.K, period=self.params.l)
        # self.J = self.params.S * self.D - (self.params.S - 1) * self.K

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
                self.bar_executed_close = self.dataclose[0]
            else:
                pass
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            pass
        self.order = None
    
    def notify_trade(self, trade):
        if not trade.isclosed:
            return

    def next(self):
        if self.order:
            return
        
        """
        long: when macd is above 0, ema-10 up through ema-20
        close long position: price down through ema-10
        short: when macd is below 0, ema-10 down through ema-20
        close short position: price is above ema-10
        stake: 1
        initial cash: 50000
        commision: 0.0002

        """

                    
if __name__ == '__main__':
    # Create cerebro
    cerebro = bt.Cerebro()

    cerebro.addstrategy(TestStrategy)

    # load data
    data = bt.feeds.GenericCSVData(
        dataname="600519.csv", # white wine csv
        fromdate=datetime.datetime(2009, 3, 30),
        todate=datetime.datetime(2018, 12, 4),
        dtformat="%Y%m%d",
        datetime=1,
        open=2,
        high=3,
        low=4,
        close=5,
        volume=6,
        reverse=False, 
    )

    cerebro.adddata(data)

    cerebro.broker.setcash(50000) # let initial cash greater, so that garuntee is not enough

    cerebro.addsizer(bt.sizers.FixedSize, stake=100)

    cerebro.broker.setcommission(commission=0.002) 

    print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())

    cerebro.run()

    print("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())
    

            



