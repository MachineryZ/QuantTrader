import datetime
import backtrader as bt
import os
import numpy as np
import pandas
from backtrader.comminfo import CommissionInfo
import pyfolio

class MacdEmaTrendFollowStrategy(bt.Strategy):
    params = (
        ('period_me1', 10),
        ('period_me2', 20),
        ('period_signal', 9),
        ('data_index', 0)
    )

    def log(self, txt, dt=None, doprint=False):
        """Logging function for this strategy"""
        dt = dt or self.datas[0].datetime.date[0]
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):

        self.ema10 = bt.indicators.EMA(self.datas[0], period=self.params.period_me1)
        self.ema20 = bt.indicators.EMA(self.datas[0], period=self.params.period_me2)
        self.macd = bt.indicators.macd(self.datas[0], period_me1=self.params.period_me1, period_me2=self.params.period_me2)

        self.bar_num = 0

        self.history = np.zeros((self.data.buflen(), 4))
        self.history[0, 1] = self.broker.getvalue()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log("BUY EXECUTED, Size %i, Price: %.2f, Cost: %.2f, Comm: %.5f" %
                    (   
                        order.executed.size,
                        order.executed.price,
                        order.executed.value,
                        order.executed.comm
                    ), 
                    doprint=True
                )
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
                self.bar_executed_close = self.dataclose[0]
            else:
                self.log("SELL EXECUTED, Size %i, Price: %.2f, Cost: %.2f, Comm %.5f" %
                    (
                        order.executed.size,
                        order.executed.price,
                        order.executed.value,
                        order.executed.comm
                    ), 
                    doprint=True
                )
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Cancled/Margin/Rejected", doprint=True)

        # Write down: no pending order
        self.order = None
    
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log("OPERATION PROFIT, GROSS %.2f, NET %.2f" % (trade.pnl, trade.pnlcomm), doprint=True)

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
        #  
        if self.order:
            return
    
    def stop(self):
        self.log("(me1 Period %i) (me2 Period %i) (period_signal Period %i) Ending Value %.2f" %
            (
                self.params.period_me1,
                self.params.period_me2,
                self.params.period_
            )
        )

        
                    
if __name__ == '__main__':
    # Create cerebro
    cerebro = bt.Cerebro()

    cerebro.addstrategy(MacdEmaTrendFollowStrategy)

    # load data
    data = bt.feeds.GenericCSVData(
        dataname="RB9999.csv", # white wine csv
        fromdate=datetime.datetime(2009, 3, 30),
        todate=datetime.datetime(2018, 12, 4),
        dtformat="%Y-%m-%d %H:%M:%S",
        datetime=1,
        open=2,
        high=3,
        low=4,
        close=5,
        volume=6,
        reverse=True, 
    )

    cerebro.adddata(data)

    cerebro.broker.setcash(50000) # let initial cash greater, so that garuntee is not enough

    cerebro.addsizer(bt.sizers.FixedSize, stake=100)

    cerebro.broker.setcommission(commission=0.002)

    print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())

    cerebro.run()

    print("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())
    

            



