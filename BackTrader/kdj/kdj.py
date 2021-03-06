import datetime
import backtrader as bt
import os

class TestStrategy(bt.Strategy):
    params = (
        ('n', 14),
        ('m', 3),
        ('l', 3),
        ('S', 3),
        ('opt', True), # can set to false
    )
    def log(self, txt, dt=None):
        """Logging function for this strategy"""
        dt = dt or self.datas[0].datetime.date[0]
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.volume = self.datas[0].volume

        self.order = None
        self.buyprice = None
        self.buycomm = None

        self.high_n = bt.indicators.Highest(self.data.high, period=self.params.n)
        self.low_n = bt.indicators.Lowest(self.data.low, period=self.params.n)

        self.rsv = 100 * bt.DivByZero(
            self.data_close - self.low_n, 
            self.high_n - self.low_n,
            zero=None
        )
        self.K = bt.indicators.EMA(self.rsv, period=self.params.m)
        self.D = bt.indicators.EMA(self.K, period=self.params.l)
        self.J = self.params.S * self.D - (self.params.S - 1) * self.K

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
        
        if self.params.opt is False: # use original strategy
            if not self.position:
                if ((self.K[-1] < 30) and (self.K[0] > 30)) or \
                ((self.K[-1] < 70) and (self.K[0] > 70)):
                    self.order = self.buy()
            else:
                if ((self.K[-1] > 30) and (self.K[0] < 30)) or \
                ((self.K[-1] > 70) and (self.K[0] < 70)):
                    self.order = self.sell()
        elif self.params.opt is True: # use optimized strategy
            if not self.position:
                if self.K[0] > 70 or ((self.K[-1] > 70) and (self.K[0] < 70) and (self.K[0] > 30)):
                    self.order = self.buy()
            else:
                if ((self.K[-1] < 30) and (self.K[0] > 30) and (self.K[0] < 70)) or \
                ((self.K[-1] > 70) and (self.K[0] < 30)):
                    self.order = self.sell()
                    
if __name__ == '__main__':
    # Create cerebro
    cerebro = bt.Cerebro()

    cerebro.addstrategy(TestStrategy)

    # load data
    data = bt.feeds.GenericCSVData(
        dataname="600519.csv", # white wine csv
        fromdate=datetime.datetime(2010, 1, 1),
        todate=datetime.datetime(2020, 4, 21),
        dtformat="%Y%m%d",
        datetime=2,
        open=3,
        high=4,
        low=5,
        close=6,
        volume=10,
        reverse=True, 
    )

    cerebro.adddata(data)

    cerebro.broker.setcash(1000000) # let initial cash greater, so that garuntee is not enough

    cerebro.addsizer(bt.sizers.FixedSize, stake=100)

    cerebro.broker.setcommission(commission=0.005) 

    print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())

    cerebro.run()

    print("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())

    cerebro.plot()
    

            



