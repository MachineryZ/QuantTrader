import backtrader as bt
import numpy as np
import pandas
import datetime
import matplotlib.pyplot as plt

class ComminfoFuturesPercent(bt.CommInfoBase):
    '''write by myself,using in the future backtest,it means we should give a percent comminfo to broker'''
    params = (
        ('stocklike', False),
        ('commtype', bt.CommInfoBase.COMM_PERC),
        ('percabs', False)
    )

    def _getcommission(self, size, price, pseudoexec):
        return abs(size) * price * self.p.mult * self.p.commission

    def get_margin(self, price):
        return price * self.p.mult * self.p.margin

class TurtleStrategy(bt.Strategy):
    params = (
        ('long_period', 20),
        ('short_period', 10),
        ('doprint', False),
        ('data_index', None),
        ('save_csv_file', False)
    )

    def __init__(self):
        # create trading history
        self.history = np.zeros((self.data.buflen(), 3))
        self.history[0, 1] = self.broker.getvalue()

        self.data_close = self.datas[0].close
        self.order = None
        self.buyprice = 0
        self.buycomm = 0
        self.buy_size = 0
        self.buy_count = 0

        self.H_line = bt.indicators.Highest(self.data.high(-1), period=self.params.long_period)
        self.L_line = bt.indicators.Lowest(self.data.low(-1), period=self.params.short_period)
        self.TR = bt.indicators.Max(
            self.data.high(0) - self.data.low(0),
            abs(self.data.close(-1) - self.data.high(0)),
            abs(self.data.close(-1) - self.data.low(0))
        )
        self.ATR = bt.indicators.SimpleMovingAverage(self.TR, period=14)
        self.buy_signal = bt.ind.CrossOver(self.data.close(0), self.H_line)
        self.sell_signal = bt.ind.CrossOver(self.data.close(0), self.L_line)

    def log(self, txt, dt=None, doprint=False):
        """Logging function for this strategy"""
        if doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log("OPERATION PROFIT, GROSS %.2f, NET %.2f" % \
            (trade.pnl, trade.pnlcomm), doprint=self.params.doprint)

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
                    doprint=self.params.doprint,
                )
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log("SELL EXECUTED, Size %i, Price: %.2f, Cost: %.2f, Comm %.5f" %
                    (
                        order.executed.size,
                        order.executed.price,
                        order.executed.value,
                        order.executed.comm
                    ), 
                    doprint=self.params.doprint,
                )
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected", doprint=self.params.doprint)

        # Write down: no pending order
        self.order = None

    def stop(self):
        # record the strategy result
        trade_result = pandas.DataFrame(self.history)
        trade_result.columns = {'Close', 'Value', 'Signal'}
        trade_result.index = self.params.data_index
        if self.params.save_csv_file is True:
            trade_result.to_csv('turtle_result.csv')

    def prenext(self):
        self.curbar = len(self) - 1
        self.history[self.curbar, 0] = self.data_close[0]
    
    def nextstart(self):
        self.curbar = len(self) - 1
        self.history[self.curbar, 0] = self.data_close[0]
        self.history[:self.curbar + 1, 1] = self.broker.getvalue()

    def next(self):
        if self.order:
            return
        # buy: price go through the H_line and position is zero
        if self.buy_signal > 0 and self.buy_count == 0:
            self.buy_size = self.broker.getvalue() * 0.01 / self.ATR
            self.buy_size = int(self.buy_size / 100) * 100
            self.sizer.p.stake = self.buy_size
            self.buy_count = 1
            self.order = self.buy()
        elif self.data.close[0] > self.buyprice + 0.5 * self.ATR[0] and \
            self.buy_count > 0 and self.buy_count <= 4:
            self.buy_size = self.broker.getvalue() * 0.01 / self.ATR
            self.buy_size = int(self.buy_size / 100) * 100
            self.sizer.p.stake = self.buy_size
            self.order = self.buy()
            self.buy_count += 1
        elif self.sell_signal < 0 and self.buy_count > 0:
            self.order = self.sell()
            self.buy_count = 0
        elif self.data.close < (self.buyprice - 2 * self.ATR[0]) and self.buy_count > 0:
            self.order = self.sell()
            self.buy_count = 0

class TurtleSizer(bt.Sizer):
    params = (
        ('stake', 1),
    )
    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy:
            return self.p.stake
        position = self.broker.getposition(data)
        if not position.size:
            return 0
        else:
            return position.size
        return self.p.stake

def save_plot(cerebro: bt.Cerebro):
    pass

def pipeline(file_name):
    # Create cerebro engine
    cerebro = bt.Cerebro()
    
    # Load Data
    ori_data = pandas.read_csv(file_name)

    # Add Turtle Strategy into cerebro
    cerebro.addstrategy(TurtleStrategy, data_index=ori_data['date'])

    # Load data into cerebro
    data = bt.feeds.GenericCSVData(
        dataname=file_name,
        fromdate=datetime.datetime(2009, 1, 1),
        todate=datetime.datetime(2019, 4, 12),
        timeframe=bt.TimeFrame.Minutes,
        dtformat="%Y-%m-%d %H:%M:%S",
        tmformat="%H:%M:%S",
        datetime=1,
        open=2,
        high=3,
        low=4,
        close=5,
        volume=6,
    )
    cerebro.adddata(data)

    # set initial cash and commission
    start_cash = 500000
    commission = 0.0002
    stake_size = 1 # fix size

    cerebro.broker.setcash(start_cash)
    # cerebro.addsizer(bt.sizers.FixedSize, stake=stake_size)
    cerebro.addsizer(TurtleSizer)
    
    cerebro.addanalyzer(bt.analyzers.Returns, _name="Return")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="Sharpe_Ratio")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="Drawdown")

    # Print initial cash
    print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())
    
    backtest = cerebro.run()

    Return = backtest[0].analyzers.Return.get_analysis()['rtot'] * 100
    Sharpe_Ratio = backtest[0].analyzers.Sharpe_Ratio.get_analysis()['sharperatio']
    Max_Drawdown = backtest[0].analyzers.Drawdown.get_analysis()['max']['drawdown']
    
    
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    print('Return: %.2f%%' % Return)
    print('Sharpe Ratio: %.2f' % Sharpe_Ratio)
    print('Max Drawdown: %.2f%%' % Max_Drawdown)

    try:
        turtle_result = pandas.read_csv('./turtle_result.csv')
    except FileNotFoundError:
        print("There is no turtle_result.csv")
    except PermissionError:
        print("You do not have permission to access the file")

    # cerebro.plot()

if __name__ == '__main__':
    pipeline(file_name='./data/RB9999.csv')
