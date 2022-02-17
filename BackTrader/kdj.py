import datetime
import backtrader as bt
import os

class TestStrategy(bt.Strategy):
    params = (
        ('n', 14),
        ('m', 3),
        ('l', 3),
        ('S', 3),
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
        

