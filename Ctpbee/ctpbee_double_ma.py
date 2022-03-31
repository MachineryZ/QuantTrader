from ctpbee import CtpbeeApi, CtpBee
from ctpbee.constant import Offset, TradeData, Direction
from ctpbee.indicator.ta_lib import ArrayManager


class DoubleMaStrategy(CtpbeeApi):
    def __init__(self, name):
        super().__init__(name)
        self.manager = ArrayManager(100)
        self.instrument_set = ["rb2101.SHFE"]  # 
        self.buy = 0
        self.sell = 0
        self.slow = 60
        self.fast = 30

    def on_trade(self, trade: TradeData):
        if trade.offset == Offset.OPEN:
            if trade.direction == Direction.LONG:
                self.buy += trade.volume
            else:
                self.sell += trade.volume
        else:
            if trade.direction == Direction.LONG:
                self.sell -= trade.volume
            else:
                self.buy -= trade.volume

    def on_bar(self, bar):
        """ """
        self.manager.add_data(bar)
        if not self.manager.inited:
            return
        fast_avg = self.manager.sma(self.fast, array=True)
        slow_avg = self.manager.sma(self.slow, array=True)

        if slow_avg[-2] < fast_avg[-2] and slow_avg[-1] >= fast_avg[-1]:
            self.action.cover(bar.close_price, self.buy, bar)
            self.action.sell(bar.close_price, 3, bar)

        if fast_avg[-2] < slow_avg[-2] and fast_avg[-1] >= slow_avg[-1]:
            self.action.sell(bar.close_price, self.sell, bar)
            self.action.buy(bar.close_price, 3, bar)

    def on_tick(self, tick):
        pass

    def on_init(self, init: bool):
        print("init success")

app = CtpBee("doublema", __name__, refresh=True)
app.config.from_mapping({
    "CONNECT_INFO": {
        "userid": "089131",
        "password": "350888",
        "brokerid": "9999",
        "md_address": "tcp://218.202.237.33:10112",
        "td_address": "tcp://218.202.237.33:10102",
        "product_info": "",
        "appid": "simnow_client_test",
        "auth_code": "0000000000000000"
    },
    "INTERFACE": "ctp",  # 
    "TD_FUNC": True,  # 
    "MD_FUNC": True,
    "XMIN": [1]
})
strategy = DoubleMaStrategy("doublema")
app.add_extension(strategy)
app.start()

if __name__ == '__main__':
    app = CtpBee("doublema", __name__, refresh=True)
    app.config.from_mapping({
        "CONNECT_INFO": {
            "userid": "089131",
            "password": "350888",
            "brokerid": "9999",
            "md_address": "tcp://218.202.237.33:10112",
            "td_address": "tcp://218.202.237.33:10102",
            "product_info": "",
            "appid": "simnow_client_test",
            "auth_code": "0000000000000000"
        },
        "INTERFACE": "ctp",  # 
        "TD_FUNC": True,  # 
        "MD_FUNC": True,
        "XMIN": [1]
    })
    strategy = DoubleMaStrategy("doublema")
    app.add_extension(strategy)
    app.start()