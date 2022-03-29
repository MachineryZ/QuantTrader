from ctpbee import CtpBee
from ctpbee import CtpbeeApi
from ctpbee.constant import *


class CTA(CtpbeeApi):
    def __init__(self, name):
        super().__init__(name)

    def on_init(self, init: bool) -> None:  # 初始化完成回调 
        self.info("init successful")

    def on_tick(self, tick: TickData) -> None:
        print(tick.datetime, tick.last_price)  # 打印tick时间戳以及最新价格 

        # 买开
        self.action.buy_open(tick.last_price, 1, tick)
        # 买平
        self.action.buy_close(tick.last_price, 1, tick)
        # 卖开
        self.action.sell_open(tick.last_price, 1, tick)
        # 卖平 
        self.action.sell_close(tick.last_price, 1, tick)

        # 获取合约的仓位
        position = self.center.get_position(tick.local_symbol)
        print(position)

    def on_contract(self, contract: ContractData) -> None:
        if contract.local_symbol == "rb2205.SHFE":
            self.action.subscribe(contract.local_symbol)  # 订阅行情 
            print("合约乘数: ", contract.size)


if __name__ == '__main__':
    app = CtpBee('ctp', __name__)
    info = {
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
    "INTERFACE": "ctp",  # 接口声明
    "TD_FUNC": True,  # 开启交易功能
    "MD_FUNC": True
    }
    app.config.from_mapping(info)  # loading config from dict object
    cta = CTA("cta")
    app.add_extension(cta)
    app.start() 