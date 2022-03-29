import ctpbee
from ctpbee import CtpBee
from ctpbee import CtpbeeApi
from ctpbee.constant import *

app = CtpBee("somewheve", __name__)

# Need parameter to get want you need
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

# class MacdStrategy(CtpbeeApi):
#     def __init__(self, name):
#         super().__init__(name)


#     def on_realtime(self):
#         # run on_realtime each 1 second
#         return super().on_realtime()
    
#     def on_tick(self, tick: TickData) -> None:
#         # tick data trigger
#         print(f"on_tick: {tick}")
#         pass

#     def on_3_min_bar(self, bar: BarData) -> None:
#         print(f"on_3_min_bar: {bar}")

#     def on_bar(self, bar: BarData) -> None:
#         print(f"on_bar: {bar}")

#     def on_contract(self, contract: ContractData) -> None:
#         print(f"on_contract: {contract}")
#         return super().on_contract(contract)

app.config.from_mapping(info)
# macd = MacdStrategy()
# app.add_extension(macd)
app.start()