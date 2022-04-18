import tushare
import pandas
import pymysql

# Tushare data api
#https://tushare.pro/document/2

tushare.set_token("3d45894b34e8525357cbb56778152da599698faa7d4a865097b6941d")
pro = tushare.pro_api()

# ==================== 沪深股票 基础数据 ====================
df_tradeData = pro.query('trade_cal', start_date='20220101', end_date='20221231')
# print(df_tradeData)

# 查询当前所有正常上市交易的股票列表
data = pro.stock_basic(exchange='', list_status='L', 
    fields='ts_code,symbol,name,area,industry,list_date')
# print(data.columns)

#查询交易日
df = pro.trade_cal(exchange='', start_date='20100101', end_date='20220401')
# print(df)

#取000001的前复权行情
df = tushare.pro_bar(ts_code='000001.SZ', adj='qfq', start_date='20180101', end_date='20181011')
print(df)