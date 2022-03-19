import tushare
import pandas
import pymysql

tushare.set_token("3d45894b34e8525357cbb56778152da599698faa7d4a865097b6941d")
pro = tushare.pro_api()

df_tradeData = pro.query('trade_cal', start_date='20220101', end_date='20221231')
print(df_tradeData)