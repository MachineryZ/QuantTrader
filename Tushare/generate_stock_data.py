import tushare
import pandas
import datetime
import numpy as np

def main():
    tocken = open('./tocken.txt').read()
    tushare.set_token(tocken)
    today_date = datetime.date.today()
    end_date = str(today_date.year) + '0' + str(today_date.month) + str(today_date.day) \
        if today_date.month < 10 else str(today_date.year) + str(today_date.month) + str(today_date.day)
    df = tushare.pro_bar(
        ts_code="000001.SZ",
        adj="qfq",
        start_date="20150101",
        end_date=end_date,
        asset="E",
        freq="D",
        ma=[10],
    )
    df.to_feather("/home/zzz/Data/stock/s1d/000001.feather")

if __name__ == '__main__':
    main()