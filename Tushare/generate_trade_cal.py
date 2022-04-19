import tushare
import datetime

def main():
    tocken = open('./tocken.txt').read()
    tushare.set_token(tocken)
    pro = tushare.pro_api()
    today_date = datetime.date.today()
    end_date = str(today_date.year) + '0' + str(today_date.month) + str(today_date.day) \
        if today_date.month < 10 else str(today_date.year) + str(today_date.month) + str(today_date.day)
    dataframe = pro.trade_cal(exchange='', start_date='20150101', end_date=end_date)
    """
            exchange  cal_date  is_open pretrade_date
    0         SSE  20150101        0      20141231
    1         SSE  20150102        0      20141231
    2         SSE  20150103        0      20141231

    """
    dataframe.to_feather('/home/zzz/Data/trade_cal.feather', compression='uncompressed')
    

if __name__ == '__main__':
    main()
