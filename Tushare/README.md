# Tushare Data

generate_trade_cal.py: generate trading calendar (up to newest)



### Knowledge


除权，复权
**除权**：假设某公司的总股本100股，每股10元，总权益就是1000元。当公司进行送股，每10股转送10股，总股本就变成了200股，而公司的总权益并没有变，于是每股价格就要变成1000/200=5元。这就是除权。
**复权**：当股价因送股、配股等原因而发生下跌时，原来10元/股的股票瞬间变成了5元/股，但该股票实际价值并没有发生变化，也就是说现在的5元实际上还是相当于10元。这就是复权。
前复权，不复权，后复权
**不复权**：当股价因送股等原因发生变化，在K线走势图上就有可能形成断崖式的下跌，比如从10元/股变为5元/股，当日的涨跌幅就变成了-50%.而实际上该股票的价值并没有发生重大变化，而当这个价格变化反应到技术指标上时，就可能影响到指标的准确性，影响到部分投资者的判断。
**前复权**：以除权后第一天的价格点为基础把除权以前的数据进行复权。
**后复权**：以除权前最后一天的价格点为基础把除权后的数据进行复权。


