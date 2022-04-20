import numpy
import math
from torch import inverse

def main():
    r = 1.0001
    sigma = 0.25
    K = 2.6
    T = 30
    s0 = 2.5
    u = pow(math.exp(1), sigma * math.sqrt(1/246))
    d = 1/u
    q = (r - d) / (u - d) # 0.507857
    stock_price = []
    option_price = []
    print(f"u = {u}")
    print(f"d = {d}")
    print(f"q = {q}")
    print(f"r = {r}")
    
    # Get the index 
    for t in range(0, T+1):
        price_list = []
        price = t
        while(price >= -t):
            price_list.append(price)
            price -= 2
        stock_price.append(price_list)
    # print(stock_price[-1])

    # Calculate the latest call or put option price
    option_T = []
    for i in range(len(stock_price[-1])):
        # Put price calculation
        option_T.append( 
            max( -(pow(u, stock_price[-1][i]) * s0 - K), # This is where we need to change, to suit for put or call option price 
            0)
        )
    # print(option_T)
    option_price.append(option_T)
    # Calculate the option price in each period (0-T-1):
    for t in reversed(range(0, T)):
        option_list = []
        for i in range(len(option_price[T - (t+1)]) - 1):
            price = 1/r * (q * option_price[T - (t+1)][i] +
                         (1-q) * option_price[T - (t+1)][i+1])
            option_list.append(price)
        option_price.append(option_list)
    print(option_price[-1])

    
        
if __name__ == '__main__':
    main()


"""
在bilibili的https://www.bilibili.com/video/BV1XP4y147Xx/?spm_id_from=333.788
第16章的内容里，数据变更为：
验证put-call parity等式：
初始数据为：
r = 1.0001
sigma = 0.25
K = 2.6
T = 15/30
s0 = 2.5

C + Ke^{-rt} = P + S0 (这里的r对应上面的1.0001-1=0.0001)

对于15个交易日到期的买入期权价格应该是：
0.026490201312083822
对于15个交易日到期的卖出期权价格应该是：
0.12259331954487887
等式左侧：2.622593124850132
等式右侧：2.6225933195448787
几乎完美

对于30个交易日到期的买入期权价格应该是：
0.050265598909718476
对于30个交易日到期的卖出期权价格应该是：
0.14247767602434963
等式左侧：2.6463685224477667
等式右侧：2.64247767602435
也是非常准确，但是相比较15个交易日来说，会稍微的误差大
说明时间越久，put-call parity的等式越会被挑战
"""