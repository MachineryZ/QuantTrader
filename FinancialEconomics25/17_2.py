import numpy
import math
from torch import inverse


def main():
    r = 0.00011
    sigma = 0.177
    K = 2.5
    T = 49
    s0 = 2.341
    u = pow(math.exp(1), sigma * math.sqrt(1/246))
    d = 1/u
    q = (1+r - d) / (u - d) # 0.507857
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
        # option_T.append( 
        #     max( -(pow(u, stock_price[-1][i]) * s0 - K), # This is where we need to change, to suit for put or call option price 
        #     0)
        # )
        option_T.append(
            max(0, max( K - pow(u, stock_price[-1][i]) * s0 ,0))
        )
    option_price.append(option_T)
    # Calculate the option price in each period (0-T-1):
    for t in reversed(range(0, T)):
        option_list = []
        for i in range(len(option_price[T - (t+1)]) - 1):
            price = 1/(1+r) * (q * option_price[T - (t+1)][i] +
                         (1-q) * option_price[T - (t+1)][i+1])
            option_list.append(price)
        option_price.append(option_list)
    print(option_price[-1])



    
        
if __name__ == '__main__':
    main()


"""
在bilibili的https://www.bilibili.com/video/BV1uU4y1H77H/?spm_id_from=333.788
第17章的内容里
初始数据为：
r = 0.00011
sigma = 0.177
K = 2.5
T = 15/30
s0 = 2.341
美式卖出期权现在的价格
0.17015327087246174

"""