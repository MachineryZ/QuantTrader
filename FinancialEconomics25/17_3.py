import numpy
import math
from torch import inverse


def Normal():
    q = 0.5
    u = math.exp(0.1)
    d = math.exp(-0.1)
    print(f"u = {u}")
    print(f"d = {d}")
    r_bar = 0.055
    r0 = 0.05
    s0 = 100
    T = 10

    # Get the ratio list of r
    r = []
    for t in range(0, T+1):
        r_list = []
        ratio = t 
        while(ratio >= -t):
            r_list.append(ratio)
            ratio -= 2
        r.append(r_list)
    
    B = [s0 / T * i for i in reversed(range(T+1))]

    value = []
    v = []
    for i in range(len(r[-1])):
        v.append(0.0)
    value.append(v)

    for t in reversed(range(0, T)):
        v = []
        for i in range(len(r[t+1]) - 1):
            v.append(1/(1 + r0 * math.pow(u, r[t][i])) * 
             (q * (r_bar * B[t] + B[t] - B[t+1] + value[T-t-1][i]) + 
            (1-q) * (r_bar * B[t] + B[t] - B[t+1] + value[T-t-1][i+1])))
        value.append(v)
    print(value[-1]) # 102.05162190626781

    


def Immediate():
    q = 0.5
    u = math.exp(0.1)
    d = math.exp(-0.1)
    print(f"u = {u}")
    print(f"d = {d}")
    r_bar = 0.055
    r0 = 0.05
    s0 = 100
    T = 10

    # Get the ratio list of r
    r = []
    for t in range(0, T+1):
        r_list = []
        ratio = t 
        while(ratio >= -t):
            r_list.append(ratio)
            ratio -= 2
        r.append(r_list)
    
    B = [s0 / T * i for i in reversed(range(T+1))]

    value = []
    v = []
    for i in range(len(r[-1])):
        v.append(0.0)
    value.append(v)
    for t in reversed(range(0, T)):
        v = []
        for i in range(len(r[t+1]) - 1):
            v.append(min(1/(1 + r0 * math.pow(u, r[t][i])) * 
             (q * (r_bar * B[t] + B[t] - B[t+1] + value[T-t-1][i]) + 
            (1-q) * (r_bar * B[t] + B[t] - B[t+1] + value[T-t-1][i+1])), B[t]))
        value.append(v)
    print(value[-1]) # 99.97262273916917
        
if __name__ == '__main__':
    Normal()
    Immediate()

"""
如果没有option
那么这笔贷款的价值是：
102.05162190626781
如果又option可以直接付清贷款，那么：
99.97262273916917
"""