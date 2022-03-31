# https://www.bilibili.com/video/BV1Tz4y1o79w?spm_id_from=333.999.0.0
import backtrader

def f(x):
    return -1200 + 500/(1+x) + 500/(1+x)**2 + 500/(1+x)**3

x_left = 0.0
x_right = 1.0
x = (x_left + x_right)/2
epsilon = 1e-5
while f(x) > epsilon or f(x) < -epsilon:
    if f(x) > epsilon:
        x_left = (x_left + x_right) / 2
    else:
        x_right = (x_left + x_right) / 2
    x = (x_left + x_right) / 2

print(x)

