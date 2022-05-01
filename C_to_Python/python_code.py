# import ctypes

# libtest = ctypes.cdll.LoadLibrary("./test.so")

# lst = [1,2,3,4]
# Array4 = ctypes.c_int * 4
# array = Array4(*lst)

# print(libtest.sum(array, 4))

import ctypes  
ll = ctypes.cdll.LoadLibrary   

lib = ll("./libtest.so") 
input1 = 100
input2 = 220
result1 = lib.add(input1,input2)
result2 = lib.main()
print(result1,result2)
print('***finish***') 