# C++ 与 python 的通信

简单的demo
1. 编写c++程序
2. 生成.so文件
   1. 注意：文件名必须以lib开头
   2. 生成命令为：g++ xxx.cpp -fpic -shared -o libxxx.so
3. 编写python程序
4. 注意调用的ctypes的库
我的这个例子就是
~~~bash
g++ cpp_code.cpp -fpic -shared -o libtest.so
python python_code.py
~~~
这样生成so和运行python文件即可

