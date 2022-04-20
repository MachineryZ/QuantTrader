#include <iostream>
#include <string>
#include <vector>
#include <cmath>
using namespace std;
/*
假设摸球赌局里，有30个红球和30个绿球
这个赌局的支付期望是多少
*/
int main()
{
    int red = 30;
    int green = 30;
    vector<vector<double>> dp(red+1, vector<double>(green+1, 0));
    for(int i = 0; i <= red; i++)
        dp[i][0] = i;
    for(int j = 1; j <= red; j++)
    {
        for(int i = 1; i <= green; i++)
        {
            dp[i][j] = max(0.0, double(i)/(i+j) * (1 + dp[i-1][j]) + 
                double(j)/(i+j) * (-1 + dp[i][j-1]));
            // std::cout << i << " " << j << " " << dp[i][j] << endl;
        }
    }
    // for(int i = 0; i <= red; i++)
    // {
    //     for(int j = 0; j <= red; j++)
    //     {
    //         std::cout << dp[i][j] << " ";
    //     }
    //     std::cout << endl;
    // }
    std::cout << dp[red][green] << endl; //2.82339
    return 0;
}