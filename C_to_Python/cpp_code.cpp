// #include <stdio.h>

// int sum(const int* arr, size_t len)
// {
// 	int sum = 0;
// 	for(size_t i = 0; i != len; ++i)
// 		sum += arr[i];
// 	return sum;
// }

#include <stdio.h>
using namespace std;

extern "C"{

   double add(int, int);

}
double add(int x1, int x2)
{
    return x1+x2;
}
int main()
{
  int a = 1;
  int b =2 ;
  int c;
  c = add(a,b);
  return c;
}