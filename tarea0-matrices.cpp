#include <iostream>
#include <vector>
#include <time.h>
#include <ctime>
const int SIZE = 100;
using namespace std;
int main()
{
// A:
clock_t begin = clock();
int a[SIZE][SIZE];
for(int i=0;i<SIZE;i++)
{
for(int j=0;j<SIZE;j++)
a[j][i] = i+j;
}
double aTime = double(clock() - begin) / CLOCKS_PER_SEC;
cout<<"Elapsed time: "<<aTime<<"s"<<endl;
// B:
begin = clock();
int b[SIZE][SIZE];
for(int i=0;i<SIZE;i++)
{
for(int j=0;j<SIZE;j++)
b[i][j] = i+j;
}
double bTime = double(clock() - begin) / CLOCKS_PER_SEC;
cout<<"Elapsed time: "<<bTime<<"s"<<endl;
cout<<"Gain: "<<aTime/bTime<<endl;
}
