#pragma once
#include "../LOAD/MnistData.hpp"
#include <vector>
#include <map>
#include <omp.h>
#include <mutex>

namespace RNN
{
namespace MnistKNN
{

struct Compare
{
double distance;
int id;
int value;
};

double distance( const mat& xi, const mat& xj )
{
    double distance = 0;
    for(int i = 0; i < 784; i++)
    {
        distance += ( xi(0,i)-xj(0,i) ) * ( xi(0,i)-xj(0,i) );
    }
    return sqrt(distance);
}

//  g++ Source.cpp -O3 -fopenmp -march=native
//  10000 * 55000 , k = 50  5min 95.14%
//  g++ Source.cpp -03
//  10000 * 55000 , k = 50  14min 95.29%
//  不加锁的话，很多push_back是失败的，说明是非VECTOR的PUSH_BACK操作是非线程安全的。
//  加mutex锁，5min20s， 95.29%
//  开启-march=native， 95.29%  4min45s
//  k = 10  96.68%
void MnistKNN()
{
    MNistData::loadMnist();
    int isNum = 0;
    std::mutex mutex;
    for (int i = 0; i < 10000; i++)
    {
        auto xi = ORIGDATA::mat_L_Te_2D->data_f.row(i);
        std::vector<Compare> vecCom;
        vecCom.reserve(55000);

        #pragma omp parallel for
        for (int j = 0; j < 55000; j++)
        {
            auto xj = ORIGDATA::mat_L_Tr_2D->data_f.row(j);
            Compare comp { distance(xi, xj), j, colMax(ORIGDATA::mat_R_Tr_2D->data_b.row(j)) };
            mutex.lock();
            vecCom.push_back(comp);
            mutex.unlock();
        }
        std::sort(vecCom.begin(), vecCom.end(),[]( Compare& c_1, Compare& c_2 ){ return c_1.distance < c_2.distance; });

        std::map<int,int> maxNumMap;
        for( int j = 0; j < 10; j++ )
        {
            maxNumMap[ vecCom[j].value ]++;
        }
        
        int maxNum = 0;
        int currentNum = 0;
        for( auto j = maxNumMap.begin(); j != maxNumMap.end(); j++ )
        {
            if( (*j).second >= maxNum )
            {
                maxNum = (*j).second;
                currentNum = (*j).first;
            }
        }

        if( currentNum == colMax(ORIGDATA::mat_R_Te_2D->data_b.row(i)) )
        {
            isNum ++;
        }
    }
    std::cout << isNum << std::endl;
}
} // namespace MnistKNN
} // namespace RNN