#pragma once
#include <iostream>
#include <vector>
#include <unordered_map>
namespace Vector
{
namespace Array
{
int singleNum( std::vector<int>& vec )
{
    std::unordered_map<int,int> hash;
    for ( auto num : vec )
    {
        hash[num]++;
    }
    for ( auto num : vec )
    {
        if ( hash[num] == 1 )
            return num;
    }
}

void Test()
{
    std::vector<int> vec{3,8,9,2,8,3,2};
    std::cout << singleNum(vec) << std::endl;
}

}
}