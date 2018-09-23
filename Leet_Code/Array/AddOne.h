#pragma once
#include <vector>
#include <iostream>

namespace Vector
{
namespace AddOne
{
std::vector<int> plusOne(std::vector<int> &digits)
{
    for( auto i = digits.rbegin(); i != digits.rend(); i++ )
    {
        int num = *i;
        if( i == digits.rbegin() )
        {
            (*i) = (num+1)%10;
        }
        else
        {
            (*i) = num%10;
        }

        *(i+1) += (num+1)/10;
    }
    return digits;
}

void Test()
{
    std::vector<int> vec{ 1,9,9,9 };
    auto vec_Back = plusOne(vec);

    for(auto i : vec_Back)
    {
        std::cout << i << std::endl;
    }
}
} // namespace AddOne
} // namespace Vector