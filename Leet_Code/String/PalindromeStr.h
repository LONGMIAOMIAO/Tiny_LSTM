#pragma once
#include <iostream>
#include <string>
#include <vector>
namespace String
{
namespace Palindrome
{
bool isPalindrome(std::string s)
{
    std::vector<char> vec;
    for (char c : s)
    {
        if ( c < 91 && c > 64 )
        {
            vec.push_back(c);
        }
        else if ( c < 123 && c >96 )
        {
            vec.push_back( c-32 );
        }
    }

    int vecS = vec.size();
    int vecSize = vec.size() / 2;

    for ( int i = 0; i < vecSize; i++ )
    {
        std::cout << vec[i] << vec[vecS - i - 1] << std::endl;
        if ( vec[i] != vec[vecS - i - 1] )
            return false;
    }
    return true;
}

void Test()
{
    isPalindrome("s. TsTTsTs");
}

}
} // namespace String