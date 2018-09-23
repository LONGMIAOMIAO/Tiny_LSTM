#pragma once
#include <iostream>
#include <eigen-eigen-b3f3d4950030/Eigen/Core>
#include "../LAYER/Ful.hpp"
#include "../DATA/OrigData.hpp"
namespace RNN
{
namespace TEST
{
void Ful_T_1()
{
    Ful* f_1 = new Ful(3,3);

    std::cout << f_1->w << std::endl;
}
}
} // namespace RNN