#pragma once
#include <eigen-eigen-b3f3d4950030/Eigen/Core>
namespace RNN
{
using mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

int colMax(const mat &m)
{
    int max_r_1, max_c_1;
    m.maxCoeff(&max_r_1, &max_c_1);
    return max_c_1;
}

const mat tanh_F(const mat &inputMatrix)
{
    return (inputMatrix.array().exp() - (-1.0 * inputMatrix).array().exp()) / (inputMatrix.array().exp() + (-1.0 * inputMatrix).array().exp());
}

const mat tanh_B(const mat &inputMatrix)
{
    return 1 - inputMatrix.array() * inputMatrix.array();
}

} // namespace RNN