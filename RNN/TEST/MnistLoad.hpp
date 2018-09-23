#pragma once
#include "../LOAD/MnistData.hpp"
namespace RNN
{
namespace TEST
{
void MnistLoad()
{
    MNistData::loadMnist();

    int lineNum = 30000;
    for (int s = lineNum; s < lineNum + 4; s++)
    {
        for (int i = 0; i < 28; i++)
        {
            for (int j = 0; j < 28; j++)
            {
                std::cout << static_cast<int>(ORIGDATA::mat_L_Tr_2D->data_f(s, i * 28 + j) / 255 * 8);
            }
            std::cout << std::endl;
        }
        std::cout << ORIGDATA::mat_R_Tr_2D->data_b.row(s) << std::endl;
    }

    int lineNum_T = 3000;
    for (int s = lineNum_T; s < lineNum_T + 4; s++)
    {
        for (int i = 0; i < 28; i++)
        {
            for (int j = 0; j < 28; j++)
            {
                std::cout << static_cast<int>(ORIGDATA::mat_L_Te_2D->data_f(s, i * 28 + j) / 255 * 8);
            }
            std::cout << std::endl;
        }
        std::cout << ORIGDATA::mat_R_Te_2D->data_b.row(s) << std::endl;
    }
}
} // namespace TEST
} // namespace RNN