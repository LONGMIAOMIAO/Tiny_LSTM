#pragma once
#include "../LOAD/MnistData.hpp"
#include "../LAYER/Ful.hpp"

namespace RNN
{
namespace MnistPerceptron
{
void MnistPerceptron()
{
    MNistData::loadMnist();

    Layer *layer_Ful = new Ful(784, 10);
    for (int j = 0; j < 2; j++)
    {
        for (int i = 0; i < 55000; i++)
        {
            layer_Ful->left->data_f = ORIGDATA::mat_L_Tr_2D->data_f.row(i);

            layer_Ful->calF();

            layer_Ful->right->data_b = layer_Ful->right->data_f - ORIGDATA::mat_R_Tr_2D->data_b.row(i);

            layer_Ful->calB();
            layer_Ful->upW();
        }
    }

    int rightNum = 0;
    for (int i = 0; i < 10000; i++)
    {
        layer_Ful->left->data_f = ORIGDATA::mat_L_Te_2D->data_f.row(i);
        layer_Ful->calF();

        if(colMax(layer_Ful->right->data_f) == colMax(ORIGDATA::mat_R_Te_2D->data_b.row(i)))
        {
            rightNum ++;
        }
    }
    std::cout << rightNum << std::endl;    
}
} // namespace MnistCal
} // namespace RNN