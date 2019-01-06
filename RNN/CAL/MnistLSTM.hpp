#pragma once
#include <iostream>
#include "../LAYER_RNN/Layer_LSTM.hpp"
#include "../LOAD/MnistData.hpp"

namespace RNN
{
namespace MnistLSTM
{
//  g++ Source.cpp -O3
void MnistLSTM()
{
    MNistData::loadMnist();
    std::vector<std::vector<MidData *>> bottom_x_vec;
    for (int i = 0; i < 55000; i++)
    {
        mat s = ORIGDATA::mat_L_Tr_2D->data_f.row(i);
        std::vector<MidData *> vec_in;
        for (int j = 0; j < 28; j++)
        {
            MidData *mid = new MidData;
            mid->data_f = s.block(0, j * 28, 1, 28);
            vec_in.push_back(mid);
        }
        bottom_x_vec.push_back(vec_in);
    }

    std::vector<std::vector<MidData *>> bottom_x_vec_T;
    for (int i = 0; i < 10000; i++)
    {
        mat s = ORIGDATA::mat_L_Te_2D->data_f.row(i);
        std::vector<MidData *> vec_in;
        for (int j = 0; j < 28; j++)
        {
            MidData *mid = new MidData;
            mid->data_f = s.block(0, j * 28, 1, 28);
            vec_in.push_back(mid);
        }
        bottom_x_vec_T.push_back(vec_in);
    }

    //  ================================================================================================

    Net_LSTM net_LSTM_T(28, 28, 128);
    mat classify(128, 10);
    classify.setRandom();
    MidData out;

    for (int p = 0; p < 2; p++)
    {
        for (int i = 0; i < 55000; i++)
        {
            net_LSTM_T.setInitial(i, bottom_x_vec);

            net_LSTM_T.calForward();

            out.data_f = (net_LSTM_T.cell_LSTM_Vec.back().h_L4->out->data_f * classify).array().exp().matrix() / (net_LSTM_T.cell_LSTM_Vec.back().h_L4->out->data_f * classify).array().exp().sum();

            out.data_b = out.data_f - ORIGDATA::mat_R_Tr_2D->data_b.row(i);

            net_LSTM_T.cell_LSTM_Vec.back().h_L4->out->data_b = out.data_b * classify.transpose();

            net_LSTM_T.calBackward();

            net_LSTM_T.upW();

            classify = classify - 0.05 * net_LSTM_T.cell_LSTM_Vec.back().h_L4->out->data_f.transpose() * out.data_b;

            //std::cout << i << ":::" << out.data_f << std::endl;
        }
    }

    int totalNum = 0;
    for( int i = 0; i < 10000; i++ )
    {
        net_LSTM_T.setInitial(i, bottom_x_vec_T);

        net_LSTM_T.calForward();

        out.data_f = (net_LSTM_T.cell_LSTM_Vec.back().h_L4->out->data_f * classify).array().exp().matrix() / (net_LSTM_T.cell_LSTM_Vec.back().h_L4->out->data_f * classify).array().exp().sum();

        if( colMax(out.data_f) == colMax( ORIGDATA::mat_R_Te_2D->data_b.row(i) ) )
        {
            totalNum++;
        }
    }
    std::cout << totalNum << std::endl;
}
} // namespace MnistLSTM
} // namespace RNN