#pragma once
#include <iostream>
#include "../LAYER_RNN/Layer_RNN.hpp"

namespace RNN
{
namespace MnistRNN
{

void RNN_T_1()
{
    mat m1(1, 4);
    m1.setConstant(0);
    m1(0, 0) = 1;

    mat m2(1, 4);
    m2.setConstant(0);
    m2(0, 1) = 1;

    mat m3(1, 4);
    m3.setConstant(0);
    m3(0, 2) = 1;

    mat m4(1, 4);
    m4.setConstant(0);
    m4(0, 3) = 1;

    std::vector<MidData *> feedData{new MidData, new MidData, new MidData, new MidData};
    std::for_each(feedData.begin(), feedData.end(), [](MidData *&mid) {
        mid->data_f.resize(1, 4);
        //  mid->data_f.setRandom();
        mid->data_f.setConstant(0.0);
    });
    Layer_Net_RNN rnn_T(4, 4, 4, 4, 4, feedData);

    for (int i = 0; i < 200; i++)
    {
        if (i % 2 == 0)
        {
            feedData[0]->data_f = m1;
            feedData[1]->data_f = m2;
            feedData[2]->data_f = m3;
            feedData[3]->data_f = m4;
        }
        else if (i % 2 == 1)
        {
            feedData[0]->data_f = m4;
            feedData[1]->data_f = m3;
            feedData[2]->data_f = m2;
            feedData[3]->data_f = m1;
        }

        rnn_T.calForward();

        if (i % 2 == 0)
        {
            rnn_T.end->data_b = rnn_T.end->data_f - m1;
        }
        else if (i % 2 == 1)
        {
            rnn_T.end->data_b = rnn_T.end->data_f - m4;
        }

        rnn_T.calBackward();
        rnn_T.upW();

        if (i == 198)
        {
            std::cout << "End 0 IS:" << std::endl
                      << rnn_T.vec_Layer_RNN[0].out->data_f << std::endl;
            std::cout << "End 1 IS:" << std::endl
                      << rnn_T.vec_Layer_RNN[1].out->data_f << std::endl;
            std::cout << "End 2 IS:" << std::endl
                      << rnn_T.vec_Layer_RNN[2].out->data_f << std::endl;
            std::cout << "End 3 IS:" << std::endl
                      << rnn_T.end->data_f << std::endl;
        }
        if (i == 199)
        {
            std::cout << "End 0 IS:" << std::endl
                      << rnn_T.vec_Layer_RNN[0].out->data_f << std::endl;
            std::cout << "End 1 IS:" << std::endl
                      << rnn_T.vec_Layer_RNN[1].out->data_f << std::endl;
            std::cout << "End 2 IS:" << std::endl
                      << rnn_T.vec_Layer_RNN[2].out->data_f << std::endl;
            std::cout << "End 3 IS:" << std::endl
                      << rnn_T.end->data_f << std::endl;
        }
    }

    // feedData[0]->data_f = m1;
    // feedData[1]->data_f = m2;
    // feedData[2]->data_f = m3;
    // feedData[3]->data_f = m4;
    // rnn_T.calForward();

    // std::cout << "End 0 IS:" << std::endl
    //           << rnn_T.vec_Layer_RNN[0].out->data_f << std::endl;
    // std::cout << "End 1 IS:" << std::endl
    //           << rnn_T.vec_Layer_RNN[1].out->data_f << std::endl;
    // std::cout << "End 2 IS:" << std::endl
    //           << rnn_T.vec_Layer_RNN[2].out->data_f << std::endl;
    // std::cout << "End 3 IS:" << std::endl
    //           << rnn_T.end->data_f << std::endl;
}


void MnistRNN()
{
    std::cout << "Good Better Best!" << std::endl;

    std::vector<MidData *> feedData{new MidData, new MidData, new MidData, new MidData};
    std::for_each(feedData.begin(), feedData.end(), [](MidData *&mid) {
        mid->data_f.resize(1, 4);
        //mid->data_f.setRandom();
        mid->data_f.setConstant(1.0);
    });

    Layer_Net_RNN rnn_T(4, 4, 4, 4, 4, feedData);

    std::cout << "=================================================" << std::endl;
    rnn_T.calForward();

    for (auto i = rnn_T.vec_Layer_RNN.begin(); i != rnn_T.vec_Layer_RNN.end(); i++)
    {
        std::cout << "Left Is :" << (*i).left->data_f << std::endl;
        std::cout << "Right Is :" << (*i).out->data_f << std::endl;
    }

    std::cout << "=================================================" << std::endl;
    rnn_T.calBackward();

    for (auto i = rnn_T.vec_Layer_RNN.rbegin(); i != rnn_T.vec_Layer_RNN.rend(); i++)
    {
        std::cout << "Right Back Is :" << (*i).out->data_b << std::endl;
        std::cout << "Left Back Is :" << (*i).left->data_b << std::endl;
    }

    std::cout << "=================================================" << std::endl;
    for (auto i = rnn_T.vec_Layer_RNN.rbegin(); i != rnn_T.vec_Layer_RNN.rend(); i++)
    {
        std::cout << "Right Bottom Is :" << (*i).bottom->data_b << std::endl;
    }

    std::cout << "*****************************" << std::endl;

    rnn_T.upW();
    for (auto i = rnn_T.vec_Layer_RNN.begin(); i != rnn_T.vec_Layer_RNN.end(); i++)
    {
        // std::cout << "l_r_w :" << std::endl << (*i).w_l_r << std::endl;

        // std::cout << "b_t_w :" << std::endl << (*i).w_b_t << std::endl;

        // std::cout << "*****************************" << std::endl;
    }
}
} // namespace MnistRNN
} // namespace RNN