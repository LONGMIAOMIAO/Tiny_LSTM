#pragma once
#include <iostream>
#include "../LAYER_RNN/Layer_RNN.hpp"

namespace RNN
{
namespace MnistRNN
{
void MnistRNN()
{
    std::cout << "Good Better Best!" << std::endl;

    std::vector<MidData*> feedData{ new MidData, new MidData, new MidData, new MidData, new MidData };
    std::for_each( feedData.begin(), feedData.end(), []( MidData*& mid )
    { 
        mid->data_f.resize(1,3);
        //mid->data_f.setRandom();
        mid->data_f.setConstant(1.0);
    } );
    
    Layer_Net_RNN rnn_T( 5, 2, 2, 3, 2, feedData );

    std::cout << "=================================================" << std::endl;
    rnn_T.calForward();

    for( auto i = rnn_T.vec_Layer_RNN.begin(); i != rnn_T.vec_Layer_RNN.end(); i++ )
    {
        std::cout << "Left Is :" << (*i).left->data_f << std::endl;
        std::cout << "Right Is :" << (*i).out->data_f << std::endl;
    }

    std::cout << "=================================================" << std::endl;
    rnn_T.calBackward();
    
    for( auto i = rnn_T.vec_Layer_RNN.rbegin(); i != rnn_T.vec_Layer_RNN.rend(); i++ )
    {
        std::cout << "Right Back Is :" << (*i).out->data_b << std::endl;
        std::cout << "Left Back Is :" << (*i).left->data_b << std::endl;
    }

    std::cout << "=================================================" << std::endl;
    for ( auto i = rnn_T.vec_Layer_RNN.rbegin(); i != rnn_T.vec_Layer_RNN.rend(); i++ )
    {
        std::cout << "Right Bottom Is :" << (*i).bottom->data_b << std::endl;
    }

    std::cout << "*****************************" << std::endl;

    rnn_T.upW();
    for ( auto i = rnn_T.vec_Layer_RNN.begin(); i != rnn_T.vec_Layer_RNN.end(); i++ )
    {
        // std::cout << "l_r_w :" << std::endl << (*i).w_l_r << std::endl;

        // std::cout << "b_t_w :" << std::endl << (*i).w_b_t << std::endl;

        // std::cout << "*****************************" << std::endl;        
    }
}
} // namespace MnistRNN
} // namespace RNN