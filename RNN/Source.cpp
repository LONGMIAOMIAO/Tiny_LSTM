#include "TEST/Ful_T.hpp"
#include "LOAD/MnistData.hpp"
#include "TEST/MnistLoad.hpp"
#include "CAL/MnistPerceptron.hpp"
#include "CAL/MnistKNN.hpp"
#include "CAL/MnistRNN.hpp"
#include "LAYER_RNN/Layer_LSTM.hpp"
int main()
{
    //RNN::TEST::Ful_T_1();
    //RNN::MNistData::loadMnist();
    //RNN::TEST::MnistLoad();
    //RNN::MnistPerceptron::MnistPerceptron();
    //RNN::MnistKNN::MnistKNN();
    //RNN::MnistRNN::MnistRNN();
    RNN::MnistRNN::RNN_T_1();
}