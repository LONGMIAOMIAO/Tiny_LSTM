#pragma once
#include "../LAYER/Layer.hpp"
//#include "../DATA/MidData.hpp"
namespace RNN
{
class Layer_RNN //: public Layer
{
  public:
    Layer_RNN(int l_r_r, int l_r_c, int b_t_r, int b_t_c, mat &w_l_r, mat &w_b_t) : left(new MidData), out(new MidData), bottom(new MidData), w_l_r(w_l_r), w_b_t(w_b_t)
    {
        // w_l_r.resize( l_r_r, l_r_c );
        // w_l_r.setRandom();

        // w_b_t.resize( b_t_r, b_t_c );
        // w_b_t.setRandom();
    }

    void calForward()
    {
        out->data_f = left->data_f * w_l_r + bottom->data_f * w_b_t;
    }

    void calBackward()
    {
        left->data_b = out->data_b * w_l_r.transpose();
        bottom->data_b = out->data_b * w_b_t.transpose();
    }

    void upW()
    {
        w_l_r = w_l_r - 0.01 * (left->data_f.transpose()) * (out->data_b);
        w_b_t = w_b_t - 0.01 * (bottom->data_f.transpose()) * (out->data_b);
    }

    mat &w_l_r;
    mat &w_b_t;

    MidData *left;
    MidData *out;
    MidData *bottom;
};

class Layer_Net_RNN
{
  public:
    Layer_Net_RNN(int layer_Num, int l_r_r, int l_r_c, int b_t_r, int b_t_c, std::vector<MidData *> &vec_Feed_Data) : vec_Feed_Data(vec_Feed_Data)
    {
        w_l_r.resize(l_r_r, l_r_c);
        w_l_r.setRandom();
        //w_l_r.setConstant(1.0);

        w_b_t.resize(b_t_r, b_t_c);
        w_b_t.setRandom();
        //w_b_t.setConstant(1.0);

        for (int i = 0; i < layer_Num; i++)
        {
            vec_Layer_RNN.emplace_back(l_r_r, l_r_c, b_t_r, b_t_c, w_l_r, w_b_t);
            //  std::cout << vec_Layer_RNN[i].w_b_t.rows() << std::endl;
        }

        //  create internalData
        start = new MidData;
        start->data_f.resize(1, 4);
        //  start->data_f.setRandom();
        start->data_f.setConstant(0.0);
        vec_Layer_RNN.front().left = start;
        for (int i = 0; i < layer_Num - 1; i++)
        {
            MidData *mid = new MidData;
            vec_Layer_RNN[i].out = mid;
            vec_Layer_RNN[i + 1].left = mid;

            vec_Mid_Data.push_back(mid);
        }
        end = new MidData;
        end->data_b.resize(1, 4);
        end->data_b.setConstant(1.0);
        vec_Layer_RNN.back().out = end;
        //  cout InternalData
        for (int i = 0; i < layer_Num - 1; i++)
        {
            std::cout << i << std::endl;
            std::cout << vec_Layer_RNN[i].left << std::endl;
            std::cout << vec_Layer_RNN[i].out << std::endl;
        }

        for (int i = 0; i < layer_Num; i++)
        {
            vec_Layer_RNN[i].bottom = vec_Feed_Data[i];
        }

        std::cout << vec_Layer_RNN.size() << std::endl;
        std::cout << vec_Mid_Data.size() << std::endl;
    }

    void calForward()
    {
        for (auto i = vec_Layer_RNN.begin(); i != vec_Layer_RNN.end(); i++)
        {
            (*i).calForward();
            std::cout << "w_l_r_rows:" << (*i).w_l_r.rows() << std::endl;
            std::cout << "w_l_r_cols:" << (*i).w_l_r.cols() << std::endl;
            std::cout << (*i).left->data_f.rows() << std::endl;
            std::cout << (*i).out->data_f.cols() << std::endl;
        }
    }

    void calBackward()
    {
        for (auto i = vec_Layer_RNN.rbegin(); i != vec_Layer_RNN.rend(); i++)
        {
            (*i).calBackward();
        }
    }

    void upW()
    {
        for (auto i = vec_Layer_RNN.begin(); i != vec_Layer_RNN.end(); i++)
        {
            (*i).upW();
            
            std::cout << "l_r_w :" << std::endl
                      << (*i).w_l_r << std::endl;

            std::cout << "b_t_w :" << std::endl
                      << (*i).w_b_t << std::endl;

            std::cout << "*****************************" << std::endl;
        }
    }

    MidData *start;
    MidData *end;

    mat w_l_r;
    mat w_b_t;

    std::vector<MidData *> &vec_Feed_Data;

    std::vector<MidData *> vec_Mid_Data;
    std::vector<Layer_RNN> vec_Layer_RNN;
};
} // namespace RNN