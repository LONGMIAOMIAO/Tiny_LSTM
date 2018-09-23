#pragma once
#include "Layer.hpp"
namespace RNN
{
    class Ful : public Layer
    {
    public:
        Ful( int row, int col )
        {
            w.resize( row, col );
            w.setRandom();
        }
        void calF() override
        {
            right->data_f = left->data_f * w;
        }
        virtual void calB() override
        {
            left->data_b = right->data_b * w.transpose();
        }
        virtual void upW() override
        {
            w = w - 0.005 * left->data_f.transpose() * right->data_b;
        }

        mat w;
    };
}