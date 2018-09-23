#pragma once
#include "../DATA/MidData.hpp"
namespace RNN
{
    class Layer
    {
    public:
        Layer()
        {
            left = new MidData;
            right = new MidData;
        }

        virtual void calF() = 0;
        virtual void calB() = 0;
        virtual void upW() = 0;

        MidData* left;
        MidData* right;
    };




}