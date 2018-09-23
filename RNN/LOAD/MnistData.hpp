#pragma once
#include <fstream>  
#include <sstream>  
#include <iostream>
#include "../DATA/OrigData.hpp"

namespace RNN
{
namespace MNistData
{
    void loadMnist()
    {
        ORIGDATA::mat_L_Tr_2D = new MidData;
        ORIGDATA::mat_L_Tr_2D->data_f.resize(55000, 784);
        ORIGDATA::mat_R_Tr_2D = new MidData;
        ORIGDATA::mat_R_Tr_2D->data_b.resize(55000,  10);

        ORIGDATA::mat_L_Te_2D = new MidData;
        ORIGDATA::mat_L_Te_2D->data_f.resize(10000, 784);
        ORIGDATA::mat_R_Te_2D = new MidData;
        ORIGDATA::mat_R_Te_2D->data_b.resize(10000,  10);

        std::ifstream L_Tr_File( "../../DATA/MNIST/train.csv", std::ios::in );
		std::string L_Tr_Str;
		int L_Tr_Num = 0;
		while (std::getline(L_Tr_File, L_Tr_Str))
		{
			std::stringstream ss(L_Tr_Str);
			std::string str;
			int inNum = 0;
			while (std::getline(ss, str, ','))
			{
				ORIGDATA::mat_L_Tr_2D->data_f(L_Tr_Num,inNum) = atoi( str.c_str() );
				inNum++;
			}
			L_Tr_Num++;
		}

        std::ifstream R_Tr_File( "../../DATA/MNIST/trainL.csv", std::ios::in );
		std::string R_Tr_Str;
		int R_Tr_Num = 0;
		while (std::getline(R_Tr_File, R_Tr_Str))
		{
			std::stringstream ss(R_Tr_Str);
			std::string str;
			int inNum = 0;
			while (std::getline(ss, str, ','))
			{
				ORIGDATA::mat_R_Tr_2D->data_b(R_Tr_Num,inNum) = atoi( str.c_str() );
				inNum++;
			}
			R_Tr_Num++;
		}

        std::ifstream L_Te_File( "../../DATA/MNIST/test.csv", std::ios::in );
		std::string L_Te_Str;
		int L_Te_Num = 0;
		while (std::getline(L_Te_File, L_Te_Str))
		{
			std::stringstream ss(L_Te_Str);
			std::string str;
			int inNum = 0;
			while (std::getline(ss, str, ','))
			{
				ORIGDATA::mat_L_Te_2D->data_f(L_Te_Num,inNum) = atoi( str.c_str() );
				inNum++;
			}
			L_Te_Num++;
		}

        std::ifstream R_Te_File( "../../DATA/MNIST/testL.csv", std::ios::in );
		std::string R_Te_Str;
		int R_Te_Num = 0;
		while (std::getline(R_Te_File, R_Te_Str))
		{
			std::stringstream ss(R_Te_Str);
			std::string str;
			int inNum = 0;
			while (std::getline(ss, str, ','))
			{
				ORIGDATA::mat_R_Te_2D->data_b(R_Te_Num,inNum) = atoi( str.c_str() );
				inNum++;
			}
			R_Te_Num++;
		}

        ORIGDATA::mat_L_Tr_2D->data_f = ORIGDATA::mat_L_Tr_2D->data_f / 255;
        ORIGDATA::mat_L_Te_2D->data_f = ORIGDATA::mat_L_Te_2D->data_f / 255;
    }
}
} // namespace RNN