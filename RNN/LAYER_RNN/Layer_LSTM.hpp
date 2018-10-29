#pragma once
#include "../LAYER/Layer.hpp"

namespace RNN
{

class Core_Update_LSTM_L1
{
public:
  Core_Update_LSTM_L1(mat &w_l_o, mat &w_b_o) : w_l_o(w_l_o), w_b_o(w_b_o), left_h(new MidData), bottom_x(new MidData), out(new MidData) {}

  void calForward()
  {
    out->data_f = sigmoid_F(left_h->data_f * w_l_o + bottom_x->data_f * w_b_o);
  }

  void calBackward()
  {
    //  2.
    left_h->data_b = (out->data_b.array() * sigmoid_B(out->data_f).array()).matrix() * w_l_o.transpose() + left_h->data_b;
    bottom_x->data_b = (out->data_b.array() * sigmoid_B(out->data_f).array()).matrix() * w_b_o.transpose();
  }

  void upW()
  {
    w_l_o = w_l_o - 0.5 * (left_h->data_f.transpose()) * (out->data_b);
    w_b_o = w_b_o - 0.5 * (bottom_x->data_f.transpose()) * (out->data_b);
  }

  mat &w_l_o;
  mat &w_b_o;

  MidData *left_h;
  MidData *bottom_x;

  MidData *out;
};

class Core_Compute_LSTM_L1
{
public:
  Core_Compute_LSTM_L1(mat &w_l_o, mat &w_b_o) : w_l_o(w_l_o), w_b_o(w_b_o), left_h(new MidData), bottom_x(new MidData), out(new MidData) {}

  void calForward()
  {
    out->data_f = tanh_F(left_h->data_f * w_l_o + bottom_x->data_f * w_b_o);
  }

  void calBackward()
  {
    //  3.
    left_h->data_b = (out->data_b.array() * tanh_B(out->data_f).array()).matrix() * w_l_o.transpose() + left_h->data_b;
    bottom_x->data_b = (out->data_b.array() * tanh_B(out->data_f).array()).matrix() * w_b_o.transpose();
  }

  void upW()
  {
    w_l_o = w_l_o - 0.5 * (left_h->data_f.transpose()) * (out->data_b);
    w_b_o = w_b_o - 0.5 * (bottom_x->data_f.transpose()) * (out->data_b);
  }

  mat &w_l_o;
  mat &w_b_o;

  MidData *left_h;
  MidData *bottom_x;

  MidData *out;
};

class Core_Forget_LSTM_L1
{
public:
  Core_Forget_LSTM_L1(mat &w_l_o, mat &w_b_o) : w_l_o(w_l_o), w_b_o(w_b_o), left_h(new MidData), bottom_x(new MidData), out(new MidData) {}

  void calForward()
  {
    out->data_f = sigmoid_F(left_h->data_f * w_l_o + bottom_x->data_f * w_b_o);
  }

  void calBackward()
  {
    //  4.
    left_h->data_b = (out->data_b.array() * sigmoid_B(out->data_f).array()).matrix() * w_l_o.transpose() + left_h->data_b;
    bottom_x->data_b = (out->data_b.array() * sigmoid_B(out->data_f).array()).matrix() * w_b_o.transpose();
  }

  void upW()
  {
    w_l_o = w_l_o - 0.5 * (left_h->data_f.transpose()) * (out->data_b);
    w_b_o = w_b_o - 0.5 * (bottom_x->data_f.transpose()) * (out->data_b);
  }

  mat &w_l_o;
  mat &w_b_o;

  MidData *left_h;
  MidData *bottom_x;

  MidData *out;
};

class Core_c_LSTM_L2
{
public:
  void calForward()
  {
    out->data_f = left_Update->data_f.array() * left_Compute->data_f.array() + left_c->data_f.array() * left_Forget->data_f.array();
  }
  void calBackward()
  {
    left_Update->data_b = out->data_b.array() * left_Compute->data_f.array();
    left_Compute->data_b = out->data_b.array() * left_Update->data_f.array();

    left_c->data_b = out->data_b.array() * left_Forget->data_f.array();
    left_Forget->data_b = out->data_b.array() * left_c->data_f.array();
  }

  MidData *left_Update;
  MidData *left_Compute;

  MidData *left_Forget;
  MidData *left_c;

  MidData *out;
};

class Core_c_LSTM_L3
{
public:
  void calForward()
  {
    out->data_f = tanh_F(left->data_f);
  }

  //  attation, should add the left( c_out ) before
  void calBackward()
  {
    left->data_b = (out->data_b.array() * tanh_B(out->data_f).array()).matrix() + left->data_b;
  }

  MidData *left;
  MidData *out;
};

class Core_Output_LSTM_L1
{
public:
  Core_Output_LSTM_L1(mat &w_l_o, mat &w_b_o) : w_l_o(w_l_o), w_b_o(w_b_o), left_h(new MidData), bottom_x(new MidData), out(new MidData) {}

  void calForward()
  {
    out->data_f = sigmoid_F(left_h->data_f * w_l_o + bottom_x->data_f * w_b_o);
  }

  void calBackward()
  {
    // 1_back
    left_h->data_b = (out->data_b.array() * sigmoid_B(out->data_f).array()).matrix() * w_l_o.transpose();
    bottom_x->data_b = (out->data_b.array() * sigmoid_B(out->data_f).array()).matrix() * w_b_o.transpose();
  }

  void upW()
  {
    w_l_o = w_l_o - 0.5 * (left_h->data_f.transpose()) * (out->data_b);
    w_b_o = w_b_o - 0.5 * (bottom_x->data_f.transpose()) * (out->data_b);
  } 

  mat &w_l_o;
  mat &w_b_o;

  MidData *left_h;
  MidData *bottom_x;

  MidData *out;
};

class Core_h_LSTM_L4
{
public:
  void calForward()
  {
    out->data_f = left_Output->data_f.array() * left_c_L3->data_f.array();
  }

  void calBackward()
  {
    left_Output->data_b = out->data_b.array() * left_c_L3->data_f.array();
    left_c_L3->data_b = out->data_b.array() * left_Output->data_f.array();
  }

  MidData *left_Output;
  MidData *left_c_L3;

  MidData *out;
};

class Cell_LSTM
{
public:
  Cell_LSTM(
      //  Core_Update_LSTM_L1
      mat &w_l_o_update, mat &w_b_o_update,
      //  Core_Compute_LSTM_L1
      mat &w_l_o_compute, mat &w_b_o_compute,
      //  Core_Forget_LSTM_L1
      mat &w_l_o_forget, mat &w_b_o_forget,
      //  Core_Output_LSTM_L1
      mat &w_l_o_output, mat &w_b_o_output) : update_L1(new Core_Update_LSTM_L1(w_l_o_update, w_b_o_update)),
                                              compute_L1(new Core_Compute_LSTM_L1(w_l_o_compute, w_b_o_compute)),
                                              forget_L1(new Core_Forget_LSTM_L1(w_l_o_forget, w_b_o_forget)),
                                              output_L1(new Core_Output_LSTM_L1(w_l_o_output, w_b_o_output)),

                                              c_L2(new Core_c_LSTM_L2),

                                              c_L3(new Core_c_LSTM_L3),

                                              h_L4(new Core_h_LSTM_L4)
  {
    MidData *o_update = new MidData;
    MidData *o_compute = new MidData;
    MidData *o_forget = new MidData;
    MidData *o_output = new MidData;

    update_L1->out = o_update;
    compute_L1->out = o_compute;
    forget_L1->out = o_forget;
    output_L1->out = o_output;

    c_L2->left_Update = o_update;
    c_L2->left_Compute = o_compute;
    c_L2->left_Forget = o_forget;

    MidData *o_c_L2 = new MidData;
    c_L2->out = o_c_L2;
    c_L3->left = o_c_L2;

    MidData *o_c_L3 = new MidData;
    c_L3->out = o_c_L3;
    h_L4->left_c_L3 = o_c_L3;
    h_L4->left_Output = o_output;
  }

  void calForward()
  {
    update_L1->calForward();

    compute_L1->calForward();

    forget_L1->calForward();

    c_L2->calForward();

    c_L3->calForward();

    output_L1->calForward();

    h_L4->calForward();

    // std::cout << "output_L1" << h_L4->left_c_L3->data_f << std::endl;
    // std::cout << "output_L1_1" << h_L4->left_Output->data_f << std::endl;
    // std::cout << "================" << std::endl;
  }

  void calBackward()
  {
    h_L4->calBackward();

    output_L1->calBackward();

    c_L3->calBackward();
    c_L2->calBackward();

    forget_L1->calBackward();
    compute_L1->calBackward();
    update_L1->calBackward();

    //std::cout << update_L1->left_h->data_b << std::endl;
  }

  void upW()
  {
    update_L1->upW();
    compute_L1->upW();
    forget_L1->upW();
    output_L1->upW();
  }

  Core_Update_LSTM_L1 *update_L1;
  Core_Compute_LSTM_L1 *compute_L1;
  Core_Forget_LSTM_L1 *forget_L1;
  Core_Output_LSTM_L1 *output_L1;

  Core_c_LSTM_L2 *c_L2;

  Core_c_LSTM_L3 *c_L3;

  Core_h_LSTM_L4 *h_L4;
};

class Net_LSTM
{
public:
  Net_LSTM(
          int cell_Size,
          //  in_x_Size
          int x_Size,
          //  out Size
          int h_Size
          )
  {
    w_l_o_update.resize( h_Size, h_Size );
    w_b_o_update.resize( x_Size, h_Size );

    w_l_o_compute.resize( h_Size, h_Size );
    w_b_o_compute.resize( x_Size, h_Size );

    w_l_o_forget.resize( h_Size, h_Size );
    w_b_o_forget.resize( x_Size, h_Size );

    w_l_o_output.resize( h_Size, h_Size );
    w_b_o_output.resize( x_Size, h_Size );

    w_l_o_update.setRandom();
    w_b_o_update.setRandom();

    w_l_o_compute.setRandom();
    w_b_o_compute.setRandom();

    w_l_o_forget.setRandom();
    w_b_o_forget.setRandom();

    w_l_o_output.setRandom();
    w_b_o_output.setRandom();


    std::cout << "good_1" << std::endl;


    for (int i = 0; i < cell_Size; i++)
    {
      Cell_LSTM cell(
          //  Core_Update_LSTM_L1
          w_l_o_update, w_b_o_update,
          //  Core_Compute_LSTM_L1
          w_l_o_compute, w_b_o_compute,
          //  Core_Forget_LSTM_L1
          w_l_o_forget, w_b_o_forget,
          //  Core_Output_LSTM_L1
          w_l_o_output, w_b_o_output);
      cell_LSTM_Vec.push_back(cell);
    }

    std::cout << "good_2" << std::endl;    

    for( int i = 0; i < cell_LSTM_Vec.size()-1; i++ )
    {
      MidData* mid = new MidData;
      
      cell_LSTM_Vec[i].h_L4->out = mid;

      cell_LSTM_Vec[i+1].update_L1->left_h = mid;
      cell_LSTM_Vec[i+1].compute_L1->left_h = mid;
      cell_LSTM_Vec[i+1].forget_L1->left_h = mid;
      cell_LSTM_Vec[i+1].output_L1->left_h = mid;
    }
    for( int i = 0; i < cell_LSTM_Vec.size()-1; i++ )
    {
      cell_LSTM_Vec[i+1].c_L2->left_c = cell_LSTM_Vec[i].c_L2->out;
    }

    MidData* start_h = new MidData;
    cell_LSTM_Vec.front().update_L1->left_h = start_h;
    cell_LSTM_Vec.front().compute_L1->left_h = start_h;
    cell_LSTM_Vec.front().forget_L1->left_h = start_h;
    cell_LSTM_Vec.front().output_L1->left_h = start_h;
    start_h->data_f.resize(1, h_Size);
    start_h->data_f.setConstant(0);

    MidData* start_c = new MidData;
    cell_LSTM_Vec.front().c_L2->left_c = start_c;
    start_c->data_f.resize(1, h_Size);
    start_c->data_f.setConstant(0);

    //====================================================================== new delete
    
    cell_LSTM_Vec.back().c_L2->out->data_b.resize( 1, h_Size );
    cell_LSTM_Vec.back().c_L2->out->data_b.setConstant( 0 );

    MidData* end_h = new MidData;
    cell_LSTM_Vec.back().h_L4->out = end_h;

    std::cout << "good_3" << std::endl;

  }

  void calForward()
  {
    for( auto i = cell_LSTM_Vec.begin(); i != cell_LSTM_Vec.end(); i++ )
    {
      (*i).calForward();
    }
    std::cout << "Out is:******" << cell_LSTM_Vec.back().h_L4->out->data_f << std::endl;
  }

  void calBackward()
  {
    for( auto i = cell_LSTM_Vec.rbegin(); i != cell_LSTM_Vec.rend(); i++ )
    {
      (*i).calBackward();
    }
  }

  void upW()
  {
    for( auto i = cell_LSTM_Vec.begin(); i != cell_LSTM_Vec.end(); i++ )
    {
      (*i).upW();
    }
  }

  void setInitial( int s, std::vector<std::vector<MidData*>>& bottom_x_vec )
  {
    for( int i = 0; i < cell_LSTM_Vec.size(); i++ )
    {
      cell_LSTM_Vec[i].update_L1->bottom_x = bottom_x_vec[s][i];
      cell_LSTM_Vec[i].compute_L1->bottom_x = bottom_x_vec[s][i];
      cell_LSTM_Vec[i].forget_L1->bottom_x = bottom_x_vec[s][i];
      cell_LSTM_Vec[i].output_L1->bottom_x = bottom_x_vec[s][i];
    }
    //  注意： 每次这里都需要清灵，因为在反向传播的时候，h(t)的LOSS会传递给这里，导致发生错误
    cell_LSTM_Vec.back().c_L2->out->data_b.setConstant( 0 );    
  }

      //  Core_Update_LSTM_L1
  mat w_l_o_update, w_b_o_update,
      //  Core_Compute_LSTM_L1
      w_l_o_compute, w_b_o_compute,
      //  Core_Forget_LSTM_L1
      w_l_o_forget, w_b_o_forget,
      //  Core_Output_LSTM_L1
      w_l_o_output, w_b_o_output;

  std::vector<Cell_LSTM> cell_LSTM_Vec;
};

void Test_LSTM()
{
  Net_LSTM net_LSTM_T(4,4,4);

  std::vector<std::vector<MidData*>> bottom_x_vec;

  std::vector<mat> out_lable_vec;

  //=================================================================
  std::vector<MidData*> layer_vec;

  MidData* mid_1 = new MidData;
  mid_1->data_f.resize(1,4);
  mid_1->data_f.setConstant(0);
  mid_1->data_f(0,0) = 1;

  MidData* mid_2 = new MidData;
  mid_2->data_f.resize(1,4);
  mid_2->data_f.setConstant(0);
  mid_2->data_f(0,1) = 1;

  MidData* mid_3 = new MidData;
  mid_3->data_f.resize(1,4);
  mid_3->data_f.setConstant(0);
  mid_3->data_f(0,2) = 1;

  MidData* mid_4 = new MidData;
  mid_4->data_f.resize(1,4);
  mid_4->data_f.setConstant(0);
  mid_4->data_f(0,3) = 1;

  mat out_lable(1,4);
  out_lable.setConstant(0);
  out_lable(0,0) = 1;
  
  layer_vec.push_back(mid_1);
  layer_vec.push_back(mid_2);
  layer_vec.push_back(mid_3);
  layer_vec.push_back(mid_4);

  bottom_x_vec.push_back(layer_vec);

  out_lable_vec.push_back(out_lable);

  //  ================================================================
  std::vector<MidData*> layer_vec_1;

  MidData* mid_1_1 = new MidData;
  mid_1_1->data_f.resize(1,4);
  mid_1_1->data_f.setConstant(0);
  mid_1_1->data_f(0,3) = 1;

  MidData* mid_2_1 = new MidData;
  mid_2_1->data_f.resize(1,4);
  mid_2_1->data_f.setConstant(0);
  mid_2_1->data_f(0,2) = 1;

  MidData* mid_3_1 = new MidData;
  mid_3_1->data_f.resize(1,4);
  mid_3_1->data_f.setConstant(0);
  mid_3_1->data_f(0,1) = 1;

  MidData* mid_4_1 = new MidData;
  mid_4_1->data_f.resize(1,4);
  mid_4_1->data_f.setConstant(0);
  mid_4_1->data_f(0,0) = 1;

  mat out_lable_1(1,4);
  out_lable_1.setConstant(0);
  out_lable_1(0,3) = 1;

  layer_vec_1.push_back(mid_1_1);
  layer_vec_1.push_back(mid_2_1);
  layer_vec_1.push_back(mid_3_1);
  layer_vec_1.push_back(mid_4_1);

  bottom_x_vec.push_back(layer_vec_1);

  out_lable_vec.push_back(out_lable_1);
  //===================================================================

  for( int i = 0; i < 10000; i ++ )
  {
    net_LSTM_T.setInitial( i%2, bottom_x_vec ); 
    net_LSTM_T.calForward();
    net_LSTM_T.cell_LSTM_Vec.back().h_L4->out->data_b = net_LSTM_T.cell_LSTM_Vec.back().h_L4->out->data_f - out_lable_vec[i%2];
    net_LSTM_T.calBackward();

    net_LSTM_T.upW();
  }
}

} // namespace RNN