#pragma once
#include "../LAYER/Layer.hpp"

namespace RNN
{

class Core_Update_LSTM_L1
{
public:
  Core_Update_LSTM_L1( mat& w_l_o, mat& w_b_o ) : w_l_o(w_l_o), w_b_o(w_b_o), left_h(new MidData), bottom_x(new MidData), out(new MidData){}

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
  mat &w_l_o;
  mat &w_b_o;

  MidData* left_h;
  MidData* bottom_x;

  MidData* out;
};

class Core_Compute_LSTM_L1
{
public:
  Core_Compute_LSTM_L1( mat& w_l_o, mat& w_b_o ) : w_l_o(w_l_o), w_b_o(w_b_o), left_h(new MidData), bottom_x(new MidData), out(new MidData){}

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

  mat &w_l_o;
  mat &w_b_o;

  MidData* left_h;
  MidData* bottom_x;

  MidData* out;
};

class Core_Forget_LSTM_L1
{
public:
  Core_Forget_LSTM_L1( mat& w_l_o, mat& w_b_o ) : w_l_o(w_l_o), w_b_o(w_b_o), left_h(new MidData), bottom_x(new MidData), out(new MidData){}

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

  mat &w_l_o;
  mat &w_b_o;

  MidData* left_h;
  MidData* bottom_x;

  MidData* out;
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

  MidData* left_Update;
  MidData* left_Compute;

  MidData* left_Forget;
  MidData* left_c;

  MidData* out;
};



class Core_c_LSTM_L3
{
public:

    void calForward()
    {
      out->data_f = tanh_F( left->data_f );
    }

    //  attation, should add the left( c_out ) before
    void calBackward()
    {
      left->data_b = (out->data_b.array() * tanh_B(out->data_f).array()).matrix() + left->data_b;
    }

  MidData* left;
  MidData* out;
};



class Core_Output_LSTM_L1
{
public:
  Core_Output_LSTM_L1( mat& w_l_o, mat& w_b_o ) : w_l_o(w_l_o), w_b_o(w_b_o), left_h(new MidData), bottom_x(new MidData), out(new MidData){}

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

  mat &w_l_o;
  mat &w_b_o;

  MidData* left_h;
  MidData* bottom_x;

  MidData* out;
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

  MidData* left_Output;
  MidData* left_c_L3;

  MidData* out;
};


class Cell_LSTM
{
public:
  Cell_LSTM( 
    //  Core_Update_LSTM_L1
    mat& w_l_o_update, mat& w_b_o_update,
    //  Core_Compute_LSTM_L1
    mat& w_l_o_compute, mat& w_b_o_compute,
    //  Core_Forget_LSTM_L1
    mat& w_l_o_forget, mat& w_b_o_forget,
    //  Core_Output_LSTM_L1
    mat& w_l_o_output, mat& w_b_o_output
    ) : 
    update_L1( new Core_Update_LSTM_L1( w_l_o_update, w_b_o_update ) ),
    compute_L1( new Core_Compute_LSTM_L1( w_l_o_compute, w_b_o_compute ) ),
    forget_L1( new Core_Forget_LSTM_L1( w_l_o_forget, w_b_o_forget ) ),
    output_L1( new Core_Output_LSTM_L1( w_l_o_output, w_b_o_output ) ),

    c_L2( new Core_c_LSTM_L2 ),

    c_L3( new Core_c_LSTM_L3 ),

    h_L4( new Core_h_LSTM_L4 )
  {
    MidData* o_update = new MidData;
    MidData* o_compute = new MidData;
    MidData* o_forget = new MidData;
    MidData* o_output = new MidData;

    update_L1->out = o_update;
    compute_L1->out = o_compute;
    forget_L1->out = o_forget;
    output_L1->out = o_output;

    c_L2->left_Update = o_update;
    c_L2->left_Compute = o_compute;
    c_L2->left_Forget = o_forget;

    
    
    MidData* o_c_L2 = new MidData;
    c_L2->out = o_c_L2;
    c_L3->left = o_c_L2;


    MidData* o_c_L3 = new MidData;
    c_L3->out = o_c_L3;
    h_L4->left_c_L3 = o_c_L3;
    h_L4->left_Output = o_output;

  }

  
  Core_Update_LSTM_L1* update_L1;
  Core_Compute_LSTM_L1* compute_L1;
  Core_Forget_LSTM_L1* forget_L1;
  Core_Output_LSTM_L1* output_L1;

  Core_c_LSTM_L2* c_L2;

  Core_c_LSTM_L3* c_L3;

  Core_h_LSTM_L4* h_L4;
};





class Layer_LSTM
{ 
  public:
    void calForward()
    {
        //  
        out_c->data_f = 
        sigmoid_F( left_h->data_f * w_f_h + bottom_x->data_f * w_f_b ).array() 
        * 
        left_c->data_f.array() 
        + 
        sigmoid_F( left_h->data_f * w_i_h + bottom_x->data_f * w_i_b ).array() 
        * 
        tanh_F( left_h->data_f * w_h + bottom_x->data_f * w_b ).array();

        //  
        out_h->data_f = 
        sigmoid_F( left_h->data_f * w_o_h + bottom_x->data_f * w_o_b ).array() 
        * 
        tanh_F( out_c->data_f ).array();

    }


  public:
    //mat& w_i_c;
    mat& w_i_h;
    mat& w_i_b;

    //mat& w_f_c;
    mat& w_f_h;
    mat& w_f_b;

    //mat& w_c;
    mat& w_h;
    mat& w_b;

    //mat& w_o_c;
    mat& w_o_h;
    mat& w_o_b;

    // MatrixIN
    MidData *left_c;
    MidData *left_h;
    MidData *bottom_x;

    // MatrixOUT
    MidData *out_c;
    MidData *out_h;
};

} // namespace RNN