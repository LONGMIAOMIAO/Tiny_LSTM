# Tiny_LSTM是一个轻型RNN框架，基于C++和C++矩阵运算库Eigen实现，目前仅支持CPU运算。  

1、 Tiny_LSTM自定义实现了 LSTM Cell 及其FPTT、BPTT 和Weight Update，可以快速进行模型搭建，快速运算。  

2、 基于该框架快速计算模型精度如下：  
Mnist Accuracy  :  96%    
Net 结构        :   LSTM + Softmax_Cross_Entropy （seqLength = 28, inputSize = 28, hiddenSize = 128 ）   
LSTM Cell 包括 Update Cell, Compute Cell, Forget Cell, Output Cell。  

3、 水平有限，代码有不完善的地方请在ISSUE批评指正！
4、 如果需要加载数据集的csv文件，请来神经网络设计研讨QQ群 826245492。
