#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Matrix.h"
#include <vector>
#include <random> 
#include <chrono> 

// 激活函数类型（底层int避免类型歧义）
enum class ActivationType : int {
    LINEAR,  // 线性激活（回归任务）
    RELU     // ReLU激活（隐藏层）
};

// 优化器类型（底层int避免类型歧义）
enum class OptimizerType : int {
    BGD,  // 批量梯度下降
    ADAM  // Adam自适应优化器
};

constexpr static double EPS = 1e-8;               // 防止除零的小值

// 网络层结构体（包含权重、偏置、激活函数、梯度及批归一化/ADAM参数）
struct Layer {
    Matrix weight;        // 权重矩阵 [output_size, input_size]
    Matrix bias;          // 偏置矩阵 [output_size, 1]
    ActivationType activation;  // 激活函数类型

    Matrix d_weight;      // 权重梯度 [output_size, input_size]
    Matrix d_bias;        // 偏置梯度 [output_size, 1]

    Matrix batch_input;   // 批次输入数据
    Matrix z;             // 前向传播中间值（z = Wx + b）
    Matrix a;             // 前向传播输出（a = activate(z)）

    // ADAM优化器参数
    Matrix m_weight;      // 权重一阶矩（动量）
    Matrix v_weight;      // 权重二阶矩（平方梯度）
    Matrix m_bias;        // 偏置一阶矩（动量）
    Matrix v_bias;        // 偏置二阶矩（平方梯度）

    // 批归一化相关参数
    bool use_batch_norm;  // 当前层是否启用批归一化
    Matrix gamma;         // 缩放参数 γ (output_size, 1)
    Matrix beta;          // 偏移参数 β (output_size, 1)
    Matrix d_gamma;       // γ的梯度
    Matrix d_beta;        // β的梯度
    // 批归一化运行时统计（训练更新，推理使用）
    Matrix running_mean;  // 移动平均均值
    Matrix running_var;   // 移动平均方差
    double momentum = 0.9;// 移动平均动量

    // 批归一化中间变量（反向传播用）
    Matrix z_hat;         // 标准化后的z（未缩放偏移）
    Matrix var;           // 批次方差
    Matrix std;           // 批次标准差
    Matrix inv_std;       // 标准差的倒数

    Matrix _weight;        // 原始权重矩阵 [output_size, input_size]
    Matrix _bias;          // 原始偏置矩阵 [output_size, 1]
    Matrix _gamma;         // 原始缩放参数 γ (output_size, 1)
    Matrix _beta;          // 原始偏移参数 β (output_size, 1)

    // 新增：BN 参数的 ADAM 一阶/二阶矩
    Matrix m_gamma;  // γ 的一阶矩
    Matrix v_gamma;  // γ 的二阶矩
    Matrix m_beta;   // β 的一阶矩
    Matrix v_beta;   // β 的二阶矩

    // 层构造函数
    Layer(Matrix _w, Matrix _b, ActivationType _a, bool _use_bn = false)
        : weight(_w), bias(_b), activation(_a),
        d_weight(_w.getRows(), _w.getCols()),
        d_bias(_b.getRows(), _b.getCols()),
        // ADAM参数初始化（全0）
        m_weight(_w.getRows(), _w.getCols(), 0.0),
        v_weight(_w.getRows(), _w.getCols(), 0.0),
        m_bias(_b.getRows(), _b.getCols(), 0.0),
        v_bias(_b.getRows(), _b.getCols(), 0.0),
        // 批归一化参数初始化
        use_batch_norm(_use_bn),
        gamma(_b.getRows(), 1, 1.0),
        beta(_b.getRows(), 1, 0.0),
        d_gamma(_b.getRows(), 1, 0.0),
        d_beta(_b.getRows(), 1, 0.0),
        running_mean(1, _b.getRows(), 0.0),
        running_var(1, _b.getRows(), 0.0),
        var(_w.getRows(), 1),
        std(_w.getRows(), 1),
        inv_std(_w.getRows(), 1),
        m_gamma(_b.getRows(), 1, 0.0),  // 新增
        v_gamma(_b.getRows(), 1, 0.0),  // 新增
        m_beta(_b.getRows(), 1, 0.0),   // 新增
        v_beta(_b.getRows(), 1, 0.0)    // 新增
    {
        // 创建随机数引擎（mt19937 是性能和随机性都很好的梅森旋转算法）
        std::mt19937 generator(std::chrono::system_clock::now().time_since_epoch().count());

        std::uniform_real_distribution<double> gamma_dist(0.9, 1.1);// gamma范围 (0.9, 1.1)
        std::uniform_real_distribution<double> beta_dist(-0.1, 0.1);//beta范围 (-0.1,0.1)
        // 4. 生成0到1之间的随机数
        for (int i = 0; i < gamma.getRows();++i) {
            beta(i, 0) = beta_dist(generator);
            gamma(i, 0) = gamma_dist(generator);

            //gamma(i, 0) = 1; // TODO
            //beta(i, 0) = 0;  // TODO
        }
    }
};

// 神经网络类（单变量回归任务）
class NeuralNetwork {
private:
    int epochs;                 // 训练轮次数
    int true_epochs;            // 实际训练轮次
    double lr;                  // 初始学习率
    double current_lr;          // 当前学习率（支持衰减）
    bool is_training;           // 是否为训练模式
    OptimizerType opt;          // 优化器类型
    std::vector<std::pair<double, double>> true_points; // 原始数据(x,y)
    std::vector<Layer> layers;  // 网络层列表
    Matrix norm_input;          // 标准化输入 [n,1]
    Matrix norm_target;         // 标准化目标 [n,1]

    // 标准化参数
    double x_mean = 0.0;
    double x_std = 1.0;
    double y_mean = 0.0;
    double y_std = 1.0;

    // ADAM优化器超参数
    double beta1 = 0.9;         // 一阶矩衰减系数
    double beta2 = 0.999;       // 二阶矩衰减系数
    int adam_step = 0;          // ADAM迭代步数（偏差校正用）

    // 学习率衰减参数
    double lr_decay_rate = 0.995;// 学习率衰减率（每轮衰减0.5%）0.995
    int lr_decay_step = 100;    // 学习率衰减步长
    double bn_lr_rate = 0.01;     //批归一化参数学习率衰减

    std::vector<std::pair<int,double>> lossVector; //损失函数图像

    // 反标准化函数
    double inverseStandardizeY(double normalizedY) const { return normalizedY * y_std + y_mean; }
    double inverseStandardizeX(double normalizedX) const { return normalizedX * x_std + x_mean; }

    // 激活函数
    Matrix activate(const Matrix& input, ActivationType type) const;

    // 激活函数导数
    Matrix activationDerivative(const Matrix& input, ActivationType type) const;

    // 批归一化前向传播
    Matrix batchNormForward(const Matrix& z, Layer& layer);

    // 批归一化反向传播
    Matrix batchNormBackward(const Matrix& dz_norm, Layer& layer);

    // 记录网络原始参数
    void recordOriginalParameters();

public:
    // 构造函数（仅学习率+优化器，移除批归一化全局控制）
    NeuralNetwork(double _lr, OptimizerType _opt = OptimizerType::BGD)
        : lr(_lr), current_lr(_lr), opt(_opt), is_training(false) {
    }

    // 加载数据（支持自定义分隔符）
    void loadData(const std::string& filename, char delimiter = '\t');

    // 标准化数据
    void standardizeData();

    // 初始化网络层（带批归一化控制）
    void initLayers(const std::vector<int>& hidden_layers, bool use_bn = false);

    // 打印网络结构
    void printNet() const;

    // 训练网络（指定轮数和批次大小）
    void train(size_t epochs, size_t batch_size = 32, double early_stopping_loss = 0.0001);

    // 预训练检查权重参数是否造成神经元死亡
    void preTrain();

    // 前向传播
    Matrix forward(const Matrix& epoch_input,bool pre_train = false);

    // 反向传播
    void backward(const Matrix& epoch_target, const Matrix& epoch_output);

    // 更新网络参数
    void updateParameters();
    
    // BN梯度裁剪
    void clipBNGradients(double max_norm = 0.1);

    // 重置梯度参数
    void resetParameters();

    // 打印最终参数比对
    void printTrainedNet();

    // 预测单输入值
    double predict(double x);

    // 检测ReLU神经元死亡
    bool checkNeuronDeath(double death_ratio = 0.6);

    // 控制台可视化拟合结果
    void plotFunction(bool ptrue = true, int width = 110, int height = 50);

    // 在坐标图中打印预测点和真实数据点
    void plotLossCurve(int width = 100, int height = 50, int precision = 6);
};

#endif // NEURALNETWORK_H