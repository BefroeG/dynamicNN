#include "NeuralNetwork.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <numeric>
#include <cmath>
#include <utility>
#include <random>
#include <chrono>
#include <algorithm>
#include <iomanip>

// 激活函数实现
Matrix NeuralNetwork::activate(const Matrix& input, ActivationType type) const {
    Matrix output(input.getRows(), input.getCols());
    for (int i = 0; i < input.getRows(); ++i) {
        for (int j = 0; j < input.getCols(); ++j) {
            if (type == ActivationType::LINEAR) {
                output(i, j) = input(i, j); // 线性激活：y=x
            }
            else if (type == ActivationType::RELU) {
                output(i, j) = std::max(0.0, input(i, j)); // ReLU：max(0,x)
            }
        }
    }
    return output;
}

// 激活函数导数
Matrix NeuralNetwork::activationDerivative(const Matrix& input, ActivationType type) const {
    Matrix output(input.getRows(), input.getCols());
    for (int i = 0; i < input.getRows(); ++i) {
        for (int j = 0; j < input.getCols(); ++j) {
            if (type == ActivationType::LINEAR) {
                output(i, j) = 1.0; // 线性激活导数为1
            }
            else if (type == ActivationType::RELU) {
                output(i, j) = (input(i, j) > 0) ? 1.0 : 0.0; // ReLU导数
            }
        }
    }
    return output;
}

// 加载数据文件
void NeuralNetwork::loadData(const std::string& filename, char delimiter) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + filename);
    }

    true_points.clear();
    double _x, _y;
    while (file >> _x >> _y) {
        true_points.push_back(std::make_pair(_x, _y));
    }
    file.close();
    std::cout << ">>>> 文件读取完成: 成功加载数据= " << true_points.size() << std::endl;
}

// 标准化数据（均值0，方差1）
void NeuralNetwork::standardizeData() {
    int n = true_points.size();
    if (n == 0) {
        throw std::runtime_error("无数据可标准化");
    }

    // 计算均值和标准差
    double x_sum = 0.0, x_sq_sum = 0.0;
    double y_sum = 0.0, y_sq_sum = 0.0;
    for (const auto& p : true_points) {
        x_sum += p.first;
        x_sq_sum += p.first * p.first;
        y_sum += p.second;
        y_sq_sum += p.second * p.second;
    }
    x_mean = x_sum / n;
    y_mean = y_sum / n;
    x_std = std::sqrt((x_sq_sum / n) - (x_mean * x_mean));
    y_std = std::sqrt((y_sq_sum / n) - (y_mean * y_mean));

    // 避免标准差为0
    if (x_std < EPS) x_std += EPS;
    if (y_std < EPS) y_std += EPS;

    // 标准化后存入矩阵
    norm_input = Matrix(n, 1);
    norm_target = Matrix(n, 1);
    for (int i = 0; i < n; ++i) {
        norm_input(i, 0) = (true_points[i].first - x_mean) / x_std;
        norm_target(i, 0) = (true_points[i].second - y_mean) / y_std;
    }
    std::cout << ">>>> 标准化完成: 成功处理数据= " << n << std::endl;
}

// 初始化网络层（带批归一化控制）
void NeuralNetwork::initLayers(const std::vector<int>& hidden_layers, bool use_bn) {
    layers.clear();

    // 初始化随机数生成器（仅初始化一次）
    static std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
    static std::normal_distribution<double> w_dist(0.0, 1.0);
    static std::uniform_real_distribution<double> b_dist(0.01, 0.1);

    int input_dim = 1; // 单变量回归，输入维度固定为1
    // 初始化隐藏层（He初始化适配ReLU）
    for (int h_dim : hidden_layers) {
        Matrix w(h_dim, input_dim);
        Matrix b(h_dim, 1);
        double std_dev_he = std::sqrt(2.0 / input_dim); // He初始化标准差
        // 权重初始化
        for (int i = 0; i < w.getRows(); ++i) {
            for (int j = 0; j < w.getCols(); ++j) {
                w(i, j) = w_dist(gen) * std_dev_he;
            }
        }
        // 偏置初始化为小正数（避免ReLU初始死亡）
        for (int i = 0; i < b.getRows(); ++i) {
            if (use_bn) {
                b(i, 0) = 0.0;
            } 
            else {
                b(i, 0) = b_dist(gen);
            }
        }
        // 创建隐藏层（ReLU激活，批归一化由参数控制）
        layers.emplace_back(w, b, ActivationType::RELU, use_bn);
        input_dim = h_dim;
    }

    // 初始化输出层（Xavier初始化适配线性激活）
    Matrix out_w(1, input_dim);
    Matrix out_b(1, 1);
    double std_dev_xavier = std::sqrt(1.0 / (input_dim + 1)); // Xavier标准差
    for (int i = 0; i < out_w.getRows(); ++i) {
        for (int j = 0; j < out_w.getCols(); ++j) {
            out_w(i, j) = w_dist(gen) * std_dev_xavier;
        }
    }
    out_b(0, 0) = b_dist(gen);
    // 输出层禁用批归一化
    layers.emplace_back(out_w, out_b, ActivationType::LINEAR, false);
    std::cout << ">>>> 网络层初始化完成" << std::endl;
}

// 打印网络结构
void NeuralNetwork::printNet() const {
    std::cout << std::string(50, '=') << " 网络结构 " << std::string(50, '=') << std::endl << std::endl;
    int input_dim = 1;
    std::cout << std::fixed << std::right;
    for (size_t i = 0; i < layers.size(); ++i) {
        const Layer& layer = layers[i];
        std::string act_type = (layer.activation == ActivationType::LINEAR) ? "LINEAR" : "RELU";
        std::cout << "第" << i + 1 << "层：输入维度=" << input_dim
            << ", 输出维度=" << layer.weight.getRows()
            << ", 激活函数=" << act_type
            << ", 权重形状=(" << layer.weight.getRows() << "," << layer.weight.getCols() << ")"
            << ", 偏置形状=(" << layer.bias.getRows() << "," << layer.bias.getCols() << ")"
            << ", 批归一化=" << (layer.use_batch_norm ? "启用" : "禁用")
            << std::endl;
        input_dim = layer.weight.getRows();

        // 打印层参数
        std::cout << "\nweight:\n";layer.weight.print();std::cout << std::endl;
        if(input_dim == 1) std::cout << std::endl;
        std::cout << "bias:  [ ";layer.bias.transpose().print();std::cout << " ]\n";
        if (layer.use_batch_norm) {
            std::cout << "gamma: [ ";layer.gamma.transpose().print();std::cout << " ]\n";
            std::cout << "beta:  [ ";layer.beta.transpose().print();std::cout << " ]\n";
        }
        if(i != layers.size()-1)
            std::cout << std::string(110, '-') << std::endl << std::endl;
    }
    //std::cout << std::string(102, '=') << std::endl << std::endl;
}

// 批归一化前向传播
Matrix NeuralNetwork::batchNormForward(const Matrix& z, Layer& layer) {
    int batch_size = z.getRows();
    int feat_size = z.getCols();

    if (!is_training) {
        // 推理阶段：使用移动平均的均值和方差（修复核心）
        Matrix mean_broadcast = layer.running_mean.broadcastRows(batch_size);
        // 重新计算running_var的inv_std（关键修复）
        Matrix std_running = layer.running_var.apply([this](double x) { return sqrt(x + EPS); });
        Matrix inv_std_running = std_running.apply([](double x) { return 1.0 / x; });
        Matrix inv_std_broadcast = inv_std_running.broadcastRows(batch_size);

        Matrix z_hat = (z - mean_broadcast).hadamard(inv_std_broadcast);
        Matrix gamma_broadcast = layer.gamma.transpose().broadcastRows(batch_size);
        Matrix beta_broadcast = layer.beta.transpose().broadcastRows(batch_size);
        Matrix z_norm = z_hat.hadamard(gamma_broadcast) + beta_broadcast;
        return z_norm;
    }

    // 训练阶段：计算批次均值和方差
    Matrix mean(1, feat_size, 0.0);
    for (int j = 0; j < feat_size; ++j) {
        double sum = 0.0;
        for (int i = 0; i < batch_size; ++i) {
            sum += z(i, j);
        }
        mean(0, j) = sum / batch_size;
    }
    layer.running_mean = layer.running_mean * layer.momentum + mean * (1 - layer.momentum);

    // 计算批次方差
    Matrix var(1, feat_size, 0.0);
    for (int j = 0; j < feat_size; ++j) {
        double sum = 0.0;
        double m = mean(0, j);
        for (int i = 0; i < batch_size; ++i) {
            sum += (z(i, j) - m) * (z(i, j) - m);
        }
        var(0, j) = sum / batch_size;
    }
    layer.running_var = layer.running_var * layer.momentum + var * (1 - layer.momentum);

    // 标准化（防止除零）
    Matrix std = var.apply([this, &layer](double x) { return sqrt(x + EPS); });
    Matrix inv_std = std.apply([](double x) { return 1.0 / x; });
    Matrix z_hat = (z - mean.broadcastRows(batch_size)).hadamard(inv_std.broadcastRows(batch_size));

    // 缩放和偏移
    Matrix z_norm = z_hat.hadamard(layer.gamma.transpose().broadcastRows(batch_size))
        + layer.beta.transpose().broadcastRows(batch_size);

    // 保存中间变量供反向传播
    layer.z_hat = z_hat;
    layer.var = var;
    layer.std = std;
    layer.inv_std = inv_std;

    return z_norm;
}

// 前向传播
Matrix NeuralNetwork::forward(const Matrix& epoch_input, bool pre_train) {
    Matrix current = epoch_input;
    for (Layer& layer : layers) {
        layer.batch_input = current;
        // 计算z = Wx + b
        layer.z = current * layer.weight.transpose() + layer.bias.transpose().broadcastRows(current.getRows());

        // 批归一化（若启用）
        Matrix z_input = layer.z;
        if (!pre_train && layer.use_batch_norm) {
            z_input = batchNormForward(layer.z, layer);
        }
        // 激活函数
        layer.a = activate(z_input, layer.activation);
        current = layer.a;
    }
    return current;
}

// 批归一化反向传播
Matrix NeuralNetwork::batchNormBackward(const Matrix& dz_norm, Layer& layer) {
    int batch_size = dz_norm.getRows();
    int feat_size = dz_norm.getCols();

    // 1. 计算 γ 和 β 的梯度（标准化：除以 batch_size 提升稳定性）
    Matrix d_gamma(feat_size, 1, 0.0);
    Matrix d_beta(feat_size, 1, 0.0);
    for (int j = 0; j < feat_size; ++j) {
        double dg = 0.0, db = 0.0;
        for (int i = 0; i < batch_size; ++i) {
            dg += dz_norm(i, j) * layer.z_hat(i, j);
            db += dz_norm(i, j);
        }
        d_gamma(j, 0) = dg;
        d_beta(j, 0) = db;
    }
    layer.d_gamma = d_gamma;
    layer.d_beta = d_beta;

    // 2. 计算 dz_hat = dy ⊙ γ（dy 是上层传入的 dz_norm）
    Matrix gamma_broadcast = layer.gamma.transpose().broadcastRows(batch_size);
    Matrix dz_hat = dz_norm.hadamard(gamma_broadcast);

    // 3. 计算原始 z 的梯度 dz（核心修复：移除多余的 gamma_j 乘法）
    Matrix dz(batch_size, feat_size, 0.0);
    for (int j = 0; j < feat_size; ++j) {
        double ivs = layer.inv_std(0, j); // 1/sqrt(var + EPS)
        double sum_dz_hat = 0.0;
        double sum_dz_hat_z_hat = 0.0;

        // 逐特征累加 dz_hat 和 dz_hat*z_hat
        for (int i = 0; i < batch_size; ++i) {
            sum_dz_hat += dz_hat(i, j);
            sum_dz_hat_z_hat += dz_hat(i, j) * layer.z_hat(i, j);
        }

        // 计算每个样本的 dz_i（标准公式实现）
        for (int i = 0; i < batch_size; ++i) {
            double term1 = dz_hat(i, j) * ivs;
            double term2 = sum_dz_hat * ivs / batch_size;
            double term3 = layer.z_hat(i, j) * sum_dz_hat_z_hat * ivs / batch_size;
            dz(i, j) = term1 - term2 - term3;
        }
    }

    return dz;
}

// 反向传播
void NeuralNetwork::backward(const Matrix& epoch_target, const Matrix& epoch_output) {
    // 输出层误差（均方误差导数）
    int n = epoch_target.getRows();
    Matrix delta = (epoch_output - epoch_target) * 2.0 * (1.0 / n);

    // 从输出层反向计算梯度
    for (int i = layers.size() - 1; i >= 0; --i) {
        Layer& layer = layers[i];
        // 激活函数导数
        Matrix z_input = layer.use_batch_norm ? layer.z_hat : layer.z;
        delta = delta.hadamard(activationDerivative(z_input, layer.activation));

        // 批归一化反向传播
        if (layer.use_batch_norm) {
            delta = batchNormBackward(delta, layer);
        }

        // 计算权重梯度
        Matrix a_prev = layer.batch_input;
        layer.d_weight = (delta.transpose() * a_prev);

        // 计算偏置梯度
        Matrix bias_sum(delta.getCols(), 1);
        for (int j = 0; j < delta.getCols(); ++j) {
            double sum = 0.0;
            for (int k = 0; k < delta.getRows(); ++k) {
                sum += delta(k, j);
            }
            bias_sum(j, 0) = sum;
        }
        layer.d_bias = bias_sum;

        // 计算上一层的delta
        if (i > 0) {
            delta = delta * layer.weight;
        }
    }
}

// 更新网络参数
void NeuralNetwork::updateParameters() {
    clipBNGradients();//裁剪BN的梯度
    if (opt == OptimizerType::BGD) {
        // 批量梯度下降更新
        for (Layer& layer : layers) {
            layer.weight = layer.weight - layer.d_weight * current_lr;
            layer.bias = layer.bias - layer.d_bias * current_lr;

            // 更新批归一化参数
            if (layer.use_batch_norm) {
                layer.gamma = layer.gamma - layer.d_gamma * current_lr * bn_lr_rate;
                layer.beta = layer.beta - layer.d_beta * current_lr * bn_lr_rate;
            }
        }
    }
    else if (opt == OptimizerType::ADAM) {
        // ADAM优化器更新
        adam_step++;
        double t = adam_step;
        double beta1_t = pow(beta1, t);
        double beta2_t = pow(beta2, t);
        double m_correction = 1.0 / (1.0 - beta1_t);
        double v_correction = 1.0 / (1.0 - beta2_t);

        for (Layer& layer : layers) {
            // 更新权重的一阶矩和二阶矩
            layer.m_weight = layer.m_weight * beta1 + layer.d_weight * (1.0 - beta1);
            layer.v_weight = layer.v_weight * beta2 + (layer.d_weight.hadamard(layer.d_weight)) * (1.0 - beta2);

            // 权重偏差校正
            Matrix m_weight_corrected = layer.m_weight * m_correction;
            Matrix v_weight_corrected = layer.v_weight * v_correction;

            // 更新权重
            Matrix weight_update = m_weight_corrected.hadamard(
                (v_weight_corrected.apply([](double x) { return sqrt(x); }) + EPS).apply([](double x) { return 1.0 / x; })
            );
            layer.weight = layer.weight - weight_update * current_lr;

            // 更新偏置的一阶矩和二阶矩
            layer.m_bias = layer.m_bias * beta1 + layer.d_bias * (1.0 - beta1);
            layer.v_bias = layer.v_bias * beta2 + (layer.d_bias.hadamard(layer.d_bias)) * (1.0 - beta2);

            // 偏置偏差校正
            Matrix m_bias_corrected = layer.m_bias * m_correction;
            Matrix v_bias_corrected = layer.v_bias * v_correction;

            // 更新偏置
            Matrix bias_update = m_bias_corrected.hadamard(
                (v_bias_corrected.apply([](double x) { return sqrt(x); }) + EPS).apply([](double x) { return 1.0 / x; })
            );
            layer.bias = layer.bias - bias_update * current_lr;

            // ADAM 优化器中 BN 参数更新的修复代码
            if (layer.use_batch_norm) {
                // 更新 γ 的一阶矩和二阶矩（使用独立的 m_gamma/v_gamma）
                layer.m_gamma = layer.m_gamma * beta1 + layer.d_gamma * (1.0 - beta1);
                layer.v_gamma = layer.v_gamma * beta2 + (layer.d_gamma.hadamard(layer.d_gamma)) * (1.0 - beta2);
                // γ 的偏差校正
                Matrix m_gamma_corrected = layer.m_gamma * m_correction;
                Matrix v_gamma_corrected = layer.v_gamma * v_correction;
                Matrix gamma_update = m_gamma_corrected.hadamard(
                    (v_gamma_corrected.apply([](double x) { return sqrt(x); }) + EPS).apply([](double x) { return 1.0 / x; })
                );
                layer.gamma = layer.gamma - gamma_update * current_lr;

                // 更新 β 的一阶矩和二阶矩（使用独立的 m_beta/v_beta）
                layer.m_beta = layer.m_beta * beta1 + layer.d_beta * (1.0 - beta1);
                layer.v_beta = layer.v_beta * beta2 + (layer.d_beta.hadamard(layer.d_beta)) * (1.0 - beta2);
                // β 的偏差校正
                Matrix m_beta_corrected = layer.m_beta * m_correction;
                Matrix v_beta_corrected = layer.v_beta * v_correction;
                Matrix beta_update = m_beta_corrected.hadamard(
                    (v_beta_corrected.apply([](double x) { return sqrt(x); }) + EPS).apply([](double x) { return 1.0 / x; })
                );
                layer.beta = layer.beta - beta_update * current_lr;
            }
        }
    }
    resetParameters();//重置所有梯度
}

// BN梯度裁剪
void NeuralNetwork::clipBNGradients(double max_norm) {

    for (auto& layer : layers) {
        double layer_grad_norm = 0.0;

        // 批归一化参数梯度
        if (layer.use_batch_norm) {
            for (int i = 0; i < layer.d_gamma.getRows(); ++i) {
                layer_grad_norm += layer.d_gamma(i, 0) * layer.d_gamma(i, 0);
                layer_grad_norm += layer.d_beta(i, 0) * layer.d_beta(i, 0);
            }
            layer_grad_norm = std::sqrt(layer_grad_norm);

            // 逐层裁剪
            double scale = 1;
            if (layer_grad_norm > max_norm) {
                scale = max_norm / layer_grad_norm;
            }
            layer.d_gamma = layer.d_gamma * scale;
            layer.d_beta = layer.d_beta * scale;
        }
    }
}

// 重置梯度参数
void NeuralNetwork::resetParameters() {
    for (Layer& layer : layers) {
        layer.d_weight = Matrix(layer.weight.getRows(), layer.weight.getCols(), 0);
        layer.d_bias = Matrix(layer.bias.getRows(), layer.bias.getCols(), 0);
        if (layer.use_batch_norm) {
            layer.d_gamma = Matrix(layer.gamma.getRows(), layer.gamma.getCols(), 0);
            layer.d_beta = Matrix(layer.beta.getRows(), layer.beta.getCols(), 0);
            // 新增：重置 BN 的 ADAM 矩
            layer.m_gamma = Matrix(layer.m_gamma.getRows(), layer.m_gamma.getCols(), 0);
            layer.v_gamma = Matrix(layer.v_gamma.getRows(), layer.v_gamma.getCols(), 0);
            layer.m_beta = Matrix(layer.m_beta.getRows(), layer.m_beta.getCols(), 0);
            layer.v_beta = Matrix(layer.v_beta.getRows(), layer.v_beta.getCols(), 0);
        }
    }
}

// 预训练检查权重参数是否造成神经元死亡
void NeuralNetwork::preTrain() {
    int max_training_limit = 3;     //最大训练次数
    double reset_death_ratio = 0.6; //重置参数阈值
    std::cout << ">>>> 开始预训练: 神经元死亡重置阈值= " << std::fixed << std::setprecision(1) << reset_death_ratio * 100
        << "%) | 最大训练次数 = " << max_training_limit
        << " | 训练数据条数= " << norm_input.getRows() << std::endl;
    for (size_t i = 0;i < max_training_limit;++i) {
        forward(norm_input, true);
        if (checkNeuronDeath(reset_death_ratio)) {
            // 清空所有层的中间数据 TODO
            std::cout << ">>>> 预训练完成: 训练次数 = (" << i + 1 << " / " << max_training_limit << ")\n" << std::endl;  
            return;
        }
        else {
            std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
            std::normal_distribution<double> w_dist(0.0, 1.0);
            std::uniform_real_distribution<double> b_dist(0.01, 0.1);
            // 重置隐藏层的全部w b参数
            int input_dim = 1;
            for (size_t l = 0;l < layers.size();++l) {
                Layer& layer = layers[l];
                if (layer.activation == ActivationType::LINEAR) {
                    continue;
                }
                double std_dev_he = std::sqrt(2.0 / input_dim); // He初始化标准差
                // 权重初始化
                for (int i = 0; i < layer.weight.getRows(); ++i) {
                    for (int j = 0; j < layer.weight.getCols(); ++j) {
                        layer.weight(i, j) = w_dist(gen) * std_dev_he;
                    }
                }
                // 偏置初始化为小正数（避免ReLU初始死亡）
                for (int i = 0; i < layer.bias.getRows(); ++i) {
                    if (layer.use_batch_norm) {
                        layer.bias(i, 0) = 0.0;
                    }
                    else {
                        layer.bias(i, 0) = b_dist(gen);
                    }
                }
                input_dim = layer.weight.getRows();
            }
        }
    }
    std::cout << ">>>> 预训练完成: 训练次数 = (" << max_training_limit << " / " << max_training_limit << ")\n" << std::endl;
}

// 训练网络
void NeuralNetwork::train(size_t _epochs, size_t batch_size) {
    if (norm_input.getRows() == 0 || norm_target.getRows() == 0) {
        throw std::runtime_error("请先加载并标准化数据");
    }
    if (layers.empty()) {
        throw std::runtime_error("请先初始化网络层");
    }
    if (batch_size == 0 || batch_size > norm_input.getRows()) {
        throw std::runtime_error("批次大小无效（需大于0且不超过数据总量）");
    }
    is_training = true;
    epochs = _epochs;
    recordOriginalParameters();//记录原始参数
    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();

    int n_samples = norm_input.getRows();
    int n_batches = (n_samples + batch_size - 1) / batch_size;
    std::cout << "\n>>>> 开始训练: 总轮数= " << epochs << " | 初始学习率= " << lr
        << " | 批次大小= " << batch_size << " | 每轮批次数= " << n_batches << std::endl << std::endl;

    // 样本索引初始化
    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);

    double final_delta = 0.0;  // 最终损失值

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        // 学习率衰减
        if (static_cast<int>(epoch) % lr_decay_step == 0 && epoch > 0) {
            current_lr *= lr_decay_rate;
        }

        // 打乱样本顺序
        std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device{}()));

        double epoch_loss = 0.0;
        for (int batch = 0; batch < n_batches; ++batch) {
            // 提取批次数据
            int start = batch * batch_size;
            int end = std::min(static_cast<int>(start + batch_size), n_samples);
            int current_batch_size = end - start;

            Matrix batch_input(current_batch_size, 1);
            Matrix batch_target(current_batch_size, 1);
            for (int i = 0; i < current_batch_size; ++i) {
                int idx = indices[start + i];
                batch_input(i, 0) = norm_input(idx, 0);
                batch_target(i, 0) = norm_target(idx, 0);
            }

            // 前向传播
            Matrix batch_output = forward(batch_input);

            // 反向传播
            backward(batch_target, batch_output);

            // 更新参数
            updateParameters();

            // 计算批次损失
            double batch_loss = 0.0;
            for (int i = 0; i < current_batch_size; ++i) {
                double diff = batch_output(i, 0) - batch_target(i, 0);
                batch_loss += diff * diff;
            }
            epoch_loss += batch_loss;
        }
        // 计算平均损失和更新参数
        final_delta = epoch_loss / static_cast<double>(n_samples);

        if (epoch % (epochs / 40) == 0 || epoch == epochs - 1)
            lossVector.push_back(final_delta);//记录每轮的损失

        // 打印训练信息
        if (epoch % (epochs / 10) == 0 || epoch == epochs - 1) {
            std::cout << "Epoch " << epoch << "，均方误差= " << std::fixed << std::setprecision(6) << final_delta
                << "，(学习率= " << current_lr << ")" << std::endl;
        }
        if (epoch % (epochs / 20) == 0) {
            checkNeuronDeath(8.0);
        }
    }

    is_training = false;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "\n>>>> 训练完毕: 耗时= " << duration.count() / 1000000000 << " (秒) | 最终损失= "
        << std::fixed << std::setprecision(6) << final_delta << std::endl << std::endl;
}

// 记录网络原始参数
void NeuralNetwork::recordOriginalParameters() {
    for (Layer& layer : layers) {
        layer._weight = layer.weight;
        layer._bias = layer.bias;
        if (layer.use_batch_norm) {
            layer._gamma = layer.gamma;
            layer._beta = layer.beta;
        }
    }
}

// 打印最终参数比对
void NeuralNetwork::printTrainedNet() {
    std::cout << std::string(46, '=') << " 完成训练网络结构 " << std::string(46, '=') << std::endl << std::endl;
    int input_dim = 1;
    for (size_t i = 0; i < layers.size(); ++i) {
        const Layer& layer = layers[i];
        std::string act_type = (layer.activation == ActivationType::LINEAR) ? "LINEAR" : "RELU";
        std::cout << "第" << i + 1 << "层：输入维度=" << input_dim
            << ", 输出维度=" << layer.weight.getRows()
            << ", 激活函数=" << act_type
            << ", 权重形状=(" << layer.weight.getRows() << "," << layer.weight.getCols() << ")"
            << ", 偏置形状=(" << layer.bias.getRows() << "," << layer.bias.getCols() << ")"
            << ", 批归一化=" << (layer.use_batch_norm ? "启用" : "禁用")
            << std::endl;
        input_dim = layer.weight.getRows();
        std::cout << std::fixed << std::right;
        // 打印层参数
        std::cout << "\nweight:\n";
        Matrix w = layer.weight;
        Matrix _w = layer._weight;
        for (int i = 0; i < w.getRows(); ++i) {
            for (int j = 0; j < w.getCols(); ++j) {
                std::cout << std::setw(10) << std::setprecision(6) << w(i,j)
                    << "(" << std::setw(8) << std::setprecision(1)
                    << ((_w(i, j) == 0.0) ? w(i, j) : ((w(i, j) - _w(i, j)) / _w(i, j))) * 100 << "%)";
                if (j != w.getCols() - 1) {
                    std::cout << " | ";
                }
            }
            if (i != w.getRows() - 1 || w.getRows() > 1) {
                std::cout << std::endl;
            }
        }
        if(w.getRows() == 1) std::cout << std::endl;
       
        std::cout << "\nbias:  [ ";
        Matrix bs = layer.bias.transpose();
        Matrix _bs = layer._bias.transpose();
        for (int i = 0; i < bs.getCols(); ++i) {
                std::cout << std::setw(10) << std::setprecision(6) << bs(0, i)
                    << "(" << std::setw(8) << std::setprecision(1)
                    << ((_bs(0, i) == 0.0) ? bs(0, i) : ((bs(0, i) - _bs(0, i)) / _bs(0, i))) * 100 << "%)";
            if (i != bs.getCols() - 1) {
                std::cout << " | ";
            }
        }
        std::cout << " ]\n";
        if (layer.use_batch_norm) {
            std::cout << "gamma: [ ";
            Matrix g = layer.gamma.transpose();
            Matrix _g = layer._gamma.transpose();
            for (int i = 0; i < g.getCols(); ++i) {
                std::cout << std::setw(10) << std::setprecision(6) << g(0, i)
                    << "(" << std::setw(8) << std::setprecision(1)
                    <<((_g(0, i)==0.0) ? g(0, i):((g(0, i) - _g(0, i)) / _g(0, i))) * 100<< "%)";
                if (i != g.getCols() - 1) {
                    std::cout << " | ";
                }
            }
            std::cout << " ]\nbeta:  [ ";
            Matrix ba = layer.beta.transpose();
            Matrix _ba = layer._beta.transpose();
            for (int i = 0; i < ba.getCols(); ++i) {
                std::cout << std::setw(10) << std::setprecision(6) << ba(0, i)
                    << "(" << std::setw(8) << std::setprecision(1)
                    << ((_ba(0, i) == 0.0) ? ba(0, i) : ((ba(0, i) - _ba(0, i)) / _ba(0, i))) * 100 << "%)";
                if (i != ba.getCols() - 1) {
                    std::cout << " | ";
                }
            }
            std::cout << " ]\n";
        }

        if (i != layers.size() - 1)
            std::cout << std::string(110, '-') << std::endl << std::endl;
    }
    //std::cout << std::string(102, '=') << std::endl << std::endl;
}

// 预测单值
double NeuralNetwork::predict(double x) {
    double norm_x = (x - x_mean) / x_std;
    Matrix input(1, 1);
    input(0, 0) = norm_x;
    Matrix norm_output = forward(input);
    return inverseStandardizeY(norm_output(0, 0));
}

// 监测ReLU神经元死亡
bool NeuralNetwork::checkNeuronDeath(double death_ratio) {
    const double death_threshold = 1e-6;//神经元死亡阈值
    for (size_t l = 0; l < layers.size(); ++l) {
        const Layer& layer = layers[l];
        if (layer.activation == ActivationType::LINEAR) continue;

        const Matrix& activations = layer.a;
        int total_neurons = activations.getRows() * activations.getCols();
        if (total_neurons == 0) continue;

        int dead_neurons = 0;
        for (int i = 0; i < activations.getRows(); ++i) {
            for (int j = 0; j < activations.getCols(); ++j) {
                if (activations(i, j) <= death_threshold) {
                    dead_neurons++;
                }
            }
        }
        // 死亡比例
        double ratio = static_cast<double>(dead_neurons) / total_neurons;
        if (ratio >= death_ratio) {
            std::cout << ">>隐藏层 " << l + 1 << " (RELU): 死亡神经元 = " << dead_neurons << "/" << total_neurons
                << " (" << std::fixed << std::setprecision(1) << death_ratio * 100 << "%)" << std::endl;
            return false;
        }
    }
    return true;
}

// 控制台可视化拟合结果
void NeuralNetwork::plotFunction(bool ptrue, int width, int height) {
    std::cout << std::endl << std::endl << std::string(49, '=') << " 函数拟合曲线 " << std::string(49, '=') << std::endl << std::endl;
    is_training = false;
    size_t size = true_points.size();
    Matrix pred_output = forward(norm_input);

    std::vector<std::pair<double, double>> pred_points(size);
    for (size_t i = 0; i < size; ++i) {
        pred_points[i] = std::make_pair(true_points[i].first, inverseStandardizeY(pred_output(i, 0)));
    }

    // 计算坐标范围
    double x_min = true_points[0].first;
    double x_max = true_points[0].first;
    double y_min = true_points[0].second;
    double y_max = true_points[0].second;

    for (const auto& p : true_points) {
        if (p.first < x_min) x_min = p.first;
        if (p.first > x_max) x_max = p.first;
        if (p.second < y_min) y_min = p.second;
        if (p.second > y_max) y_max = p.second;
    }
    for (const auto& p : pred_points) {
        if (p.second < y_min) y_min = p.second;
        if (p.second > y_max) y_max = p.second;
    }

    // 扩展范围
    double x_range = x_max - x_min;
    double y_range = y_max - y_min;
    x_min -= 0.0 * x_range;
    x_max += 0.0 * x_range;
    y_min -= 0.0 * y_range;
    y_max += 0.0 * y_range;

    // 创建画布
    std::vector<std::vector<char>> canvas(height, std::vector<char>(width, ' '));

    // 绘制坐标轴
    if (x_min <= 0 && 0 <= x_max) {
        int y_axis_x = static_cast<int>((0 - x_min) / (x_max - x_min) * (width - 1));
        y_axis_x = std::max(0, std::min(width - 1, y_axis_x));
        for (int i = 0; i < height; ++i) {
            if (canvas[i][y_axis_x] == ' ') {
                canvas[i][y_axis_x] = '|';
            }
        }
    }
    if (y_min <= 0 && 0 <= y_max) {
        int x_axis_y = static_cast<int>((y_max - 0) / (y_max - y_min) * (height - 1));
        x_axis_y = std::max(0, std::min(height - 1, x_axis_y));
        for (int i = 0; i < width; ++i) {
            if (canvas[x_axis_y][i] == ' ') {
                canvas[x_axis_y][i] = '-';
            }
        }
        // 绘制原点
        if (x_min <= 0 && 0 <= x_max) {
            int zero_y = static_cast<int>((y_max - 0) / (y_max - y_min) * (height - 1));
            int zero_x = static_cast<int>((0 - x_min) / (x_max - x_min) * (width - 1));
            zero_y = std::max(0, std::min(height - 1, zero_y));
            zero_x = std::max(0, std::min(width - 1, zero_x));
            canvas[zero_y][zero_x] = 'o';
        }
    }

    // 绘制预测曲线
    for (const auto& p : pred_points) {
        int col = static_cast<int>((p.first - x_min) / (x_max - x_min) * (width - 1));
        int row = static_cast<int>((y_max - p.second) / (y_max - y_min) * (height - 1));
        col = std::max(0, std::min(width - 1, col));
        row = std::max(0, std::min(height - 1, row));
        if (canvas[row][col] == ' ') {
            canvas[row][col] = '*';
        }
    }

    // 绘制真实数据点
    if (ptrue) {
        for (const auto& p : true_points) {
            int col = static_cast<int>((p.first - x_min) / (x_max - x_min) * (width - 1));
            int row = static_cast<int>((y_max - p.second) / (y_max - y_min) * (height - 1));
            col = std::max(0, std::min(width - 1, col));
            row = std::max(0, std::min(height - 1, row));

            if (canvas[row][col] == ' ' || canvas[row][col] == '|' || canvas[row][col] == '-') {
                canvas[row][col] = '+';
            }
            else if (canvas[row][col] == '*') {
                canvas[row][col] = '#';
            }
        }
    }

    // 打印画布
    //std::cout << "\n函数拟合可视化 (宽度=" << width << ", 高度=" << height << "):" << std::endl << std::endl;
    std::cout << std::string((width / 2 - 21 >= 1 ? (width / 2 - 21) : 1), ' ')
        << "\033[32m+ 真实数据点\033[0m   \033[31m* 预测数据点\033[0m   \033[33m# 重合数据点\033[0m"
        << std::endl << std::endl;

    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            switch (canvas[row][col]) {
            case '*':
                std::cout << "\033[31m*\033[0m";
                break;
            case '+':
                std::cout << "\033[32m+\033[0m";
                break;
            case '#':
                std::cout << "\033[33m#\033[0m";
                break;
            default:
                std::cout << canvas[row][col];
                break;
            }
        }
        std::cout << std::endl;
    }
    std::cout << std::endl << std::endl;
    std::cout << "X范围: [" << x_min << ", " << x_max << "]  Y范围: [" << y_min << ", " << y_max << "]" << std::endl << std::endl;
}

// 控制台可视化损失函数曲线
void NeuralNetwork::plotLossCurve(int width, int height, int precision) {
    // 1. 严格参数校验（强制修正为设定值，比如100x100）
    if (lossVector.empty()) {
        std::cout << "错误：损失值vector为空！" << std::endl;
        return;
    }
    if (lossVector.size() < 2) {
        std::cout << "错误：损失值数量少于2！" << std::endl;
        return;
    }
    // 5. 绘制画布（精简Y轴刻度，减少占列）
    std::cout << std::endl << std::string(48, '=') << " 损失函数曲线 " << std::string(48, '=') << std::endl << std::endl;
    const int plotWidth = std::max(1, width);   // 强制宽度=传入值
    const int plotHeight = std::max(1, height); // 强制高度=传入值

    // 2. 计算损失值极值
    double lossMin = *std::min_element(lossVector.begin(), lossVector.end());
    double lossMax = *std::max_element(lossVector.begin(), lossVector.end());
    double lossRange = lossMax - lossMin;
    if (fabs(lossRange) < 1e-12) {
        lossMin -= 0.5;
        lossMax += 0.5;
        lossRange = 1.0;
    }

    // 3. 初始化画布（严格尺寸）
    std::vector<std::vector<char>> canvas(plotHeight, std::vector<char>(plotWidth, ' '));

    // 4. 强制X轴铺满宽度（仅保留星号，删除竖线绘制）
    int dataCount = static_cast<int>(lossVector.size());
    for (int canvasX = 0; canvasX < plotWidth; ++canvasX) { // 遍历画布每一列（0~99）
        // 反向映射：画布列 → 原始数据索引（强制铺满）
        double normX = static_cast<double>(canvasX) / (plotWidth - 1); // 0~1
        int dataIdx = static_cast<int>(normX * (dataCount - 1) + 0.5);
        dataIdx = dataIdx < 0 ? 0 : (dataIdx >= dataCount ? dataCount - 1 : dataIdx);

        // Y轴映射（强制铺满高度）
        double normY = (lossVector[dataIdx] - lossMin) / lossRange;
        int canvasY = static_cast<int>((1.0 - normY) * (plotHeight - 1) + 0.5);
        canvasY = canvasY < 0 ? 0 : (canvasY >= plotHeight ? plotHeight - 1 : canvasY);

        // 仅标记星号（删除竖线绘制逻辑）
        canvas[canvasY][canvasX] = '*';
    }

   
    for (int y = 0; y < plotHeight; ++y) {
        // 精简Y轴刻度：每10行显示一次，减少占列（仅占5列）
        if (y % 10 == 0 || y == plotHeight - 1) {
            double currentLoss = lossMax - (static_cast<double>(y) / (plotHeight - 1)) * lossRange;
            std::cout << std::fixed << std::setprecision(precision) << std::setw(precision+4) << currentLoss << "|";
        }
        else {
            std::cout << std::string(precision+4,' ') << "|"; // 固定占位，仅5列
        }

        // 绘制当前行的所有列（严格设定宽度）
        for (int x = 0; x < plotWidth; ++x) {
            if (canvas[y][x] == '*') {
                std::cout << "\033[33m*\033[0m";
            }
            else {
                std::cout << canvas[y][x];
            }
        }
        std::cout << std::endl;
    }

    // X轴分隔线（严格设定宽度）
    std::cout << std::string(precision + 4, ' ') << "+" << std::string(plotWidth, '-') << std::endl;

    // X轴刻度（强制铺满设定宽度）
    std::cout << std::setw(precision + 4) << "Epoch ";
    int tickStep = plotWidth / 10; // 按宽度均分10个刻度
    for (int x = 0; x < plotWidth; ++x) {
        // 修正条件：x是刻度间隔的倍数 或 x是绘图宽度最后一列
        if (x % tickStep == 0 || x == plotWidth - 1) {
            // 映射回原始迭代次数
            double normX = static_cast<double>(x) / (plotWidth - 1);
            //int iter = static_cast<int>(normX * (dataCount - 1) + 0.5);
            int iter = static_cast<int>(normX * (dataCount - 1));
            std::string tick = std::to_string(iter * epochs / 40);
            
            // 边界检查：避免刻度超出绘图宽度
            if (x + tick.size() <= plotWidth) {
                std::cout << tick;
                x += tick.size() - 1; // 跳过已显示的字符位置
            }
            else {
                // 最后一列空间不足时，只显示最后一位或简写
                std::cout << std::to_string(epochs);
            }
        }
        else {
            std::cout << " ";
        }
    }
}

