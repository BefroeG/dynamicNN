#include <string>
#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <iomanip>

// 六次多项式函数
double polynomial(double x) {
    // 六次多项式核心：0.1x^6 - 0.8x^5 + 2x^4 - 2.5x^3 + 1.5x^2 + 3x + 5
    double x2 = x * x;
    double x3 = x2 * x;
    double x4 = x3 * x;
    double x5 = x4 * x;
    double x6 = x5 * x;
    return 0.1 * x6 - 0.8 * x5 + 2 * x4 - 2.5 * x3 + 1.5 * x2 + 3 * x + 5;
}

int main() {
    // 1. 初始化随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    // x的范围：[-2.5, 2.5] 均匀分布
    std::uniform_real_distribution<double> x_dist(-2.5, 2.5);
    // 高斯噪声：均值0，标准差1.5
    std::normal_distribution<double> noise_dist(0.0, 1.5);

    // 2. 生成数据并写入文件
    std::ofstream outfile("polynomial_data.txt");
    if (!outfile.is_open()) {
        std::cerr << "无法打开文件！" << std::endl;
        return 1;
    }

    // 设置输出精度（保留6位小数）
    outfile << std::fixed << std::setprecision(6);

    // 生成1000行数据
    const int n_samples = 600;
    for (int i = 0; i < n_samples; ++i) {
        double x = x_dist(gen);
        double y_true = polynomial(x);
        //double y_noisy = y_true + noise_dist(gen); // 添加噪声
        outfile << x << "\t" << y_true << std::endl; // 制表符分隔，符合你的loadData格式
    }

    outfile.close();
    std::cout << "成功生成1000行六次多项式带噪声数据，保存到 polynomial_data.txt" << std::endl;

    // 输出前5行示例，方便验证
    std::cout << "\n前5行数据示例：" << std::endl;
    std::ifstream infile("polynomial_data.txt");
    std::string line;
    for (int i = 0; i < 5 && std::getline(infile, line); ++i) {
        std::cout << line << std::endl;
    }
    infile.close();

    return 0;
}