#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <string>

// 多项式训练数据生成器
// 参数说明：
// filename: 输出文件名
// coefficients: 多项式系数，按x^0, x^1, x^2...顺序（如[1,-3,5,2]对应y=1-3x+5x2+2x3）
// x_min/x_max: x的取值范围
// num_samples: 生成的样本数量
// add_noise: 是否添加随机噪声（模拟真实数据误差）
// noise_scale: 噪声幅度（越小越接近纯多项式）
bool generate_polynomial_data(const std::string& filename,
                              const std::vector<double>& coefficients,
                              double x_min, double x_max,
                              int num_samples,
                              bool add_noise = true,
                              double noise_scale = 0.1) {
    // 1. 初始化随机数生成器（保证每次运行生成不同数据）
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> x_dist(x_min, x_max); // x的均匀分布
    std::normal_distribution<double> noise_dist(0.0, noise_scale); // 高斯噪声（均值0，标准差noise_scale）

    // 2. 打开输出文件
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "错误：无法打开文件 " << filename << std::endl;
        return false;
    }

    // 3. 设置输出精度（避免科学计数法，保留6位小数）
    outfile << std::fixed << std::setprecision(6);

    // 4. 生成每个样本
    for (int i = 0; i < num_samples; ++i) {
        // 生成x值
        double x = x_dist(gen);
        
        // 计算多项式y值（核心：y = a0 + a1*x + a2*x2 + ... + an*x^n）
        double y = 0.0;
        for (size_t j = 0; j < coefficients.size(); ++j) {
            y += coefficients[j] * pow(x, j); // j是次数，coefficients[j]是对应系数
        }

        // 添加噪声（可选）
        if (add_noise) {
            y += noise_dist(gen);
        }

        // 写入文件：x\t y（Tab分隔）
        outfile << x << "\t" << y << std::endl;
    }

    // 5. 关闭文件
    outfile.close();
    std::cout << "成功生成数据文件：" << filename << std::endl;
    std::cout << "样本数量：" << num_samples << std::endl;
    std::cout << "多项式：y = ";
    for (size_t j = 0; j < coefficients.size(); ++j) {
        if (j == 0) {
            std::cout << coefficients[j];
        } else {
            std::cout << (coefficients[j] >= 0 ? " + " : " - ") << fabs(coefficients[j]) << "x^" << j;
        }
    }
    std::cout << (add_noise ? "（含高斯噪声）" : "（无噪声）") << std::endl;

    return true;
}

// 主函数：示例调用
int main() {
    // 1. 配置参数（可根据需求修改）
    std::string filename = "-2X+8"; // 输出文件名
    std::vector<double> coefficients = {8.0, -2.0, 0.0, 0.0}; // 多项式系数：1 - 3x + 5x2 + 2x3
    double x_min = -4.0; // x最小值
    double x_max = 4.0;  // x最大值
    int num_samples = 500; // 生成1000个样本
    bool add_noise = false; // 添加噪声
    double noise_scale = 0.01; // 噪声幅度（越大数据越分散）

    // 2. 生成数据
    generate_polynomial_data(filename, coefficients, x_min, x_max, num_samples, add_noise, noise_scale);

    return 0;
}
