#include "NeuralNetwork.h"
#include <iostream>
#include <iomanip>

int main() {
    try {
        /******************************************************************
        *                            请配置参数                           *
        ******************************************************************/
        // 请设置学习率 （ 建议0.001 - 0.01 ）                                  
        double learning_rate = 0.001;

        // 请设置训练轮次（ 建议500 - 5000 ）                                                      
        size_t epochs = 500;//训练轮次    

        // 请设置神经网络【隐藏层】结构，无须设置【输入层】和【输出层】
        //【输出层】默认一个神经元，使用Linear作为激活函数
        //【隐藏层】使用ReLU作为激活函数
        //【隐藏层】可以设置0-3层，每层神经元数量为1-10个
        // 示例：表示【隐藏层】为2层 ，每层【隐藏层】的神经元个数分别为 8个，4个
        // std::vector<int> hidden_layer_sizes = { 8 , 4 };
        std::vector<int> hidden_layers = { 8,4 };

        // 请将训练文件拷贝到本程序目录下，并设置训练文件名称                    
        std::string file_name = "2X^3+X^2-3X+2";
        //*********************  以下为系统内置训练文件 **********************/
        //file_name = "-2X+8";               // y = -2X+8       无噪声 
        //  (-3,14)(-2,12)(-1,10)(0,8)(1,6)(2,4)(3,2)

        //file_name = "X^2";               // y = X^2           无噪声 
        //  (-3,9)(-2,4)(-1,1)(0,0)(1,1)(2,4)(3,9)

        //file_name = "X^3";               // y = X^3           无噪声  
        //  (-3,-27)(-2,-8)(-1,-1)(0,0)(1,1)(2,8)(3,27)

        //file_name = "2X^3+X^2-3X+2";     // y = 2X^3+X^2-3X+2 无噪声                                                       
        //  (-3,-34)(-2,-4)(-1,4)(0,2)(1,3)(2,16)(3,56)
        /******************************************************************
        *                          参数配置完毕                           *
        *******************************************************************/
        // 1. 创建神经网络（仅指定学习率和优化器，批归一化由initLayers控制）
        NeuralNetwork nn(learning_rate, OptimizerType::ADAM);

        // 2. 加载数据（修改file_name更换数据文件）
        nn.loadData(file_name);

        // 3. 标准化数据（固定流程，无需修改）
        nn.standardizeData();

        // 4. 初始化网络层（{隐藏层维度}, 批归一化开关：true/false）
        nn.initLayers(hidden_layers, false);

        // 5. 网络预训练避免启动时过多神经元死亡
        nn.preTrain();

        // 打印网络结构（可选，用于验证配置）
        nn.printNet();

        // 6. 训练网络（参数1：训练轮数，参数2：批次大小, 参数3：早停阈值）
        nn.train(epochs, 125, 0.0006);

        // 打印训练后网络参数（可选）
        nn.printTrainedNet();

        // 打印损失函数曲线（可选）
        nn.plotLossCurve(false);

        // 可视化开关：取消注释启用拟合曲线绘制（可选）
        nn.plotFunction(false);

        // 7. 测试新数据
        do {
            std::cout << "\n请输入x进行预测: ";
            double input_value;
            std::cin >> input_value;
            double pred_value = nn.predict(input_value);
            std::cout << "预测值是: " << pred_value << std::endl;
            std::cout << "请输出对应的真实值进行对比: ";
            double true_value;
            std::cin >> true_value;
            double loss = (true_value - pred_value) * (true_value - pred_value);
            std::cout << "MSE误差是: " << std::fixed << std::setprecision(6) << loss << std::endl;
        } while (true);

        std::cout << "\n测试结束，程序退出！" << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "错误：" << e.what() << std::endl;
        return 1;
    }
    return 0;
}