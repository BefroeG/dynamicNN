#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <functional>  // 之前添加的apply所需头文件

class Matrix {
private:
    int rows;                    // 矩阵行数
    int cols;                    // 矩阵列数
    double* data_;               // 连续内存存储的矩阵数据
    mutable std::vector<double> tempRow;  // 临时行向量，用于向后兼容operator[]

public:
    // ==================== 构造函数和析构函数 ====================
    // 默认构造函数（新增）
    Matrix();

    // 构造函数：指定维度，初始值为0
    Matrix(int r, int c);

    // ========== 新增：指定维度+初始值的构造函数 ==========
    Matrix(int r, int c, double init_value);
    // =====================================================

    // 拷贝构造函数
    Matrix(const Matrix& other);

    // 移动构造函数
    Matrix(Matrix&& other) noexcept;

    // 析构函数
    ~Matrix();

    // ==================== 赋值运算符 ====================
    // 拷贝赋值运算符
    Matrix& operator=(const Matrix& other);

    // 移动赋值运算符
    Matrix& operator=(Matrix&& other) noexcept;

    // ==================== 基本属性访问 ====================
    int getRows() const;
    int getCols() const;
    const double* data() const { return data_; }

    // ==================== 元素访问 ====================
    double& operator()(int i, int j);
    const double& operator()(int i, int j) const;
    std::vector<double>& operator[](int i);
    const std::vector<double>& operator[](int i) const;

    // ==================== 算术运算符 ====================
    // 矩阵加法（矩阵+矩阵）
    Matrix operator+(const Matrix& other) const;

    // ========== 新增：矩阵+标量的加法运算符 ==========
    Matrix operator+(double scalar) const;
    // =================================================

    // 矩阵减法
    Matrix operator-(const Matrix& other) const;

    // 矩阵乘法
    Matrix operator*(const Matrix& other) const;

    // 标量乘法（右乘）
    Matrix operator*(double scalar) const;

    // 哈达玛积（逐元素乘法）
    Matrix hadamard(const Matrix& other) const;

    // ==================== 矩阵运算 ====================
    Matrix transpose() const;
    Matrix broadcastRows(int targetRows) const;

    // apply函数：逐元素应用自定义函数
    Matrix apply(std::function<double(double)> func) const;

    // ==================== 实用函数 ====================
    void print() const;
    void resize(int r, int c);
    bool empty() const { return rows == 0 || cols == 0 || data_ == nullptr; }

    // ==================== 友元函数 ====================
    friend Matrix operator*(double scalar, const Matrix& mat);
    // ========== 新增：标量+矩阵的友元加法（保证交换律） ==========
    friend Matrix operator+(double scalar, const Matrix& mat);
    // ===========================================================
};

// 全局标量乘法运算符（左乘）
Matrix operator*(double scalar, const Matrix& mat);

#endif // MATRIX_H