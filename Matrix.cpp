#include <iostream>
#include <cmath>
#include <stdexcept>
#include <string>
#include <chrono>
#include <cstring>
#include <iomanip>
#include "Matrix.h"

// ==================== 默认构造函数 ====================
Matrix::Matrix() : rows(0), cols(0), data_(nullptr) {}

// ==================== 带参数的构造函数（维度） ====================
Matrix::Matrix(int r, int c) : rows(r), cols(c) {
    if (r <= 0 || c <= 0) {
        throw std::invalid_argument("Matrix构造：矩阵的维度必须为正数");
    }
    data_ = new double[rows * cols];
    std::memset(data_, 0, rows * cols * sizeof(double));
}

// ========== 新增：指定维度+初始值的构造函数 ==========
Matrix::Matrix(int r, int c, double init_value) : rows(r), cols(c) {
    if (r <= 0 || c <= 0) {
        throw std::invalid_argument("Matrix构造：矩阵的维度必须为正数");
    }
    data_ = new double[rows * cols];
    // 初始化所有元素为指定值（替代memset，支持非0初始值）
    const int totalElements = rows * cols;
    for (int i = 0; i < totalElements; ++i) {
        data_[i] = init_value;
    }
}
// =====================================================

// ==================== 拷贝构造函数 ====================
Matrix::Matrix(const Matrix& other) : rows(other.rows), cols(other.cols) {
    if (rows > 0 && cols > 0) {
        data_ = new double[rows * cols];
        std::memcpy(data_, other.data_, rows * cols * sizeof(double));
    }
    else {
        data_ = nullptr;
    }
}

// ==================== 移动构造函数 ====================
Matrix::Matrix(Matrix&& other) noexcept
    : rows(other.rows), cols(other.cols), data_(other.data_) {
    other.rows = 0;
    other.cols = 0;
    other.data_ = nullptr;
}

// ==================== 析构函数 ====================
Matrix::~Matrix() {
    delete[] data_;
    data_ = nullptr;
}

// ==================== 拷贝赋值运算符 ====================
Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        delete[] data_;
        rows = other.rows;
        cols = other.cols;
        if (rows > 0 && cols > 0) {
            data_ = new double[rows * cols];
            std::memcpy(data_, other.data_, rows * cols * sizeof(double));
        }
        else {
            data_ = nullptr;
        }
    }
    return *this;
}

// ==================== 移动赋值运算符 ====================
Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        delete[] data_;
        rows = other.rows;
        cols = other.cols;
        data_ = other.data_;
        other.rows = 0;
        other.cols = 0;
        other.data_ = nullptr;
    }
    return *this;
}

// ==================== 基本属性访问 ====================
int Matrix::getRows() const { return rows; }
int Matrix::getCols() const { return cols; }

// ==================== 高效元素访问 ====================
double& Matrix::operator()(int i, int j) {
    if (i < 0 || i >= rows || j < 0 || j >= cols) {
        throw std::out_of_range("矩阵索引超出范围");
    }
    return data_[i * cols + j];
}

const double& Matrix::operator()(int i, int j) const {
    if (i < 0 || i >= rows || j < 0 || j >= cols) {
        throw std::out_of_range("矩阵索引超出范围");
    }
    return data_[i * cols + j];
}

// ==================== 向后兼容的元素访问 ====================
std::vector<double>& Matrix::operator[](int i) {
    if (i < 0 || i >= rows) {
        throw std::out_of_range("矩阵索引超出范围");
    }
    tempRow.resize(cols);
    for (int j = 0; j < cols; ++j) {
        tempRow[j] = data_[i * cols + j];
    }
    return tempRow;
}

const std::vector<double>& Matrix::operator[](int i) const {
    if (i < 0 || i >= rows) {
        throw std::out_of_range("矩阵索引超出范围");
    }
    const_cast<Matrix*>(this)->tempRow.resize(cols);
    for (int j = 0; j < cols; ++j) {
        const_cast<Matrix*>(this)->tempRow[j] = data_[i * cols + j];
    }
    return tempRow;
}

// ==================== 重置矩阵大小 ====================
void Matrix::resize(int r, int c) {
    if (r <= 0 || c <= 0) {
        throw std::invalid_argument("Matrix-resize：矩阵的维度必须为正数");
    }
    if (r == rows && c == cols) return;
    delete[] data_;
    rows = r;
    cols = c;
    data_ = new double[rows * cols];
    std::memset(data_, 0, rows * cols * sizeof(double));
}

// ==================== 矩阵加法（矩阵+矩阵） ====================
Matrix Matrix::operator+(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("进行加法运算时，矩阵的维度必须匹配");
    }
    if (empty()) return Matrix();
    Matrix result(rows, cols);
    const int totalElements = rows * cols;
    double* resultData = result.data_;
    for (int i = 0; i < totalElements; ++i) {
        resultData[i] = data_[i] + other.data_[i];
    }
    return result;
}

// ========== 新增：矩阵+标量的加法运算符 ==========
Matrix Matrix::operator+(double scalar) const {
    if (empty()) return Matrix();
    Matrix result(rows, cols);
    const int totalElements = rows * cols;
    double* resultData = result.data_;
    for (int i = 0; i < totalElements; ++i) {
        resultData[i] = data_[i] + scalar;  // 每个元素加标量
    }
    return result;
}

// ========== 新增：标量+矩阵的友元加法（支持交换律） ==========
Matrix operator+(double scalar, const Matrix& mat) {
    return mat + scalar;  // 复用矩阵+标量的逻辑
}
// =====================================================

// ==================== 矩阵减法 ====================
Matrix Matrix::operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("进行减法运算时，矩阵的维度必须匹配");
    }
    if (empty()) return Matrix();
    Matrix result(rows, cols);
    const int totalElements = rows * cols;
    double* resultData = result.data_;
    for (int i = 0; i < totalElements; ++i) {
        resultData[i] = data_[i] - other.data_[i];
    }
    return result;
}

// ==================== 优化矩阵乘法 ====================
Matrix Matrix::operator*(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("进行乘法运算时，矩阵维度不匹配");
    }
    if (empty() || other.empty()) return Matrix(rows, other.cols);
    Matrix result(rows, other.cols);
    const int n = rows;
    const int m = cols;
    const int p = other.cols;
    double* resultData = result.data_;
    if (n <= 10 && m <= 10 && p <= 10) {
        double otherT[100];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < p; ++j) {
                otherT[j * m + i] = other.data_[i * p + j];
            }
        }
        for (int i = 0; i < n; ++i) {
            const double* thisRow = &data_[i * m];
            double* resultRow = &resultData[i * p];
            for (int j = 0; j < p; ++j) {
                const double* otherCol = &otherT[j * m];
                double sum = 0.0;
                for (int k = 0; k < m; ++k) {
                    sum += thisRow[k] * otherCol[k];
                }
                resultRow[j] = sum;
            }
        }
    }
    else {
        for (int i = 0; i < n; ++i) {
            double* resultRow = &resultData[i * p];
            for (int k = 0; k < m; ++k) {
                double aik = data_[i * m + k];
                if (aik != 0.0) {
                    const double* otherRow = &other.data_[k * p];
                    for (int j = 0; j < p; ++j) {
                        resultRow[j] += aik * otherRow[j];
                    }
                }
            }
        }
    }
    return result;
}

// ==================== 标量乘法 ====================
Matrix Matrix::operator*(double scalar) const {
    if (empty()) return Matrix();
    Matrix result(rows, cols);
    const int totalElements = rows * cols;
    double* resultData = result.data_;
    for (int i = 0; i < totalElements; ++i) {
        resultData[i] = data_[i] * scalar;
    }
    return result;
}

// ==================== 全局标量乘法 ====================
Matrix operator*(double scalar, const Matrix& mat) {
    return mat * scalar;
}

// ==================== 矩阵转置 ====================
Matrix Matrix::transpose() const {
    if (empty()) return Matrix();
    Matrix result(cols, rows);
    double* resultData = result.data_;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            resultData[j * rows + i] = data_[i * cols + j];
        }
    }
    return result;
}

// ==================== 行广播 ====================
Matrix Matrix::broadcastRows(int targetRows) const {
    if (targetRows <= 0) {
        throw std::invalid_argument("目标矩阵的行数必须为正数");
    }
    if (rows != 1 && rows != targetRows) {
        throw std::invalid_argument("无法从指定行数进行广播扩展:" +
            std::to_string(rows) + " to " + std::to_string(targetRows));
    }
    if (rows == targetRows) return Matrix(*this);
    Matrix result(targetRows, cols);
    double* resultData = result.data_;
    for (int i = 0; i < targetRows; ++i) {
        for (int j = 0; j < cols; ++j) {
            resultData[i * cols + j] = data_[j];
        }
    }
    return result;
}

// ==================== 哈达玛积 ====================
Matrix Matrix::hadamard(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for Hadamard product");
    }
    if (empty()) return Matrix();
    Matrix result(rows, cols);
    const int totalElements = rows * cols;
    double* resultData = result.data_;
    for (int i = 0; i < totalElements; ++i) {
        resultData[i] = data_[i] * other.data_[i];
    }
    return result;
}

// ==================== apply函数实现 ====================
Matrix Matrix::apply(std::function<double(double)> func) const {
    if (empty()) return Matrix();
    Matrix result(rows, cols);
    const int totalElements = rows * cols;
    double* resultData = result.data_;
    for (int i = 0; i < totalElements; ++i) {
        resultData[i] = func(data_[i]);
    }
    return result;
}

// ==================== 打印矩阵 ====================
void Matrix::print() const {
    if (empty()) {
        std::cout << "Empty matrix (0x0)" << std::endl;
        return;
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::fixed << std::setw(10) << std::setprecision(6) << data_[i * cols + j] ;
            if (j!= cols-1) {
                std::cout << " | ";
            }
        }
        if (i != rows - 1 || rows > 1) {
            std::cout << std::endl;
        }
    }
}