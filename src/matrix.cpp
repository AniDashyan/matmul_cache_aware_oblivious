#include "../includes/matrix.hpp"
#include <algorithm>
#include <random>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace matmul {
    Matrix::Matrix(int r, int c) : m_rows(r), m_cols(c), m_data(r * c, 0) {}

    Matrix::~Matrix() {}

    void Matrix::fill_matrix() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 99);
        for (int i = 0; i < m_rows * m_cols; ++i) {
            m_data[i] = dis(gen);
        }
    }

    int Matrix::get_rows() const {
        return m_rows;
    }

    int Matrix::get_cols() const {
        return m_cols;
    }

    std::vector<int> Matrix::get_data() const {
        return m_data;
    }

    int& Matrix::at(int i, int j) {
        return m_data[i * m_cols + j];
    }

    const int& Matrix::at(int i, int j) const {
        return m_data[i * m_cols + j];
    }

    Matrix matmul_naive(const Matrix& A, const Matrix& B) {
        if (A.get_cols() != B.get_rows()) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication");
        }
        Matrix result(A.get_rows(), B.get_cols());
        for (int i = 0; i < A.get_rows(); i++) {
            for (int j = 0; j < B.get_cols(); j++) {
                for (int k = 0; k < A.get_cols(); k++) {
                    result.at(i, j) += A.at(i, k) * B.at(k, j);
                }
            }
        }
        return result;
    }

    Matrix matmul_blocked(const Matrix& A, const Matrix& B, int block_size) {
        if (A.get_cols() != B.get_rows()) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication");
        }
        Matrix result(A.get_rows(), B.get_cols());
        int n_rows = A.get_rows();
        int n_cols = B.get_cols();
        int n_inner = A.get_cols();

        // Parallelize outer loops with OpenMP
        #pragma omp parallel for collapse(2) if(n_rows * n_cols > 1000000) // Threshold for parallelization
        for (int i = 0; i < n_rows; i += block_size) {
            for (int j = 0; j < n_cols; j += block_size) {
                for (int k = 0; k < n_inner; k += block_size) {
                    int i_max = std::min(i + block_size, n_rows);
                    int j_max = std::min(j + block_size, n_cols);
                    int k_max = std::min(k + block_size, n_inner);
                    for (int ii = i; ii < i_max; ii++) {
                        for (int jj = j; jj < j_max; jj++) {
                            for (int kk = k; kk < k_max; kk++) {
                                result.at(ii, jj) += A.at(ii, kk) * B.at(kk, jj);
                            }
                        }
                    }
                }
            }
        }
        return result;
    }

    static void matmul_recursive_helper(const Matrix& A, const Matrix& B, Matrix& C, 
                                        int rA, int cA, int rB, int cB, int rC, int cC, int size) {
        if (size <= 64) {
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    for (int k = 0; k < size; k++) {
                        C.at(rC + i, cC + j) += A.at(rA + i, cA + k) * B.at(rB + k, cB + j);
                    }
                }
            }
            return;
        }
        int half = size / 2;
        matmul_recursive_helper(A, B, C, rA, cA, rB, cB, rC, cC, half);
        matmul_recursive_helper(A, B, C, rA, cA + half, rB + half, cB, rC, cC, half);
        matmul_recursive_helper(A, B, C, rA, cA, rB, cB + half, rC, cC + half, half);
        matmul_recursive_helper(A, B, C, rA, cA + half, rB + half, cB + half, rC, cC + half, half);
        matmul_recursive_helper(A, B, C, rA + half, cA, rB, cB, rC + half, cC, half);
        matmul_recursive_helper(A, B, C, rA + half, cA + half, rB + half, cB, rC + half, cC, half);
        matmul_recursive_helper(A, B, C, rA + half, cA, rB, cB + half, rC + half, cC + half, half);
        matmul_recursive_helper(A, B, C, rA + half, cA + half, rB + half, cB + half, rC + half, cC + half, half);
    }

    Matrix matmul_recursive(const Matrix& A, const Matrix& B) {
        if (A.get_cols() != B.get_rows()) {
            throw std::runtime_error("Matrix dimensions do not match for multiplication");
        }
        Matrix result(A.get_rows(), B.get_cols());
        matmul_recursive_helper(A, B, result, 0, 0, 0, 0, 0, 0, A.get_rows());
        return result;
    }
}