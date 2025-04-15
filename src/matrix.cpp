#include "../includes/matrix.hpp"
#include <algorithm>
#include <random>
#include <stdexcept>
#include <cassert>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace matmul {
    constexpr size_t CACHE_LINE_SIZE = 64;
    constexpr size_t ALIGNMENT = CACHE_LINE_SIZE / sizeof(int); // 16 ints for 64 bytes

    Matrix::Matrix(int r, int c, bool align) : m_rows(r), m_cols(c) {
        if (r <= 0 || c <= 0) {
            throw std::invalid_argument("Matrix dimensions must be positive");
        }
        m_row_stride = align ? ((c + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT : c;
        m_data.resize(static_cast<size_t>(r) * m_row_stride, 0);
    }

    void Matrix::fill_matrix() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 99);
        for (int i = 0; i < m_rows; ++i) {
            for (int j = 0; j < m_cols; ++j) {
                at(i, j) = dis(gen);
            }
        }
    }

    int Matrix::get_rows() const 
    { 
        return m_rows; 
    }

    int Matrix::get_cols() const { 
        return m_cols; 
    }

    size_t Matrix::row_stride() const {
        return m_row_stride;
    }

    std::vector<int> Matrix::get_data() const {
        std::vector<int> result(static_cast<size_t>(m_rows) * m_cols);
        for (int i = 0; i < m_rows; ++i) {
            for (int j = 0; j < m_cols; ++j) {
                result[i * m_cols + j] = at(i, j);
            }
        }
        return result;
    }

    int& Matrix::at(int i, int j) {
        if (i < 0 || i >= m_rows || j < 0 || j >= m_cols) {
            throw std::out_of_range("Matrix index out of bounds");
        }
        return m_data[static_cast<size_t>(i) * m_row_stride + j];
    }

    const int& Matrix::at(int i, int j) const {
        if (i < 0 || i >= m_rows || j < 0 || j >= m_cols) {
            throw std::out_of_range("Matrix index out of bounds");
        }
        return m_data[static_cast<size_t>(i) * m_row_stride + j];
    }

    Matrix matmul_naive(const Matrix& A, const Matrix& B) {
        if (A.get_cols() != B.get_rows()) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication");
        }
        Matrix result(A.get_rows(), B.get_cols());
        for (int i = 0; i < A.get_rows(); i++) {
            for (int j = 0; j < B.get_cols(); j++) {
                int sum = 0;
                for (int k = 0; k < A.get_cols(); k++) {
                    sum += A.at(i, k) * B.at(k, j);
                }
                result.at(i, j) = sum;
            }
        }
        return result;
    }

    Matrix matmul_blocked(const Matrix& A, const Matrix& B, int block_size, int num_threads) {
        if (A.get_cols() != B.get_rows()) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication");
        }

        int n = A.get_rows(); 
        assert(A.get_cols() == n && B.get_cols() == n && "Optimized for square matrices");

        Matrix C(n, n);
        const size_t stride_a = A.row_stride();
        const size_t stride_b = B.row_stride();
        const size_t stride_c = C.row_stride();

        block_size = std::min(block_size, n);
        block_size = (block_size / ALIGNMENT) * ALIGNMENT;
        if (block_size == 0) block_size = ALIGNMENT;

        // Disable parallelism for small matrices
        if (n < 512) num_threads = 1;

        
        #ifdef _OPENMP
        int max_threads = omp_get_max_threads();
        #else
        int max_threads = 4; 
        #endif
        num_threads = std::min(num_threads, max_threads);
        num_threads = std::max(num_threads, 1); //] at least 1 thread

        #ifdef _OPENMP
        #pragma omp parallel for collapse(2) schedule(dynamic) num_threads(num_threads)
        #endif
        for (int i = 0; i < n; i += block_size) {
            for (int j = 0; j < n; j += block_size) {
                // Zero out block in C
                for (int ii = i; ii < std::min(i + block_size, n); ++ii) {
                    for (int jj = j; jj < std::min(j + block_size, n); ++jj) {
                        C.at(ii, jj) = 0;
                    }
                }
                for (int k = 0; k < n; k += block_size) {
                    int i_max = std::min(i + block_size, n);
                    int j_max = std::min(j + block_size, n);
                    int k_max = std::min(k + block_size, n);

                    // Optimized kernel
                    for (int ii = i; ii < i_max; ++ii) {
                        for (int kk = k; kk < k_max; ++kk) {
                            int a_ik = A.at(ii, kk);
                            for (int jj = j; jj < j_max; ++jj) {
                                C.at(ii, jj) += a_ik * B.at(kk, jj);
                            }
                        }
                    }
                }
            }
        }
        return C;
    }

    static void matmul_recursive_helper(const Matrix& A, const Matrix& B, Matrix& C, 
                                       int rA, int cA, int rB, int cB, int rC, int cC, int size) {
        if (size <= 64) {
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    int sum = 0;
                    for (int k = 0; k < size; k++) {
                        sum += A.at(rA + i, cA + k) * B.at(rB + k, cB + j);
                    }
                    C.at(rC + i, cC + j) = sum;
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