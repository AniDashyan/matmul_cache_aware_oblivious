#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <memory>

namespace matmul {
    class Matrix {
        private:
            int m_rows, m_cols;
            std::vector<int> m_data;
            size_t m_row_stride; // For cache-aligned rows   

        public:
            Matrix(int r, int c, bool align = true);

            void fill_matrix();

            int get_rows() const;
            int get_cols() const;
            std::vector<int> get_data() const;
            int& at(int i, int j);
            const int& at(int i, int j) const;

            int* data();
            size_t row_stride() const;
        };

        Matrix matmul_naive(const Matrix& A, const Matrix& B);
        Matrix matmul_blocked(const Matrix& A, const Matrix& B, int block_size, int num_threads);
        Matrix matmul_recursive(const Matrix& A, const Matrix& B);
}

#endif