#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>

namespace matmul {
    class Matrix {
        private:
            int m_rows, m_cols;
            std::vector<int> m_data;

        public:
            Matrix(int r, int c);
            ~Matrix();

            void fill_matrix();

            int get_rows() const;
            int get_cols() const;
            std::vector<int> get_data() const;
            int& at(int i, int j);
            const int& at(int i, int j) const;
        };

        Matrix matmul_naive(const Matrix& A, const Matrix& B);
        Matrix matmul_blocked(const Matrix& A, const Matrix& B, int blockSize);
        Matrix matmul_recursive(const Matrix& A, const Matrix& B);
}

#endif