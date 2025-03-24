#include <iostream>
#include <iomanip>
#include <string>
#include <format>
#include "../includes/matrix.hpp"
#include "../includes/cache_info.h"
#include "../includes/kaizen.h"

#define NOMINMAX

int main(int argc, char** argv) {
    zen::cmd_args args(argv, argc);
    int rows = 1000, cols = 2000;
    if (!args.is_present("--row") || !args.is_present("--col")) {
        zen::log(zen::color::yellow("either --row or --col, or none of the options is not provided. Using the default value: " + std::to_string(rows) + "x" + std::to_string(cols)));
    } else {
        rows = std::stoi(args.get_options("--row")[0]);
        cols = std::stoi(args.get_options("--col")[0]);
    }

    // Compute blockSize from cache info
    CacheInfo info = get_cache_info();
    int blockSize;
    if (info.l1d_size > 0 && info.line_size > 0) {
        int maxElements = info.l1d_size / sizeof(double);
        blockSize = static_cast<int>(std::sqrt(maxElements));
        blockSize = (blockSize / info.line_size) * info.line_size;
        if (blockSize <= 0) blockSize = info.line_size;
    } else {
        blockSize = 64;
        std::cerr << "Warning: Could not retrieve cache info, using blockSize = 64\n";
    }

    matmul::Matrix A(rows, cols);
    matmul::Matrix B(cols, rows);
    matmul::Matrix C(rows, rows);

    A.fill_matrix();
    B.fill_matrix();

    zen::timer t;
    try {  
        t.start();
        matmul::Matrix C = matmul::matmul_naive(A, B);
        t.stop();
        auto time_naive = t.duration_string();

        t.start();
        matmul::Matrix D = matmul::matmul_blocked(A, B, blockSize);
        t.stop();
        auto time_blocked = t.duration_string();

        t.start();
        matmul::Matrix E = matmul::matmul_recursive(A, B);
        t.stop();
        auto time_recursive = t.duration_string();

        zen::log(std::format("Matrix Multiplication Performance (size = {}x{}):", rows, cols));
        zen::log("----------------------------------------");
        zen::log(std::format("{:<25}Time (ms)", "Method"));
        zen::log("----------------------------------------");
        zen::log(std::format("{:<25}{}", "Naive", time_naive));
        zen::log(std::format("{:<25}{}", "Blocked (blockSize=" + std::to_string(blockSize) + ")", time_blocked));
        zen::log(std::format("{:<25}{}", "Recursive", time_recursive));
        zen::log("----------------------------------------");

        // Display cache info using std::format
        zen::log("Cache Information:");
        zen::log(std::format("L1D Size: {} bytes", info.l1d_size));
        zen::log(std::format("Line Size: {} bytes", info.line_size));
    } 
    catch (std::exception& e) {
        zen::log(zen::color::red(e.what()));
    }
    return 0;
}