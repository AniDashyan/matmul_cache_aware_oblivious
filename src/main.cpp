#include <format>
#include <string>
#include <utility>
#include <thread>
#include "../includes/matrix.hpp"
#include "../includes/cache_info.h"
#include "../includes/kaizen.h"

// #define NOMINMAX

std::pair<int, int> parse_args(int argc, char** argv) {
    zen::cmd_args args(argv, argc);
    int size = 1024;
    int num_threads = std::thread::hardware_concurrency();
    if (!args.is_present("--size")|| !args.is_present("--threads")) {
        zen::log(zen::color::yellow("No --size or --threads provided. Using default values: "));
        return {size, num_threads};
    } else {
        size = std::stoi(args.get_options("--size")[0]);
        num_threads = std::stoi(args.get_options("--threads")[0]);
    }
    return {size, num_threads};
}

int main(int argc, char** argv) {
    
    auto [ size, num_threads ] = parse_args(argc, argv);
    // Compute blockSize from cache info
    CacheInfo info = get_cache_info();
    int blockSize;
    if (info.l1d_size > 0 && info.line_size > 0) {
        int maxElements = info.l1d_size / sizeof(int);
        blockSize = static_cast<int>(std::sqrt(maxElements / 3)); // For A, B, C
        blockSize = (blockSize / (info.line_size / sizeof(int))) * (info.line_size / sizeof(int));
        if (blockSize <= 0) blockSize = info.line_size / sizeof(int);
    } else {
        blockSize = 64; // 64 ints
        std::cerr << "Warning: Could not retrieve cache info, using blockSize = 64\n";
    }

    matmul::Matrix A(size, size);
    matmul::Matrix B(size, size);
    matmul::Matrix C(size, size);

    A.fill_matrix();
    B.fill_matrix();

    zen::timer t;
    try {  
        t.start();
        matmul::Matrix C = matmul::matmul_naive(A, B);
        t.stop();
        auto time_naive = t.duration_string();

        t.start();
        matmul::Matrix D = matmul::matmul_blocked(A, B, blockSize, num_threads);
        t.stop();
        auto time_blocked = t.duration_string();

        t.start();
        matmul::Matrix E = matmul::matmul_recursive(A, B);
        t.stop();
        auto time_recursive = t.duration_string();

        zen::log(std::format("Matrix Multiplication Performance (size = {}x{}):", size, size));
        zen::log("----------------------------------------");
        zen::log(std::format("{:<25}Time", "Method"));
        zen::log("----------------------------------------");
        zen::log(std::format("{:<25}{}", "Naive", time_naive));
        zen::log(std::format("{:<25}{}", "Blocked (blockSize=" + std::to_string(blockSize) + ")", time_blocked));
        zen::log(std::format("{:<25}{}", "Recursive", time_recursive));
        zen::log("----------------------------------------");

        zen::log("Cache Information:");
        zen::log(std::format("L1D Size: {} bytes", info.l1d_size));
        zen::log(std::format("Line Size: {} bytes", info.line_size));
        zen::log(std::format("Block Size: {} ints", blockSize));
        if (num_threads > 0) {
            zen::log(std::format("Threads Used: {}", num_threads));
        } else {
            #ifdef _OPENMP
            zen::log(std::format("Threads Used: {} (default)", std::thread::hardware_concurrency()));
            #else
            zen::log("Threads Used: 1 (no OpenMP)");
            #endif
        }
    }
    catch (std::exception& e) {
        zen::log(zen::color::red(e.what()));
    }
    return 0;
}