cmake_minimum_required(VERSION 3.10)
project(matmul_cache_aware_oblivious LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(matmul 
    src/main.cpp 
    src/matrix.cpp 
    includes/matrix.hpp 
    includes/cache_info.h
)
add_definitions(-DNOMINMAX)
target_include_directories(matmul PRIVATE includes)