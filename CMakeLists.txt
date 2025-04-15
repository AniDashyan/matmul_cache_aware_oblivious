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

# Find OpenMP
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    # Add OpenMP flags and libraries
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    # Define NO_OPENMP if OpenMP is not found
    add_definitions(-DNO_OPENMP)
    message(WARNING "OpenMP not found. Building without parallelization.")
endif()


# Link OpenMP if found
if (OpenMP_CXX_FOUND)
    target_link_libraries(matmul PRIVATE OpenMP::OpenMP_CXX)
endif()


# add_definitions(-DNOMINMAX)
target_include_directories(matmul PRIVATE includes)