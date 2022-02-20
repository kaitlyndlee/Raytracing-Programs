cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
project(raytrace VERSION 0.1.0 LANGUAGES C CXX CUDA)
find_package(CUDA)
set (CMAKE_CXX_FLAGS "--std=c++14")
set (CUDA_PROPAGATE_HOST_FLAGS ON)

# Include owl API
set(owl_dir ${PROJECT_SOURCE_DIR}/owl/)
add_subdirectory(${owl_dir} EXCLUDE_FROM_ALL)

# Include custom function to handle ptx compilation
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMake/")
include("nvcuda_compile_ptx")
include("cuda_compile_and_embed")

set(CMAKE_BUILD_DIR "${PROJECT_SOURCE_DIR}/build/")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BUILD_DIR}) 

add_subdirectory(src)