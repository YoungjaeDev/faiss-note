# CMakeLists.txt for the projection test
# build with: mkdir build && cd build && cmake .. && make -j4 && ./projection_test

cmake_minimum_required(VERSION 3.15)

project(
    faiss-cpp
    VERSION 1.0.0
    LANGUAGES CXX
    DESCRIPTION "Faiss C++"
)

# CXX Flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Naming the executable
set(EXECUTABLE_NAME "faiss-cpp")

# Find_CUDA
find_package(CUDA REQUIRED)

include_directories(${CUDA_TOOLKIT_ROOT_DIR}/targets/${CMAKE_HOST_SYSTEM_PROCESSOR}-linux/include)

add_executable(${EXECUTABLE_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/5_gpu_load_test.cpp)

target_link_libraries(${EXECUTABLE_NAME} PRIVATE faiss )

install (FILES ${CMAKE_CURRENT_SOURCE_DIR}/index.faiss DESTINATION ${CMAKE_CURRENT_BINARY_DIR})