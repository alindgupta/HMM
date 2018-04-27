cmake_minimum_required (VERSION 3.4)

project (HMM)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_EXTENSIONS OFF)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
add_executable (out main.cpp src/hmm.cpp)
target_link_libraries (out Eigen3::Eigen)
