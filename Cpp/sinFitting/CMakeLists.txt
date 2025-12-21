cmake_minimum_required(VERSION 3.16)

project(Math.Sandbox  C CXX ASM)


add_executable(Math.Sandbox)
add_library(CommonCode INTERFACE)

add_subdirectory(Application)
add_subdirectory(Dependencies)

target_link_libraries(Math.Sandbox PUBLIC 
    CommonCode
    matplot
    nlopt
    Eigen3::Eigen
)