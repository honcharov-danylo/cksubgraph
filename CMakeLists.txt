cmake_minimum_required(VERSION 4.2)
project(cksubgraph C)

set(CMAKE_C_STANDARD 11)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)

add_library(cksubgraph STATIC cksubgraph.c)

target_link_libraries(cksubgraph
        PRIVATE
        Python3::Module
        Python3::NumPy
)