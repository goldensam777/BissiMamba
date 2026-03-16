cmake_minimum_required(VERSION 3.18)
project(k-mamba-bench)

add_executable(bench_paper bench_paper.c)

target_include_directories(bench_paper PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../optimatrix/include
)

target_link_libraries(bench_paper PRIVATE
    k-mamba
    k-mamba-cpu
    optimatrix::optimatrix
    m
)

target_link_options(bench_paper PRIVATE -no-pie)
target_compile_options(bench_paper PRIVATE -O3 -mavx2)
