cmake_minimum_required(VERSION 3.18)
project(k-mamba-cuda VERSION 1.0.0 LANGUAGES CUDA)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)

# Mamba-specific CUDA kernels (scan kernels are now in optimatrix/cuda/)
set(CUDA_SRCS
    mamba_scan.cu
    # mamba_block.cu          (à implémenter)
    # mamba_block_backward.cu (à implémenter)
    # muon.cu                 (à implémenter)
    # muon_mamba.cu           (à implémenter)
)

add_library(k-mamba-cuda STATIC ${CUDA_SRCS})

target_include_directories(k-mamba-cuda
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../optimatrix/include>
)

target_compile_options(k-mamba-cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_70>
)

# Link against optimatrix-cuda (fournit om_scan1d_forward/backward)
target_link_libraries(k-mamba-cuda
    PUBLIC  optimatrix-cuda
    PRIVATE cuda cudart
)
