cmake_minimum_required(VERSION 3.10)

# 依赖：
# cuda
# pybind11：pip install pybind11
# avx-512: set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512bw -mavx512bf16")
# c++11：为避免/usr/local的libstdc++.so.6与conda的版本不一致，指定到conda的lib路径里编译。

# *oneapi: https://www.intel.cn/content/www/cn/zh/developer/tools/oneapi/base-toolkit-download.html?packages=oneapi-toolkit&oneapi-toolkit-os=linux&oneapi-lin=apt
## set(CMAKE_PREFIX_PATH "/opt/intel/oneapi/mkl/latest/lib/cmake/mkl;${CMAKE_PREFIX_PATH}")
## find_package(MKL CONFIG REQUIRED)
## target_link_libraries(${PROJECT_NAME} PRIVATE pybind11::module cudart MKL::MKL)
# *openmp: set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
# note: oneapi和openmp在cuda graph里无法使用？可能跟线程的创建和释放有关？


# 设置项目名称
project(host_launcher)
set(PROJECT_NAME host_launcher)

# set(CMAKE_CXX_STANDARD 11)
set(CONDA_ENV /home/cjmcv/anaconda3/envs/venv/)

set(PYTHON_INCLUDE ${CONDA_ENV}/include/python3.11)
set(PYBIND11_INCLUDE ${CONDA_ENV}/lib/python3.11/site-packages/pybind11/include)
set(pybind11_DIR ${CONDA_ENV}/lib/python3.11/site-packages/pybind11/share/cmake/pybind11)
link_directories(${CONDA_ENV}/lib)

# include_directories(${CONDA_ENV}/lib/python3.11/site-packages/nvidia/cuda_runtime/include)
# link_directories(${CONDA_ENV}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib)

option(ENABLE_AVX512 "Enable AVX512" ON)

if(ENABLE_AVX512)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512bw -mavx512bf16 -lstdc++")
    add_definitions(-DENABLE_AVX512)
endif()

# if(ENABLE_OPENMP)
#     find_package(OpenMP)
#     add_definitions(-DENABLE_OPENMP)
# endif()

find_package(pybind11 REQUIRED)
find_package(CUDA ${CUDA_VERSION} REQUIRED EXACT)

include_directories(${PYBIND11_INCLUDE}
                    ${PYTHON_INCLUDE})

# 收集所有的 CUDA 源文件
file(GLOB_RECURSE SRC_LIST 
    "*.cpp"
    "*.cu"
    "operator/*.cpp"
    # Include
    "*.h"
)

# 生成共享库
pybind11_add_module(${PROJECT_NAME} MODULE ${SRC_LIST})

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"  # 根据你的 GPU 计算能力修改
)

target_link_libraries(${PROJECT_NAME} PRIVATE pybind11::module cudart)

# 设置共享库的输出目录
set_target_properties(${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})