cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(host_launcher)
set(PROJECT_NAME host_launcher)

set(PYTHON_INCLUDE /home/cjmcv/anaconda3/envs/venv/include/python3.11)
set(PYBIND11_INCLUDE /home/cjmcv/anaconda3/envs/venv/lib/python3.11/site-packages/pybind11/include)
set(pybind11_DIR /home/cjmcv/anaconda3/envs/venv/lib/python3.11/site-packages/pybind11/share/cmake/pybind11)
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