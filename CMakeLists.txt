cmake_minimum_required(VERSION 2.8)

project(sketch_nn)

option(WITH_CUDA "Use CUDA" ON)

if(WITH_CUDA)
    message("-- Compile with CUDA")
    set(CMAKE_PREFIX_PATH "/home/server/MaJing/thirdparty/pytorch_opencv/pytorch/torch") # Path to libtorch CUDA
    list(APPEND CUDA_NVCC_FLAGS "-D__CUDA_NO_HALF_OPERATORS__")
    message("-- nvcc ${CUDA_NVCC_FLAGS}")
    add_definitions(-DWITH_CUDA)
else()
    message("-- Compile with CPU")
    set(CMAKE_PREFIX_PATH "/home/server/MaJing/thirdparty/libtorch_cpu/libtorch") # Path to libtorch CPU
endif()

# https://stackoverflow.com/questions/2463113/g-c0x-enum-class-compiler-warnings
set(CMAKE_CXX_FLAGS "-fopenmp -Wreturn-type -Wno-deprecated-declarations ${CMAKE_CXX_FLAGS}")

#set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c89")


#link_directories(${CMAKE_SOURCE_DIR}/dependencies/lib)

# Torch
find_package(Torch REQUIRED)
if(TORCH_FOUND)
    message("-- Torch @ ${TORCH_INCLUDE_DIRS}")
    include_directories(${TORCH_INCLUDE_DIRS})
endif()

# BOOST
find_package(Boost COMPONENTS system filesystem REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message("-- cv @ ${OpenCV_INCLUDE_DIRS}")
    include_directories(${OpenCV_INCLUDE_DIRS})
endif(OpenCV_FOUND)

find_package(OpenMP)


# local dependencies
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common/)
# CUDA
if(WITH_CUDA)
    find_package(CUDA REQUIRED)
    file(GLOB source ./src/*.cpp ./src/*.cu)
    # Train
    cuda_add_executable(train train.cpp ${source})
    target_link_libraries(train "${CUDA_LIBRARIES}")

    # Test
    cuda_add_executable(test test.cpp ${source})
    target_link_libraries(test "${CUDA_LIBRARIES}")

    # Predict
    cuda_add_executable(predict predict.cpp ${source})
    target_link_libraries(predict "${CUDA_LIBRARIES}")
else()
    file(GLOB source ./src/*.cpp)
    # Train
    add_executable(train train.cpp ${source})
    # Test
    add_executable(test test.cpp ${source})
    # Predict
    add_executable(predict predict.cpp ${source})
endif()

target_link_libraries(train "${OpenCV_LIBRARIES}" "${Boost_LIBRARIES}" "${TORCH_LIBRARIES}")
target_link_libraries(test "${OpenCV_LIBRARIES}" "${Boost_LIBRARIES}" "${TORCH_LIBRARIES}")
target_link_libraries(predict "${OpenCV_LIBRARIES}" "${Boost_LIBRARIES}" "${TORCH_LIBRARIES}")


if(OpenMP_CXX_FOUND)
    target_link_libraries(train OpenMP::OpenMP_CXX)
    target_link_libraries(test OpenMP::OpenMP_CXX)
endif()



