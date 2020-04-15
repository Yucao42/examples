cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
set(CMAKE_PREFIX_PATH /proj/rdma-prefetch-PG0/groups/eric/libtorch/)
message(STATUS ${CMAKE_PREFIX_PATH})
project(mnist)

# set(CMAKE_SKIP_RPATH on)
find_package(Torch REQUIRED)
set(CMAKE_VERBOSE_MAKEFILE on)
find_package(CUDA)
#find_package(OpenMP)
#if (OPENMP_FOUND)
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()

option(DOWNLOAD_MNIST "Download the MNIST dataset from the internet" ON)
if (DOWNLOAD_MNIST)
  message(STATUS "Downloading MNIST dataset")
  execute_process(
    COMMAND python ${CMAKE_CURRENT_LIST_DIR}/../tools/download_mnist.py
      -d ${CMAKE_BINARY_DIR}/data
    ERROR_VARIABLE DOWNLOAD_ERROR)
  if (DOWNLOAD_ERROR)
    message(FATAL_ERROR "Error downloading MNIST dataset: ${DOWNLOAD_ERROR}")
  endif()
endif()

# PAPI needs CUPTI
# Set tje CUDA_HOME and PAPI_HOME environment variables first
set(PAPI_CUDA_ROOT $ENV{CUDA_HOME})
set(PAPI_CUPTI_ROOT $ENV{CUDA_HOME}/extras/CUPTI)
set(CUDRV_ROOT $ENV{CUDA_HOME})


#SET(CMAKE_EXE_LINKER_FLAGS "-Wl,--enable-new-dtags")
#set(RTFLAGS "-Wl,-runpath,")

SET(ENV{LD_LIBRARY_PATH} "/users/eric_nyu/eric/libtorch/lib;/users/eric_nyu/eric/conda3/envs/vas/lib;/usr/local/cuda/lib64;${PAPI_CUDA_ROOT}/lib;$ENV{LD_LIBRARY_PATH}")


include_directories($ENV{PAPI_HOME}/src/testlib $ENV{PAPI_HOME}/src ${PAPI_CUDA_ROOT}/include ${PAPI_CUPTI_ROOT}/include $ENV{PAPI_HOME}/src/validation_tests)
#add_executable(mnist mnist.cpp)
message(STATUS $ENV{LD_LIBRARY_PATH})
link_directories( $ENV{LD_LIBRARY_PATH} $ENV{PAPI_HOME}/src/libpfm4/lib ${PAPI_CUPTI_ROOT}/lib64)
cuda_add_executable(mnist mnist.cpp)
target_compile_features(mnist PUBLIC cxx_range_for)
target_link_libraries(mnist ${TORCH_LIBRARIES} 
    $ENV{PAPI_HOME}/src/libpapi.a 
    $ENV{PAPI_HOME}/src/testlib/libtestlib.a
    cudart cuda cupti pfm dl)
# 

add_compile_options(${RTFLAGS})
include_directories($ENV{PAPI_HOME}/src/testlib $ENV{PAPI_HOME}/src ${PAPI_CUDA_ROOT}/include ${PAPI_CUPTI_ROOT}/include $ENV{PAPI_HOME}/src/validation_tests)
# message(STATUS ${PAPI_CUDA_ROOT})
link_directories(helloworld ${PAPI_CUDA_ROOT}/lib64 ${CMAKE_PREFIX_PATH}/lib)
message(STATUS ${RTFLAGS})
cuda_add_executable(helloworld HelloWorld.cu)
# target_compile_features(helloworld PUBLIC cxx_range_for)
target_link_libraries(helloworld
    $ENV{PAPI_HOME}/src/libpapi.a 
    $ENV{PAPI_HOME}/src/testlib/libtestlib.a
    cudart cuda dl)
