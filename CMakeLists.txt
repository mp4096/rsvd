cmake_minimum_required(VERSION 3.5)

if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif(POLICY CMP0048)

project(Rsvd VERSION 0.1.0 LANGUAGES CXX)

include(CTest)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_EXTENSIONS OFF)
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(RSVD_BUILD_EXAMPLES "Build the examples" ON)
option(RSVD_MARCH_NATIVE "Compile for the native microarchitecture" OFF)
option(RSVD_OPENMP "Use OpenMP" ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(Eigen3 REQUIRED)

set(
  HEADERS
  include/rsvd/Constants.hpp
  include/rsvd/ErrorEstimators.hpp
  include/rsvd/GramSchmidt.hpp
  include/rsvd/Prelude.hpp
  include/rsvd/RandomizedRangeFinder.hpp
  include/rsvd/RandomizedSvd.hpp
  include/rsvd/StandardNormalRandom.hpp
)

include_directories(
    "include"
    ${Eigen3_INCLUDE_DIR}
)

# Flags
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Werror -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DEIGEN_NO_DEBUG")

# Examples
if(RSVD_BUILD_EXAMPLES)
  add_executable(example_benchmark "examples/Benchmark/main.cpp")
  add_executable(example_simple_usage "examples/SimpleUsage/main.cpp")
endif()

# -march=native
if(RSVD_MARCH_NATIVE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
endif()

# -fopenmp
if(RSVD_OPENMP)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp")
endif()

# Unit Tests
set(
  TESTS
  test/ErrorEstimators.cpp
  test/GramSchmidt.cpp
  test/RandomizedRangeFinder.cpp
  test/RandomizedSvd.cpp
  test/StandardNormalRandom.cpp
)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory("external/googletest")
  include_directories("external/googletest/include")
  add_executable(rsvd_test ${TESTS} ${HEADERS})
  target_link_libraries(rsvd_test PRIVATE gtest gtest_main)
  add_test(Rsvd rsvd_test)
endif()
