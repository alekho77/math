cmake_minimum_required(VERSION 3.10)

project(mathlib_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

enable_testing()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_BINARY_DIR}")

find_package(GTest REQUIRED)
find_package(Boost REQUIRED COMPONENTS iostreams)

include_directories(${CMAKE_SOURCE_DIR}/mathlib)
include_directories(${GTest_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

# Find all test source files
file(GLOB TEST_SOURCES "*.cpp")

# Add an executable for the tests
add_executable(mathlib_tests ${TEST_SOURCES})
target_link_libraries(mathlib_tests PRIVATE mathlib GTest::gtest_main GTest::gtest Boost::iostreams)

# Register the tests
add_test(NAME mathlib_tests COMMAND mathlib_tests)
