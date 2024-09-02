cmake_minimum_required(VERSION 3.10)

project(mathlib_test LANGUAGES CXX)

find_package(GTest REQUIRED)
find_package(Boost REQUIRED COMPONENTS iostreams)

include_directories(${GTest_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

# Find all test source files
file(GLOB TEST_SOURCES "*.cpp")

# Add an executable for the tests
add_executable(mathlib_tests ${TEST_SOURCES})
target_link_libraries(mathlib_tests PRIVATE mathlib GTest::gtest_main GTest::gtest Boost::iostreams)

# Register the tests
add_test(NAME mathlib_tests COMMAND mathlib_tests)
