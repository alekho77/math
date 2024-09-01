cmake_minimum_required(VERSION 3.10)

# Specify the project for tests
project(mathlib_test LANGUAGES CXX)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable testing support
enable_testing()

# Include GTest as an external project using Conan
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

# Specify the path to the library headers
include_directories(${CMAKE_SOURCE_DIR}/mathlib)

# Find all test source files
file(GLOB TEST_SOURCES "*.cpp")

# Add an executable for the tests
add_executable(mathlib_tests ${TEST_SOURCES})

# Link the tests with the library and GTest
target_link_libraries(mathlib_tests PRIVATE mathlib ${CONAN_LIBS})

# Register the tests
add_test(NAME mathlib_tests COMMAND mathlib_tests)
