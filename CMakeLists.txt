cmake_minimum_required(VERSION 3.10)
project(mathlib LANGUAGES CXX)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Define the main path to the header files
include_directories(${CMAKE_SOURCE_DIR}/mathlib)

# This is currently a header-only library, but sources can be added in the future
# Build a static library
add_library(mathlib INTERFACE)
target_include_directories(mathlib INTERFACE ${CMAKE_SOURCE_DIR}/mathlib)

# In the future, if cpp files are added, replace INTERFACE with STATIC or SHARED
# add_library(mathlib STATIC ${SOURCES})

# Add the subdirectory with the tests
add_subdirectory(mathlib-test/mathlib-test)
