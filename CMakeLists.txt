cmake_minimum_required(VERSION 3.10)
project(mathlib LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

# Define the main path to the header files
include_directories(${CMAKE_SOURCE_DIR}/include)

# This is currently a header-only library, but sources can be added in the future
# Build a static library
add_library(mathlib INTERFACE)
target_include_directories(mathlib INTERFACE ${CMAKE_SOURCE_DIR}/include)

# In the future, if cpp files are added, replace INTERFACE with STATIC or SHARED
# add_library(mathlib STATIC ${SOURCES})

# Add the subdirectory with the tests
add_subdirectory(src/mathlib-test)

add_custom_target(build_and_test
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
    COMMAND ctest --output-on-failure
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Building and running tests..."
    USES_TERMINAL
)
