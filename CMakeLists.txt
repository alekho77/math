cmake_minimum_required(VERSION 3.10)
project(mathlib LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Set the output directory for all executable targets
if(NOT DEFINED CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Bin/${CMAKE_BUILD_TYPE})
endif()

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "")
endif()

string(FIND "${CMAKE_CXX_FLAGS}" "-D_GLIBCXX_USE_CXX11_ABI=1" flag_index)
if(flag_index EQUAL -1)
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
endif()

# Check if D_GLIBCXX_USE_CXX11_ABI is set and set the corresponding Conan setting
if(DEFINED D_GLIBCXX_USE_CXX11_ABI)
    if(D_GLIBCXX_USE_CXX11_ABI STREQUAL "0")
        set(LIBCXX_SETTING "compiler.libcxx=libstdc++")
    else()
        set(LIBCXX_SETTING "compiler.libcxx=libstdc++11")
    endif()
else()
    # Default to libstdc++11 if the flag is not defined
    set(LIBCXX_SETTING "compiler.libcxx=libstdc++11")
endif()

execute_process(
    COMMAND conan install . --build=missing -s ${LIBCXX_SETTING} -s compiler.cppstd=${CMAKE_CXX_STANDARD} -s build_type=${CMAKE_BUILD_TYPE} -if=${CMAKE_BINARY_DIR}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error_output
)

if(NOT result EQUAL 0)
    message(FATAL_ERROR "Conan install failed: ${error_output}")
else()
    message(STATUS "Conan install succeeded: ${output}")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_BINARY_DIR}")

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
