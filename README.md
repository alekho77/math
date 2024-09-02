# Mathlib

This repository contains a compact and versatile C++ library designed for quick integration into projects that require mathematical computations and neural network capabilities. The library is particularly useful for testing concepts during the early stages of development.
Mathlib is a header-only C++ library designed for mathematical operations. It is designed to be lightweight and easy to integrate into other projects. This README provides an overview of how to build, test, and use the library, along with prerequisites and configuration options.

## Features

- **Numerical Methods**: 
  - **Differential Equations**: Includes Euler's method for solving ordinary differential equations.
  - **Integration**: Implements numerical integration methods like Simpson's rule.
  - **Derivatives**: Supports numerical differentiation with partial derivatives.

- **Linear Algebra**:
  - **Matrix Operations**: Includes basic operations such as addition, subtraction, multiplication, and transposition of matrices.
  - **Linear Systems**: Provides tools for solving linear systems of equations.
  - **Nonlinear Systems**: Supports solving nonlinear systems of equations.

- **Neural Networks**:
  - **Neuron Definition**: Classes for representing individual neurons with customizable activation functions.
  - **Neural Networks**: Classes for building, training, and evaluating neural networks.
  - **Training**: Backpropagation-based training methods with support for customizable learning rate and momentum.
  - **Training Sets**: Tools for managing and using datasets during neural network training.
  
  In this library, the neural network is represented as a template. This means that the neurons are essentially optimized by the compiler into a function of the form y = f(x1, x2, ..., xn) without any for-loops. Similarly, the entire network is optimized through templates and recursion, effectively reducing to a function of the form <x1, x2, ..., xm> = F(x1, x2, ..., xn) without any for-loops, where `n`  is the number of inputs and `m` is the number of outputs.

- **Approximation and Interpolation**:
  - Functions for approximating functions and interpolating data points.
  
- **Utilities**:
  - Includes helper functions and utilities for managing mathematical constants, static indices, and template metaprogramming.

## Prerequisites

To build and test Mathlib, you need the following tools installed on your system:

- **CMake** (version 3.10 or higher)
- **Conan** (a package manager for C++ libraries)
- **GCC** (or another compatible C++ compiler)
- **Boost** (specifically the iostreams component) This library will be installed via Conan.
- **Google Test (GTest)** (for unit testing) This library will be installed via Conan.

Make sure these tools are installed and available in your system's PATH.

## Installation

### Clone the repository

To include this library in your project, simply clone the repository and add it as a subfolder to your project. Use CMake to integrate it into your build system.

```bash
git clone git@github.com:alekho77/math.git your-project/libs/mathlib
```

In your CMakeLists.txt:

```cmake
add_subdirectory(libs/mathlib)
target_link_libraries(your_project_name PRIVATE mathlib)
```

### Configure, build and test the library

You can configure and build the library using CMake. The following commands will configure the project and build it:

```bash
cmake -S . -B build
cmake --build build --target build_and_test
```

## Configuration Options

Mathlib provides several configuration options that can be set during the CMake configuration step:

- **CMAKE_BUILD_TYPE**: Sets the build type (`Debug`, `Release`, etc.). If not specified, it defaults to `Debug`.

- **CMAKE_CXX_STANDARD**: Sets the C++ standard to use. The default is C++17.

- **CMAKE_RUNTIME_OUTPUT_DIRECTORY**: Defines the directory where the executable targets will be placed. If not defined, the executables are placed in `build/Bin/<BuildType>`.

- **D_GLIBCXX_USE_CXX11_ABI**: Defines the ABI version to use (`1` for the C++11 ABI, `0` for the old ABI). This is automatically detected and set in the CMake configuration.

## Conan Integration

Mathlib uses Conan to manage dependencies. The required dependencies are specified in the `conanfile.txt`. During the configuration step, Conan is automatically invoked to install the necessary dependencies by running something like:

```bash
conan install . --build=missing -s compiler.cppstd=17 -s compiler.libcxx=libstdc++11 -if=build
```

If you need to adjust the Conan settings, they are automatically determined based on the CMake configuration.

## Usage

### Adding Mathlib to Your Project

To add Mathlib to your project, include the header files in your source code:

```cpp
#include "mathlib/some_mathlib_header_file.h"
```

Since Mathlib is currently a header-only library, no linking is required. If source files are added in the future, the library will be built as a static or shared library, and you will need to link against `mathlib`.

### Example: Training a Neural Network

```cpp
#include "mathlib/bp_trainer.h"

int main() {
    // The input layer of the network consists of two double numbers.
    using InputLayer = input_layer<double, 2>;
    // A standard sigmoid neuron with a bias and two synapses.
    using Neuron1 = neuron<double, 2>;
    // Another neuron, identical but without a bias.
    using Neuron2 = neuron<double, 2, NOBIAS<double>>;
    // An auxiliary object for indexing from 0 to 1 (based on the number of neurons and their synapses).
    using IndexPack = index_pack<0, 1>;
    // Connects everything to everything between two layers, each consisting of two elements.
    using Map1 = make_type_pack<IndexPack, 2>::type;
    // Connects the final layer to each neuron in the hidden layer.
    using Map2 = type_pack<IndexPack>;
    // Create a hidden layer that consists of neurons of the second type (without bias) and connect it to the input layer,
    // where each synapse of each neuron is connected to each input element
    using HiddenLayer = nnetwork<InputLayer, std::tuple<Neuron2, Neuron2>, Map1>;
    // Create the final network consisting of the input layer, the hidden layer, and a single output neuron with a bias
    using Network = nnetwork<HiddenLayer, std::tuple<Neuron1>, Map2>;

    Network net;

    auto trainer = make_bp_trainer(net);
    trainer.set_learning_rate(0.01);
    trainer.set_momentum(0.9);

    // Train the network with some data...
    
    return 0;
}
```

## Contributing

Contributions are welcome! If you have suggestions for improvements or find any issues, feel free to create a pull request or open an issue on GitHub.

## License

This library is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.