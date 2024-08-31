# Compact Math and Neural Network Library

This repository contains a compact and versatile C++ library designed for quick integration into projects that require mathematical computations and neural network capabilities. The library is particularly useful for testing concepts during the early stages of development.

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

- **Approximation and Interpolation**:
  - Functions for approximating functions and interpolating data points.
  
- **Utilities**:
  - Includes helper functions and utilities for managing mathematical constants, static indices, and template metaprogramming.

## Installation

To include this library in your project, simply clone the repository and add it as a subfolder to your project. Use CMake to integrate it into your build system.

```bash
git clone git@github.com:alekho77/math.git your-project/libs/compact-math-lib
```

In your CMakeLists.txt:

```cmake
add_subdirectory(libs/compact-math-lib)
target_link_libraries(your_project_name PRIVATE compact-math-lib)
```

## Usage

### Example: Solving Differential Equations

```cpp
#include "diffequ.h"

// Define your differential equation y' = f(x, y)
Extended __fastcall myFunc(Extended& x, Extended& y) {
    return y - x*x + 1;
}

int main() {
    Extended initialX = 0.0;
    Extended initialY = 0.5;
    Extended finalX = 2.0;
    Extended eps = 0.001;

    Extended result = SolEiler(&myFunc, initialX, initialY, finalX, 100, eps);
    std::cout << "Solution: " << result << std::endl;
}
```

### Example: Training a Neural Network

```cpp
#include "nnetwork.h"
#include "bp_trainer.h"

int main() {
    using Network = neural_network<float>;
    Network net({3, 5, 1}); // A simple 3-5-1 network

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