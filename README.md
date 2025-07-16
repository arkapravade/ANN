# Artificial Neural Network using Deep Learning

A comprehensive implementation of neural networks built from scratch in C++ and enhanced with modern deep learning frameworks in Python.

## Project Overview

This project demonstrates the implementation of Artificial Neural Networks (ANNs) using two distinct approaches:

1. **From-Scratch Implementation**: A complete Multilayered Feed-forward Neural Network (MLFFNN) built entirely in C++
2. **Modern Framework Implementation**: An advanced neural network using Keras in Python, applied to real-world thermodynamic data

## Features

### C++ Implementation
- **Custom Neural Network Framework**: Built entirely from scratch without external ML libraries
- **Flexible Architecture**: Supports any number of input features, output parameters, and hidden layers
- **Configurable Training**: User-defined maximum iterations and test data split
- **Pattern Recognition**: Handles datasets with varying complexity and size
- **Extrapolation Capabilities**: Model can predict values beyond the training dataset

### Python Implementation
- **Advanced Deep Learning**: Utilizes Keras for sophisticated neural network architecture
- **Real-World Application**: Tested on Thermodynamic Steam Table with 250+ data patterns
- **Optimized Architecture**: Three hidden layers with 500 nodes each
- **Modern Techniques**: ReLU activation, Adam optimizer, and MSE loss function

## Technical Specifications

### C++ Version
- **Language**: C++
- **Architecture**: Multilayered Feed-forward Neural Network
- **Implementation**: Complete from-scratch development
- **Features**: Custom backpropagation, gradient descent, and weight optimization

### Python Version
- **Framework**: Keras with TensorFlow backend
- **Dataset**: Thermodynamic Steam Table (250+ patterns)
- **Input Parameters**: Pressure and Temperature
- **Output Parameters**: Internal Energy & Enthalpy (Liquid & Gas phases)
- **Architecture**: 
  - Input Layer: 2 neurons (Pressure, Temperature)
  - Hidden Layers: 3 layers × 500 neurons each
  - Output Layer: 4 neurons (Internal Energy & Enthalpy for both phases)
- **Activation Function**: ReLU (Rectified Linear Unit)
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Performance**: RMSE ≈ 16.5

## Model Performance

The Python implementation achieved impressive results on the thermodynamic steam table dataset:
- **Root Mean Squared Error (RMSE)**: ~16.5
- **Test Split**: 20% of total dataset
- **Data Preprocessing**: Feature scaling applied
- **Validation**: Manual sample testing performed

## Project Structure

```
ANN/
├── cpp/                    # C++ implementation
│   ├── src/               # Source files
│   ├── include/           # Header files
│   └── examples/          # Usage examples
├── python/                # Python implementation
│   ├── models/            # Keras models
│   ├── data/              # Dataset files
│   └── notebooks/         # Jupyter notebooks
├── docs/                  # Documentation
└── README.md
```

## Getting Started

### Prerequisites

**For C++ Implementation:**
- C++ compiler (GCC/Clang)
- CMake (optional)

**For Python Implementation:**
- Python 3.7+
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib (for visualization)

### Installation

```bash
# Clone the repository
git clone https://github.com/arkapravade/ANN.git
cd ANN

# For Python dependencies
pip install -r requirements.txt
```

### Usage

**C++ Implementation:**
```bash
cd cpp
g++ -o ann main.cpp
./ann
```

**Python Implementation:**
```bash
cd python
python train_model.py
```

## Applications

- **Engineering Thermodynamics**: Steam table property predictions
- **Pattern Recognition**: Classification and regression tasks
- **Educational Purpose**: Understanding neural network fundamentals
- **Research**: Comparing from-scratch vs. framework implementations

## Technical Insights

This project provides valuable insights into:
- **Neural Network Fundamentals**: Understanding the mathematics behind MLPs
- **Implementation Differences**: Comparing low-level C++ vs. high-level Python approaches
- **Performance Optimization**: Techniques for improving model accuracy and efficiency
- **Real-World Applications**: Applying ANNs to engineering problems

## Future Enhancements

- GPU acceleration for C++ implementation
- Additional activation functions
- Convolutional layers support
- Real-time prediction API
- Web-based visualization dashboard
- Performance benchmarking suite

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

**Arkaprava De** - [GitHub](https://github.com/arkapravade)

---

If you found this project helpful, please consider giving it a star!
