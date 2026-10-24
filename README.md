# Linear Regression from Scratch with NumPy

A hands-on implementation of linear regression using NumPy, built to understand the mathematical foundations behind machine learning algorithms.

## Overview

The goal of this project is to implement a linear regression model and a logistic regression model from scratch without relying on machine learning libraries such as scikit-learn for the core algorithm.

The implementation focuses on understanding how a model learns from data by minimizing a cost function through gradient descent.

## Learning Objectives

- Understand the mathematics behind linear regression.
- Implement predictions using NumPy and vectorized operations.
- Implement the Mean Squared Error (MSE) cost function.
- Calculate gradients and optimize model parameters using gradient descent.
- Evaluate model performance and visualize the learning process.
- Compare the implementation with scikit-learn's linear regression model.

## Technologies

- **Python** — Core programming language.
- **NumPy** — Numerical computations and vectorized operations.
- **Matplotlib** — Data visualization.
- **pytest** — Unit testing.

## Project Structure

```text
linear-regression-numpy/
├── notebooks/     # Experiments and exploratory analysis
├── src/           # Model implementation
├── tests/         # Unit tests
├── results/       # Generated plots and results
├── .gitignore
├── README.md
└── requirements.txt
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/linear-regression-numpy.git
cd linear-regression-numpy
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Implementation

The model is designed to support multiple input features.

Its predictions are calculated as:

\[
\hat{y} = Xw + b
\]

The cost function is Mean Squared Error (MSE):

\[
J(w,b) = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)^2
\]

Gradient descent is used to iteratively update the model parameters and minimize the cost function.

## Testing

Unit tests will be used to verify the correctness of the implementation and its individual components.

Run the tests with:

```bash
pytest
```

## Project Status

**In progress.** The project is being developed incrementally, starting with the mathematical foundations and progressing towards a complete, tested implementation.

## Future Improvements

- Support for multiple input features.
- Cost history tracking and visualization.
- Configurable learning rate and convergence criteria.
- Comparison with scikit-learn.
- Experiments with synthetic and real-world datasets.

## License

This project is intended for educational purposes.

