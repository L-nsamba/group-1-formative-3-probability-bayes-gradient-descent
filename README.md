## Formative 3 - Probability Distributions, Bayesian Probability, and Gradient Descent Implementation
## Part 1: Probability Distributions 

<br></br>

## Part 2: Bayesian Probability

<br></br>

## Part 3 : Gradient Descent 

## [View the Gradient Descent Code in Jupyter Notebook](https://colab.research.google.com/drive/1I9rasJMWwKXod7UBxlvfI1thS5Rpq972?usp=sharing)

## Overview
Implemented linear regression with two input features and applied gradient
descent to optimize the model's weights and bias.

## Implementation
- Calculated predictions and Mean Squared Error (MSE)
- Computed gradients using the chain rule
- Updated the model parameters over three iterations
- Used SciPy to approximate derivates and Matplotlib to visualize the results

## Results 
We implemented gradient descent to optimize the weights (m1, m2) and per-observation biases (b1, b2) of a linear regression model over three iterations. The MSE fell at every update, from 61.00 to 6.50, 2.50 and finally 2.16, so the parameters moved in a direction that reduces the error. The final values were m = [-1.37, 1.10] and b = [1.04, 0.90], giving predictions of [2.95, 6.38] against actual values of [5, 6]. The chain-rule gradients matched SciPy's numerical derivatives at each iteration.

## Visualizations
| Visualization 1 | Visualization 2 |
| :---: | :---: |
| <img src="./screenshots/mse_reduction_gradient_descent.png" width="500"> | <img src="./screenshots/m_and_b_changes_gradient_descent.png" width="500"> |
