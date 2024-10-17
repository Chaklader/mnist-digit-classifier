## Introduction to Deep Learning


1. Artificial Intelligence is an overarching field that includes algorithms such as local search and logic programming.
2. Machine Learning is a part of Artificial Intelligence and includes models like logistic regression and decision trees.
3. Deep Learning is a subfield of Machine Learning that consists of various neural network models.


Deep Learning Lifecycle:


We begin with a problem statement, then move to development (where code gets written). Next, we begin training, where our 
model learns our data. After training, we deploy, which is when our model goes out into the world for use. A lot of other 
people do front-end work at this stage. Then we go to monitoring. For our purposes, we'll focus on the development and 
training of models.


Training Tools

1. Experiment management like TensorBoard or Weights and Biases
   - Observe accuracy and loss at training time
   
2. Model versioning like DVC, Neptune, and Pachyderm
   - Remedy issues within the model across different versions of the model
   - DVC is very similar to Git

   
Improving our Machine Learning model means computing how far off our predictions are from their true values and minimizing 
the distance between those values. To do that, we'll need to understand how we can do that optimization programmatically. 
In this lesson, we will learn how to:


1. Create and manipulate PyTorch tensors.
2. Preprocess data using PyTorch.
3. Define and use loss functions to measure model performance.
4. Implement the foundational algorithms of deep learning: gradient descent and backpropagation.




# Gradient Descent and Error Minimization in Deep Learning

## I. Introduction to Gradient Descent

Gradient descent is a fundamental optimization algorithm used in machine learning and deep learning to minimize the error 
(or loss) of a model. It's the backbone of how neural networks learn from data.

### A. What is Gradient Descent?

1. Definition: An iterative optimization algorithm for finding the minimum of a function.
2. In deep learning: Used to minimize the error function (also called loss function or cost function).
3. Goal: Adjust model parameters to find the lowest point (global minimum) of the error function.

### B. Key Concepts

1. Gradient: The vector of partial derivatives of the function with respect to each parameter.
2. Descent: Moving in the opposite direction of the gradient to reduce the function's value.

## II. The Error Function

### A. Purpose of the Error Function

1. Measures the difference between the model's predictions and the actual target values.
2. Provides a quantitative way to assess model performance.

### B. Common Error Functions

1. Mean Squared Error (MSE): For regression problems
   - Formula: MSE = (1/n) * Σ(y_true - y_pred)^2
2. Cross-Entropy Loss: For classification problems
   - Formula: -Σ(y_true * log(y_pred))

## III. Gradient Descent Algorithm

### A. Basic Steps

1. Initialize model parameters randomly.
2. Calculate the error using the current parameters.
3. Compute the gradient of the error with respect to each parameter.
4. Update parameters in the opposite direction of the gradient.
5. Repeat steps 2-4 until convergence or for a set number of iterations.

### B. Update Rule

Parameters are updated using the following formula:
θ_new = θ_old - η * ∇J(θ)

Where:
- θ: Model parameter
- η: Learning rate (step size)
- ∇J(θ): Gradient of the error function with respect to θ

### C. Learning Rate (η)

1. Controls the size of steps taken during optimization.
2. Too large: May overshoot the minimum.
3. Too small: Slow convergence and risk of getting stuck in local minima.

## IV. Types of Gradient Descent

### A. Batch Gradient Descent

1. Uses entire dataset to compute gradient in each iteration.
2. Pros: Stable convergence, good for small datasets.
3. Cons: Slow for large datasets, requires all data to fit in memory.

### B. Stochastic Gradient Descent (SGD)

1. Uses a single randomly selected data point to compute gradient in each iteration.
2. Pros: Faster, can handle large datasets, can escape local minima.
3. Cons: High variance in parameter updates, may not converge to exact minimum.

### C. Mini-Batch Gradient Descent

1. Uses a small random subset of data (mini-batch) to compute gradient in each iteration.
2. Pros: Balance between batch and stochastic GD, widely used in practice.
3. Cons: Requires tuning of batch size hyperparameter.

## V. Challenges and Improvements

### A. Challenges

1. Choosing the right learning rate.
2. Avoiding local minima and saddle points.
3. Slow convergence for ill-conditioned problems.

### B. Improvements and Variants

1. Momentum: Adds a fraction of the previous update to the current one.
2. AdaGrad: Adapts learning rates for each parameter.
3. RMSprop: Addresses AdaGrad's radically diminishing learning rates.
4. Adam: Combines ideas from momentum and RMSprop.

## VI. Gradient Descent in Deep Neural Networks

### A. Backpropagation

1. Efficient algorithm for computing gradients in neural networks.
2. Uses the chain rule to propagate error gradients from output to input layers.

### B. Challenges in Deep Networks

1. Vanishing gradients: Gradients become very small in early layers.
2. Exploding gradients: Gradients become very large, causing unstable updates.

### C. Solutions

1. Careful weight initialization (e.g., Xavier/Glorot initialization).
2. Using activation functions like ReLU to mitigate vanishing gradients.
3. Gradient clipping to prevent exploding gradients.


## VII. Conclusion

Gradient descent is a powerful and versatile optimization algorithm that forms the foundation of learning in deep neural networks. 
By iteratively adjusting model parameters to minimize error, it enables these networks to learn complex patterns and relationships 
in data. Understanding its principles, variations, and challenges is crucial for effectively training and optimizing deep learning 
models.



# Data Preprocessing with PyTorch

Data preprocessing is a crucial step in preparing your data for machine learning models. PyTorch provides various tools and techniques 
to preprocess your data efficiently. Let's explore the key preprocessing steps mentioned in the image and how to implement them 
using PyTorch.

## I. Normalization

Normalization involves mapping numerical values to a standard range, typically [0, 1]. This helps in faster convergence 
during model training and ensures that all features contribute equally to the learning process.


### PyTorch Implementation:

```textmate
import torch
from torchvision import transforms

# For image data
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])

# For general numerical data
def min_max_normalize(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    return (tensor - min_val) / (max_val - min_val)
```

## II. Data Augmentation

Data augmentation involves creating modified versions of existing data to increase the diversity of your training set. This 
helps in reducing overfitting and improving model generalization.

### PyTorch Implementation:

```textmate
from torchvision import transforms

data_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(224),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
])

# Apply augmentation to an image
augmented_image = data_augmentation(original_image)
```

### A. Flipping

Flipping involves mirroring the image horizontally or vertically.

```textmate
flip_transform = transforms.RandomHorizontalFlip(p=0.5)
```

### B. Cropping

Cropping involves selecting a portion of the image.

```textmate
crop_transform = transforms.RandomCrop(224)
```

### C. Zooming

Zooming can be achieved through resizing or cropping.

```textmate
zoom_transform = transforms.RandomResizedCrop(224, scale=(0.8, 1.0))
```

## III. Label Encoding

Label encoding is the process of converting categorical labels into numerical format, which is necessary for many machine 
learning algorithms.


### PyTorch Implementation:

PyTorch doesn't have built-in label encoding, but we can use Python's standard library or other libraries like scikit-learn 4
for this purpose.



```textmate
from sklearn.preprocessing import LabelEncoder
import torch

# Example labels
labels = ['cat', 'dog', 'bird', 'cat', 'dog']

# Create and fit the LabelEncoder
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Convert to PyTorch tensor
encoded_tensor = torch.tensor(encoded_labels)
```

## IV. Creating PyTorch Datasets and DataLoaders

After preprocessing, it's important to create efficient data pipelines using PyTorch's Dataset and DataLoader classes.

```textmate
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label

# Create DataLoader
dataset = CustomDataset(data, labels, transform=data_augmentation)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```


## Conclusion

Proper data preprocessing is essential for building effective machine learning models. PyTorch provides powerful tools for 
normalization, data augmentation, and creating efficient data pipelines. By leveraging these techniques, you can improve your 
model's performance and generalization capabilities.



## Data Representation

Rarely can we use "out of the box" input. We need our input to be tensors, but often our raw data consists of images, text, 
or tabular data, and we can't easily input those directly into our model.

1. For image data, we need the data to be turned into tensors with entries of the tensors as bit values in color channels (usually red, green, and blue).
2. Text data needs to be tokenized, meaning, individual words or groups of letters need to be mapped to a token value.
3. For tabular data, we have categorical values (high, medium, low, colors, demographic information, etc...) that we need to 
transform into numbers for processing.


Transforming Data for Neural Networks
Often, we are faced with data that is not in a format conducive to use in neural networks in its raw form. Preprocessing is the act of turning data from that raw form into tensors that can be used as input to a neural network. This includes:


1. Encoding non-numerical features
2. Converting images to tensors of bit values in color channels
3. Tokenizing words


One-Hot Encoding:

1. Definition:
   One-Hot Encoding is a process of converting categorical variables into a form that could be provided to machine learning algorithms to do a better job in prediction. It creates binary columns for each category and uses 0 or 1 to indicate the presence of that category.

2. Purpose:
   - To represent categorical variables without assuming any ordinal relationship between categories.
   - To avoid the potential pitfall of machine learning algorithms assuming a natural ordering between categories when using simple integer encoding.

3. How it works:
   - For each category in a feature, a new binary column is created.
   - Each column represents one possible value of the categorical feature.
   - The column for the present category gets a value of 1, while all others get 0.

4. Example:
   Let's say we have a "Color" feature with categories: Red, Blue, Green
   
   One-Hot Encoding would create:
   
   | Color_Red | Color_Blue | Color_Green |
   |-----------|------------|-------------|
   |     1     |     0      |      0      |  (for Red)
   |     0     |     1      |      0      |  (for Blue)
   |     0     |     0      |      1      |  (for Green)

5. Advantages:
   - No arbitrary numerical relationships between categories.
   - Works well with many machine learning algorithms, especially tree-based models and neural networks.

6. Disadvantages:
   - Can significantly increase the number of features, leading to the "curse of dimensionality" for datasets with many categorical variables or categories.

7. PyTorch Implementation:
   While PyTorch doesn't have a built-in one-hot encoder, you can easily implement it:

```textmate
import torch
import torch.nn.functional as F

# Assume we have categorical data as integers
data = torch.tensor([0, 2, 1, 1, 0, 2])
num_categories = 3

# One-hot encode
one_hot = F.one_hot(data, num_classes=num_categories)

print(one_hot)
```

This will output:

```textmate
tensor([[1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1]])
```

8. When to use One-Hot Encoding vs Label Encoding:
   - Use One-Hot Encoding for nominal categorical variables (no inherent order).
   - Use Label Encoding for ordinal categorical variables (have a natural order).


One-Hot Encoding is particularly useful when working with neural networks or any algorithm that doesn't assume ordinal 
relationships between categories. It ensures that the machine learning model treats each category independently.




Error Function 

An error function is simply a function that measures how far the current state is from the solution. We can calculate the 
error and then make a change in an attempt to reduce the error—and then repeat this process until we have reduced the error 
to an acceptable level.



Log-Loss Error Function

Discrete and Continuous Errors
One approach to reducing errors might be to simply count the number of errors and then make changes until the number of 
errors is reduced. But taking a discrete approach like this can be problematic—for example, we could change our line in 
a way that gets closer to the solution, but this change might not (by itself) improve the number of misclassified points.

Instead, we need to construct an error function that is continuous. That way, we can always tell if a small change in the 
line gets us closer to the solution. We'll do that in this lesson using the log-loss error function. Generally speaking, 
the log-loss function will assign a large penalty to incorrectly classified points and small penalties to correctly classified 
points. For a point that is misclassified, the penalty is roughly the distance from the boundary to the point. For a point that 
is correctly classified, the penalty is almost zero.

We can then calculate a total error by adding all the errors from the corresponding points. Then we can use gradient descent 
to solve the problem, making very tiny changes to the parameters of the line in order to decrease the total error until we 
have reached an acceptable minimum.

We need to cover some other concepts before we get into the specifics of how to calculate our log-loss function, but we'll 
come back to it when we dive into gradient descent later in the lesson.



# Log-Loss Error Function and Its Use in Gradient Descent


## I. Introduction to Log-Loss Error Function

The Log-Loss Error Function, also known as Cross-Entropy Loss, is a widely used loss function in machine learning, particularly for binary and multi-class classification problems.

### A. Definition

For binary classification, the Log-Loss function is defined as:

L(y, ŷ) = -1/N ∑(i=1 to N) [yi log(ŷi) + (1-yi) log(1-ŷi)]

Where:
- N is the number of samples
- yi is the true label (0 or 1)
- ŷi is the predicted probability of the positive class

### B. Characteristics

1. Always positive: Log-Loss is always ≥ 0
2. Perfect prediction: Log-Loss = 0 when the model predicts the correct class with 100% confidence
3. Penalizes confident mistakes: Heavily penalizes predictions that are both confident and wrong

## II. Log-Loss in Multi-class Classification

For multi-class problems, the formula extends to:

L(y, ŷ) = -1/N ∑(i=1 to N) ∑(j=1 to M) yij log(ŷij)

Where:
- M is the number of classes
- yij is 1 if sample i belongs to class j, and 0 otherwise
- ŷij is the predicted probability that sample i belongs to class j


## III. Why Use Log-Loss?

1. Probabilistic interpretation: Directly models probability distributions
2. Differentiable: Suitable for optimization algorithms like gradient descent
3. Handles imbalanced datasets well
4. Provides smoother gradients compared to other loss functions (e.g., 0-1 loss)

## IV. Log-Loss and Gradient Descent

Gradient Descent is an optimization algorithm used to minimize the loss function by iteratively adjusting the model parameters.

### A. Gradient Descent Process

1. Initialize model parameters randomly
2. Calculate the predicted probabilities using current parameters
3. Compute the Log-Loss
4. Calculate the gradient of Log-Loss with respect to each parameter
5. Update parameters in the opposite direction of the gradient
6. Repeat steps 2-5 until convergence

### B. Gradient Calculation

For logistic regression (binary classification), the gradient of Log-Loss with respect to weights w is:

∂L/∂w = 1/N Xᵀ(ŷ - y)

Where:
- X is the input feature matrix
- ŷ is the vector of predicted probabilities
- y is the vector of true labels

### C. Parameter Update Rule

w_new = w_old - α ∂L/∂w

Where α is the learning rate.

## V. Implementation in PyTorch

PyTorch provides built-in functions for Log-Loss and automatic gradient computation:



```textmate
import torch
import torch.nn as nn

# Define model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Create model, loss function, and optimizer
model = LogisticRegression(input_dim=10)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    # Forward pass
    y_pred = model(X)
    loss = criterion(y_pred, y)
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## VI. Advantages and Considerations

### A. Advantages
1. Works well for probabilistic classification
2. Provides smooth gradients for optimization
3. Naturally handles multi-class problems

### B. Considerations
1. Sensitive to outliers
2. May lead to overfitting if not regularized
3. Assumes independence between features (in logistic regression)



### Gradient Descent Conditions 


1. We need to be able to take very small steps in the direction that minimizes the error, which is only possible if our error 
function is continuous. With a discrete error function (such as a simple count of the number of misclassified points), a 
single small change may not have any detectable effect on the error.

2. We also mentioned that the error function should be differentiable.

## Conclusion

The Log-Loss Error Function, combined with Gradient Descent, forms a powerful framework for training classification models. 
Its probabilistic nature and smooth gradients make it particularly suitable for a wide range of machine learning tasks, 
especially when implemented with modern deep learning frameworks like PyTorch.



## Python implementation of the Log-Loss function for binary classification based on the given formula


This Python code implements the Log-Loss function for binary classification as you specified. Here's a breakdown of the implementation:


```textmate
import numpy as np

def binary_log_loss(y_true, y_pred):
    """
    Calculate the Log-Loss (Binary Cross-Entropy) for binary classification.
    
    Parameters:
    y_true (array-like): True binary labels (0 or 1)
    y_pred (array-like): Predicted probabilities for the positive class
    
    Returns:
    float: The calculated Log-Loss
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Clip predicted values to avoid log(0)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # Calculate log loss
    N = len(y_true)
    loss = -1/N * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    return loss

# Example usage
if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.2])
    
    loss = binary_log_loss(y_true, y_pred)
    print(f"Log-Loss: {loss:.4f}")
```

1. We define a function binary_log_loss that takes two parameters: y_true (actual labels) and y_pred (predicted probabilities).
2. The function converts inputs to NumPy arrays for efficient computation.
3. We clip the predicted probabilities to avoid taking the log of 0, which would result in undefined behavior.
4. The Log-Loss is calculated using the formula you provided, leveraging NumPy's vectorized operations for efficiency.
5. An example usage is provided at the end, demonstrating how to use the function with sample data.


This implementation can be easily integrated into a larger machine learning pipeline or used standalone to evaluate binary 
classification models.


## Python implementation of the Log-Loss function for multi-class classification

This Python code implements the Log-Loss function for multi-class classification as per the formula you provided. Here's a breakdown of the implementation:


```textmate
import numpy as np

def multiclass_log_loss(y_true, y_pred):
    """
    Calculate the Log-Loss (Categorical Cross-Entropy) for multi-class classification.
    
    Parameters:
    y_true (array-like): True labels in one-hot encoded format (N x M)
    y_pred (array-like): Predicted probabilities for each class (N x M)
    
    Returns:
    float: The calculated Log-Loss
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Clip predicted values to avoid log(0)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # Calculate log loss
    N = y_true.shape[0]  # Number of samples
    loss = -1/N * np.sum(y_true * np.log(y_pred))
    
    return loss

# Example usage
if __name__ == "__main__":
    # Example with 3 samples and 4 classes
    y_true = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])
    y_pred = np.array([
        [0.9, 0.05, 0.03, 0.02],
        [0.1, 0.7, 0.1, 0.1],
        [0.05, 0.05, 0.8, 0.1]
    ])
    
    loss = multiclass_log_loss(y_true, y_pred)
    print(f"Multi-class Log-Loss: {loss:.4f}")
```


1. We define a function multiclass_log_loss that takes two parameters: y_true (actual labels in one-hot encoded format) and y_pred (predicted probabilities for each class).
2. The function converts inputs to NumPy arrays for efficient computation.
3. We clip the predicted probabilities to avoid taking the log of 0, which would result in undefined behavior.
4. The Log-Loss is calculated using the formula you provided. Note that the summation over classes (M) is implicitly handled by NumPy's element-wise multiplication and sum operations.
5. An example usage is provided at the end, demonstrating how to use the function with sample data for a scenario with 3 samples and 4 classes.


Key points about this implementation:

1. It expects y_true to be in one-hot encoded format, where each row represents a sample and each column represents a class.
2. y_pred should contain predicted probabilities for each class, with each row summing to 1.
3. The implementation is vectorized, making it efficient for large datasets.



## Maximum Likelihood

# Maximum Likelihood Estimation with Sigmoid Function

## I. Introduction

Maximum Likelihood Estimation (MLE) is a fundamental method in statistics and machine learning for estimating the parameters of a probability distribution. When combined with the sigmoid function, it forms the basis of logistic regression, a powerful tool for binary classification.

## II. The Sigmoid Function

### A. Definition

The sigmoid function, also known as the logistic function, is defined as:

σ(z) = 1 / (1 + e^(-z))

### B. Properties

1. Output range: (0, 1)
2. S-shaped curve
3. Symmetric around 0.5
4. Differentiable

### C. Use in Classification

In binary classification, the sigmoid function is used to map any real-valued number to a probability between 0 and 1.

## III. Logistic Regression Model

In logistic regression, we model the probability of the positive class as:

P(Y=1|X) = σ(wᵀX + b)

Where:
- X is the input feature vector
- w is the weight vector
- b is the bias term

## IV. Maximum Likelihood Estimation





### A. Likelihood Function



For a dataset with n independent samples, the likelihood function is:

L(w, b) = ∏(i=1 to n) P(Y=yi|Xi)
        = ∏(i=1 to n) [σ(wᵀXi + b)]^yi [1 - σ(wᵀXi + b)]^(1-yi)

Where yi is the true label (0 or 1) for the i-th sample.

### B. Log-Likelihood

We typically work with the log-likelihood for computational convenience:

ℓ(w, b) = ∑(i=1 to n) [yi log(σ(wᵀXi + b)) + (1-yi) log(1 - σ(wᵀXi + b))]

### C. Maximum Likelihood Estimator

The goal is to find w and b that maximize the log-likelihood:

(w*, b*) = argmax(w,b) ℓ(w, b)

## V. Optimization

### A. Gradient Ascent

We can use gradient ascent to find the maximum:

w := w + α ∂ℓ/∂w
b := b + α ∂ℓ/∂b

Where α is the learning rate.

### B. Gradients

The gradients of the log-likelihood with respect to w and b are:

∂ℓ/∂w = ∑(i=1 to n) (yi - σ(wᵀXi + b)) Xi

∂ℓ/∂b = ∑(i=1 to n) (yi - σ(wᵀXi + b))

## VI. Connection to Cross-Entropy Loss

Maximizing the log-likelihood is equivalent to minimizing the cross-entropy loss:

Cross-Entropy = -1/n ℓ(w, b)

This is why the cross-entropy loss is commonly used as the objective function in logistic regression and neural networks with sigmoid output.

## VII. Implementation in Python

Here's a basic implementation of logistic regression using MLE with the sigmoid function:



```textmate
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_likelihood(X, y, w, b):
    z = np.dot(X, w) + b
    ll = np.sum(y * np.log(sigmoid(z)) + (1 - y) * np.log(1 - sigmoid(z)))
    return ll

def gradient(X, y, w, b):
    z = np.dot(X, w) + b
    error = y - sigmoid(z)
    grad_w = np.dot(X.T, error)
    grad_b = np.sum(error)
    return grad_w, grad_b

def logistic_regression(X, y, learning_rate=0.1, num_iterations=1000):
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    
    for _ in range(num_iterations):
        grad_w, grad_b = gradient(X, y, w, b)
        w += learning_rate * grad_w
        b += learning_rate * grad_b
    
    return w, b

# Example usage
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([0, 0, 1, 1])
w, b = logistic_regression(X, y)
print("Weights:", w)
print("Bias:", b)
```

## VIII. Conclusion


Maximum Likelihood Estimation with the sigmoid function provides a powerful framework for binary classification. It forms 
the basis of logistic regression and is closely related to the cross-entropy loss used in many machine learning models. 
Understanding this connection helps in interpreting model outputs as probabilities and in choosing appropriate loss functions 
for classification tasks.



# Gradient Calculation


Gradient descent is a fundamental optimization algorithm used in machine learning, particularly for training models like 
logistic regression. Here's the basic idea:

1. We start with a model that makes predictions, but it's not very accurate at first.

2. We have a way to measure how wrong our predictions are, called an error function or loss function.

3. Our goal is to adjust the model's parameters (weights and bias) to make it more accurate - in other words, to reduce the error.

4. The gradient is like a compass that tells us which direction to move our parameters to reduce the error. It points in the direction of steepest increase, so we go in the opposite direction to decrease the error.

5. We calculate this gradient for each parameter of our model.

6. Then we take a small step in the direction opposite to the gradient for each parameter. How big this step is depends on our learning rate (α).

7. If the gradient is large, we'll take a bigger step. If it's small, we'll take a smaller step.

8. We repeat this process many times, each time slightly adjusting our model's parameters to make it a little bit better.

9. Over time, these small adjustments add up, and our model becomes more and more accurate.

The formulas in the image show exactly how we update our weights and bias:
- For weights: we add α(y - ŷ)xi to each weight
- For bias: we add α(y - ŷ)

Here, (y - ŷ) is the difference between the true value and our prediction. If our prediction is too low, this will be positive, 
and we'll increase our weights and bias. If our prediction is too high, this will be negative, and we'll decrease them.

The xi term for weights means we adjust weights more for features that have larger values, as they have more impact on the prediction.

By repeatedly applying these updates across our entire dataset, we gradually improve our model's accuracy.

In the last few videos, we learned that in order to minimize the error function, we need to take some derivatives. So let's 
get our hands dirty and actually compute the derivative of the error function. The first thing to notice is that the sigmoid 
function has a really nice derivative. Namely,


σ'(x) = σ(x)(1 - σ(x))

The reason for this is the following, we can calculate it using the quotient formula:

σ'(x) = ∂/∂x [1 / (1 + e^(-x))]
      = e^(-x) / (1 + e^(-x))^2
      = [1 / (1 + e^(-x))] · [e^(-x) / (1 + e^(-x))]
      = σ(x)(1 - σ(x)).

And now, let's recall that if we have m points labelled x^(1), x^(2), ..., x^(m), the error formula is:

E = -1/m ∑(i=1 to m) [yi ln(ŷi) + (1 - yi) ln(1 - ŷi)]

where the prediction is given by ŷi = σ(Wx^(i) + b).

Our goal is to calculate the gradient of E, at a point x = (x1, ..., xn), given by the partial derivatives

∇E = (∂E/∂w1, ..., ∂E/∂wn, ∂E/∂b)

To simplify our calculations, we'll actually think of the error that each point produces, and calculate the derivative of this error. The total error, then, is the average of the errors at all the points. The error produced by each point is, simply,

E = -y ln(ŷ) - (1 - y) ln(1 - ŷ)

In order to calculate the derivative of this error with respect to the weights, we'll first calculate ∂ŷ/∂wj. Recall that ŷ = σ(Wx + b), so:

∂ŷ/∂wj = ∂/∂wj σ(Wx + b)
        = σ(Wx + b)(1 - σ(Wx + b)) · ∂/∂wj (Wx + b)
        = ŷ(1 - ŷ) · ∂/∂wj (Wx + b)
        = ŷ(1 - ŷ) · ∂/∂wj (w1x1 + ... + wjxj + ... + wnxn + b)
        = ŷ(1 - ŷ) · xj.

Now, we can go ahead and calculate the derivative of the error E at a point x, with respect to the weight wj.

∂E/∂wj = ∂/∂wj [-y log(ŷ) - (1 - y) log(1 - ŷ)]
        = -y ∂/∂wj log(ŷ) - (1 - y) ∂/∂wj log(1 - ŷ)
        = -y · 1/ŷ · ∂ŷ/∂wj - (1 - y) · 1/(1 - ŷ) · ∂/∂wj (1 - ŷ)
        = -y · 1/ŷ · ŷ(1 - ŷ)xj - (1 - y) · 1/(1 - ŷ) · (-1)ŷ(1 - ŷ)xj
        = -y(1 - ŷ)xj + (1 - y)ŷxj
        = -(y - ŷ)xj.

A similar calculation will show us that

∂E/∂b = -(y - ŷ).

This actually tells us something very important. For a point with coordinates (x1, ..., xn), label y, and prediction ŷ, 
the gradient of the error function at that point is (-(y - ŷ)x1, ..., -(y - ŷ)xn, -(y - ŷ)). In summary, the gradient is

∇E = -(y - ŷ)(x1, ..., xn, 1).


So, a small gradient means we'll change our coordinates by a little bit, and a large gradient means we'll change our coordinates by a lot.

# Gradient Descent Step

Therefore, since the gradient descent step simply consists in subtracting a multiple of the gradient of the error function 
at every point, then this updates the weights in the following way:

w'i ← wi - α[-(y - ŷ)xi],

which is equivalent to

w'i ← wi + α(y - ŷ)xi.

Similarly, it updates the bias in the following way:

b' ← b + α(y - ŷ),


Note: Since we've taken the average of the errors, the term we are adding should be 1/m · α instead of α, but as α is a 
constant, then in order to simplify calculations, we'll just take 1/m · α to be our learning rate, and abuse the notation 
by just calling it α.


# Gradient Calculation for Logistic Regression

## I. Introduction

In logistic regression, we need to calculate derivatives to minimize the error function. This lecture note focuses on computing the gradient of the error function.

## II. Sigmoid Function Derivative

The sigmoid function has a convenient derivative:

σ'(x) = σ(x)(1 - σ(x))

Proof:

σ'(x) = ∂/∂x [1 / (1 + e^(-x))]
       = e^(-x) / (1 + e^(-x))^2
       = [1 / (1 + e^(-x))] · [e^(-x) / (1 + e^(-x))]
       = σ(x)(1 - σ(x))

## III. Error Function

For m points labeled x^(1), x^(2), ..., x^(m), the error formula is:

E = -1/m ∑(i=1 to m) [yi ln(ŷi) + (1 - yi) ln(1 - ŷi)]

Where the prediction is given by ŷi = σ(Wx^(i) + b).

## IV. Gradient Calculation

Our goal is to calculate the gradient of E at a point x = (x1, ..., xn):

∇E = (∂E/∂w1, ..., ∂E/∂wn, ∂E/∂b)

### A. Derivative with respect to weights

First, we calculate ∂ŷ/∂wj:

∂ŷ/∂wj = ŷ(1 - ŷ) · xj

Then, we calculate ∂E/∂wj:

∂E/∂wj = -y · (1/ŷ) · ∂ŷ/∂wj - (1-y) · (1/(1-ŷ)) · ∂(1-ŷ)/∂wj
        = -y(1 - ŷ)xj + (1 - y)ŷxj
        = -(y - ŷ)xj

### B. Derivative with respect to bias

Similarly, we can show that:

∂E/∂b = -(y - ŷ)

## V. Final Gradient Formula

For a point with coordinates (x1, ..., xn), label y, and prediction ŷ, the gradient of the error function at that point is:

∇E = -(y - ŷ)(x1, ..., xn, 1)


## VI. Significance

The gradient is a scalar (y - ŷ) multiplied by the coordinates of the point (with an additional 1 for the bias term). This 
scalar represents the difference between the true label and the prediction, highlighting the error's direct influence on the gradient's magnitude and direction.


### Logistic Regression


1. Take your data.
2. Pick a random model.
3. Calculate the error.
4. Minimize the error and obtain a better model



## I. Introduction
Logistic regression is a fundamental machine learning algorithm used for binary classification problems. Despite its name, it's a classification algorithm, not a regression algorithm.

## II. The Logistic Function
At the core of logistic regression is the logistic function (also called sigmoid function):

σ(z) = 1 / (1 + e^(-z))

This function maps any real-valued number to a value between 0 and 1, which can be interpreted as a probability.

## III. Model Representation
In logistic regression, we model the probability that an input X belongs to the positive class:

P(Y=1|X) = σ(WᵀX + b)

Where:
- W is the weight vector
- X is the input feature vector
- b is the bias term

## IV. Decision Boundary
The decision boundary is the point where the model predicts a 0.5 probability. This occurs when WᵀX + b = 0.

## V. Cost Function
We use the log loss (also called cross-entropy loss) as our cost function:

J(W,b) = -1/m ∑[y·log(ŷ) + (1-y)·log(1-ŷ)]

Where:
- m is the number of training examples
- y is the true label (0 or 1)
- ŷ is the predicted probability

## VI. Gradient Descent
To minimize the cost function, we use gradient descent:

1. Initialize W and b
2. Repeat until convergence:
   W := W - α · ∂J/∂W
   b := b - α · ∂J/∂b

Where α is the learning rate.

## VII. Advantages and Limitations
Advantages:
- Simple and interpretable
- Performs well on linearly separable classes
- Outputs have a nice probabilistic interpretation

Limitations:
- Assumes a linear decision boundary
- May underperform on complex datasets

## VIII. Applications
- Medical diagnosis
- Email spam detection
- Credit risk assessment
- Marketing campaign response prediction

---

## Calculating the Error Function

For reference, here is the formula for the error function (for binary classification problems):

Error function = -1/m ∑(i=1 to m) [(1 - yi) ln(1 - ŷi) + yi ln(ŷi)]

And the total formula for the error is then:

E(W, b) = -1/m ∑(i=1 to m) [(1 - yi) ln(1 - σ(Wx^(i) + b)) + yi ln(σ(Wx^(i) + b))]

For multiclass problems, the error function is:

Error function = -1/m ∑(i=1 to m) ∑(j=1 to n) yij ln(ŷij)

Now that we know how to calculate the error, our goal will be to minimize it.



## Minimizing the Error Function in Logistic Regression

## I. Introduction

Minimizing the error function is a crucial step in training a logistic regression model. The goal is to find the optimal 
parameters (weights and bias) that result in the lowest possible error on the training data.

## II. The Error Function

Recall the error function for binary logistic regression:

E(W, b) = -1/m ∑(i=1 to m) [(1 - yi) ln(1 - ŷi) + yi ln(ŷi)]

Where:
- m is the number of training examples
- yi is the true label (0 or 1)
- ŷi is the predicted probability

## III. Gradient Descent

The primary method for minimizing the error function is gradient descent. This iterative optimization algorithm takes steps 
proportional to the negative of the gradient of the function at the current point.

### Steps of Gradient Descent:

1. Initialize parameters W and b randomly
2. Calculate the gradient of E with respect to W and b
3. Update parameters:
   W := W - α ∂E/∂W
   b := b - α ∂E/∂b
4. Repeat steps 2-3 until convergence

Where α is the learning rate.

## IV. Calculating the Gradient

The gradient for each weight wj is:

∂E/∂wj = 1/m ∑(i=1 to m) (ŷi - yi) xij

And for the bias b:

∂E/∂b = 1/m ∑(i=1 to m) (ŷi - yi)

## V. Learning Rate

The learning rate α determines the step size at each iteration. 
- If α is too small, convergence will be slow.
- If α is too large, the algorithm might overshoot the minimum and fail to converge.

Choosing an appropriate learning rate is crucial for effective minimization.

## VI. Convergence

The algorithm converges when:
- The change in error between iterations becomes very small
- A maximum number of iterations is reached

## VII. Variants of Gradient Descent

1. Batch Gradient Descent: Uses the entire dataset for each update
2. Stochastic Gradient Descent (SGD): Uses a single example for each update
3. Mini-Batch Gradient Descent: Uses a small random subset of data for each update

## VIII. Optimization Techniques

Several techniques can improve the basic gradient descent algorithm:

1. Momentum: Adds a fraction of the previous update to the current one
2. AdaGrad: Adapts the learning rate for each parameter
3. RMSprop: Addresses AdaGrad's radically diminishing learning rates
4. Adam: Combines ideas from momentum and RMSprop

## IX. Avoiding Local Minima

The error function in logistic regression is convex, meaning it has only one global minimum. This ensures that gradient 
descent will converge to the optimal solution, regardless of the initial parameter values.

## X. Regularization

To prevent overfitting, we often add a regularization term to the error function:

E(W, b) = -1/m ∑(i=1 to m) [(1 - yi) ln(1 - ŷi) + yi ln(ŷi)] + λ||W||^2

Where λ is the regularization parameter. This discourages the model from relying too heavily on any single feature.

## Conclusion

Minimizing the error function is a critical step in training an effective logistic regression model. By using gradient 
descent and its variants, we can find the optimal parameters that best fit our training data while avoiding overfitting.




# Implementing Gradient Descent

## I. Introduction

Gradient descent is a fundamental optimization algorithm used to minimize the error function in machine learning models, including neural networks. This lecture focuses on implementing gradient descent for a simple neural network using the graduate school admissions dataset.

## II. Data Preprocessing

Before implementing gradient descent, it's crucial to preprocess the data:

1. Convert categorical variables (like rank) into dummy variables.
2. Standardize continuous variables (GRE and GPA) to have zero mean and unit standard deviation.

## III. Error Function

We use the Mean Squared Error (MSE) instead of Sum of Squared Errors (SSE):

E = 1/(2m) ∑μ(yμ - ŷμ)^2

Where:
- m is the number of records
- y is the true label
- ŷ is the predicted output

## IV. Gradient Descent Algorithm

1. Initialize weights
2. For each epoch:
   a. Set weight step Δwi = 0
   b. For each record in the training data:
      - Forward pass: ŷ = f(∑i wi xi)
      - Calculate error term: δ = (y - ŷ) * f'(∑i wi xi)
      - Update weight step: Δwi = Δwi + δxi
   c. Update weights: wi = wi + η Δwi / m

Where:
- η is the learning rate
- f is the activation function (sigmoid in this case)
- f' is the derivative of the activation function

## V. Sigmoid Function and Its Derivative

Sigmoid function: f(h) = 1 / (1 + e^(-h))
Derivative: f'(h) = f(h)(1 - f(h))

## VI. Implementation in Python

Here's the Python implementation of the gradient descent algorithm:



```textmate
def update_weights(weights, features, targets, learnrate):
    del_w = np.zeros(weights.shape)
    for x, y in zip(features.values, targets):
        output = sigmoid(np.dot(x, weights))
        error = y - output
        error_term = error * output * (1 - output)
        del_w += error_term * x
    
    n_records = features.shape[0]
    weights += learnrate * del_w / n_records
    
    return weights
```

## VII. Key Points

1. Initialize weights randomly from a normal distribution with scale 1/sqrt(n_features).
2. Use np.dot() for efficient computation of the dot product.
3. The error term calculation simplifies due to the properties of the sigmoid function.
4. Adjust the learning rate to balance between convergence speed and accuracy.


## VIII. Conclusion

Implementing gradient descent involves careful data preprocessing, understanding of the error function and its derivatives, 
and efficient computation using libraries like NumPy. This implementation forms the basis for training more complex neural 
networks and can be extended to other optimization problems in machine learning.




# Perceptrons

<br>

![Alt text](images/perceptrons.png)

<br>


## I. Definition of a Perceptron

A Perceptron is a fundamental building block of neural networks, representing the simplest form of an artificial neuron. 
It takes multiple inputs, processes them, and produces a binary output.

## II. Structure of a Perceptron

1. Inputs: The Perceptron receives n inputs, denoted as x₁, x₂, ..., xₙ.
2. Weights: Each input is associated with a weight (W₁, W₂, ..., Wₙ).
3. Bias: An additional input with a constant value of 1 and its associated weight b.

## III. Computation in a Perceptron

The Perceptron performs two main steps:

1. Linear Combination:
   It calculates a weighted sum of inputs plus the bias:

   Wx + b = ∑(i=1 to n) WᵢXᵢ + b

2. Activation:

The result is then passed through an activation function. In the simplest case, this is a step function:
   - If Wx + b ≥ 0, the output is "Yes" (or 1)
   - If Wx + b < 0, the output is "No" (or 0)

## IV. Decision Making

- The Perceptron essentially defines a decision boundary in the input space.
- This boundary is a hyperplane defined by the equation Wx + b = 0.
- Inputs on one side of this hyperplane are classified as "Yes", and on the other side as "No".

## V. Learning Process

- The weights and bias of a Perceptron can be adjusted through a learning process.
- This typically involves presenting the Perceptron with labeled training data and adjusting the weights based on the errors it makes.

## VI. Limitations

- Perceptrons can only learn linearly separable functions.
- For more complex problems, multiple Perceptrons need to be combined into multi-layer networks.

## VII. Historical Significance

- Introduced by Frank Rosenblatt in 1958, Perceptrons were among the first machine learning algorithms.
- They laid the groundwork for more complex neural network architectures.

## VIII. Applications

- Binary classification tasks
- As building blocks in more complex neural networks
- Simple decision-making systems

Understanding Perceptrons is crucial as they form the basis for understanding more complex neural network architectures 
and deep learning models.




# Multilayer Perceptrons (MLPs)


## I. Introduction

Multilayer Perceptrons (MLPs) are neural networks with one or more hidden layers between the input and output layers. They 
can solve linearly inseparable problems, unlike single-layer perceptrons.

## II. Structure of an MLP
- Input layer
- One or more hidden layers
- Output layer

Example: A network with three input units, two hidden units, and one output unit.


## III. Calculation Process

1. Input to hidden layer: Weighted sum of inputs plus bias
2. Hidden layer activation: Apply activation function (e.g., sigmoid)
3. Output layer input: Weighted sum of hidden layer activations
4. Output layer activation: Apply activation function to get final output


## IV. Weight Notation
For multiple layers, weights require two indices: wᵢⱼ
- i: input unit index
- j: hidden unit index

## V. Weight Matrix
For a network with 3 input units and 2 hidden units:

[w₁₁ w₁₂]
[w₂₁ w₂₂]
[w₃₁ w₃₂]

## VI. Hidden Layer Calculation
For each hidden unit hⱼ:

hⱼ = ∑ᵢ wᵢⱼxᵢ

Using matrix multiplication:

hidden_inputs = np.dot(inputs, weights_input_to_hidden)

## VII. Implementation in NumPy

```textmate
# Initialize weights
n_records, n_inputs = features.shape
n_hidden = 2
weights_input_to_hidden = np.random.normal(0, n_inputs**-0.5, size=(n_inputs, n_hidden))

# Calculate hidden layer inputs
hidden_inputs = np.dot(inputs, weights_input_to_hidden)
```

## VIII. Forward Pass Implementation

The forward pass in a perceptron, or more generally in a neural network, is the process of propagating input data through 
the network to generate an output. Let me explain this process step-by-step for a multilayer perceptron:

Input Layer:

The process starts with the input data being fed into the input layer.


Hidden Layer(s):

For each neuron in the hidden layer:
a. Calculate the weighted sum of inputs:
h_j = ∑(i=1 to n) w_ij * x_i + b_j
Where:

h_j is the input to the j-th hidden neuron
w_ij is the weight from the i-th input to the j-th hidden neuron
x_i is the i-th input
b_j is the bias for the j-th hidden neuron

b. Apply the activation function:
a_j = f(h_j)
Where f is typically a non-linear function like sigmoid or ReLU


Output Layer:

Similar to the hidden layer, but using the outputs from the hidden layer as inputs:
a. Calculate the weighted sum:
y_k = ∑(j=1 to m) w_jk * a_j + b_k
b. Apply the activation function:
output_k = f(y_k)


Final Output:

The activations of the output layer neurons are the final output of the network.

```textmate
def forward_pass(x, weights_input_to_hidden, weights_hidden_to_output):
    # Calculate input to hidden layer
    hidden_layer_in = np.dot(x, weights_input_to_hidden)
    
    # Calculate hidden layer output
    hidden_layer_out = sigmoid(hidden_layer_in)
    
    # Calculate input to output layer
    output_layer_in = np.dot(hidden_layer_out, weights_hidden_to_output)
    
    # Calculate network output
    output_layer_out = sigmoid(output_layer_in)
    
    return hidden_layer_out, output_layer_out
```

## IX. Key Points
- MLPs can learn more complex patterns with deep stacks of hidden layers.
- Matrix multiplication is crucial for efficient computation in MLPs.
- The dimensions of weight matrices must match the number of units in connected layers.
- Activation functions (e.g., sigmoid) introduce non-linearity, allowing MLPs to solve complex problems.



# Backpropagation in Neural Networks


Backpropagation is a fundamental algorithm used to train artificial neural networks, especially those with multiple layers. 
Here's a brief explanation:

1. Purpose: It's used to calculate gradients efficiently, which are needed to update the network's weights during training.

2. Process: 
   - First, the network makes a forward pass to generate predictions.
   - Then, it calculates the error between predictions and actual targets.
   - Finally, it propagates this error backwards through the network layers.

3. Key Idea: It uses the chain rule of calculus to compute how each weight contributes to the overall error.

4. Weight Updates: Based on these calculated gradients, the weights are adjusted to minimize the error.

5. Efficiency: Backpropagation allows for efficient computation of gradients for all weights simultaneously, making it 
feasible to train large networks.

6. Iterative: This process is repeated many times over the training data to gradually improve the network's performance.

In essence, backpropagation is the "learning" mechanism that allows neural networks to adjust their internal parameters 
to better fit the training data.


## I. Introduction

Backpropagation is a fundamental algorithm for training multilayer neural networks. It extends the concept of gradient 
descent to hidden layers, allowing the network to learn complex patterns.

## II. The Concept of Backpropagation

In a multilayer network, we need to calculate the error for hidden layer units to update their weights. Backpropagation 
solves this by propagating the error backwards through the network.

The error in a hidden unit is proportional to the sum of the errors in the output units to which it is connected, weighted 
by the strength of those connections. This aligns with the intuition that units more strongly connected to the output contribute 
more to the final error.

## III. Mathematical Formulation

For a hidden unit j, the error term δⱼʰ is calculated as:

δⱼʰ = ∑ Wⱼₖ δᵏₒ f'(hⱼ)

Where:
- Wⱼₖ is the weight between hidden unit j and output unit k
- δᵏₒ is the error term of output unit k
- f'(hⱼ) is the derivative of the activation function at the input to unit j

The weight update rule remains similar to single-layer networks:

Δwᵢⱼ = η δⱼʰ xᵢ

Where:
- η is the learning rate
- xᵢ is the input to the layer

## IV. Detailed Example

Let's walk through a simple two-layer network with:
- Two input values
- One hidden unit
- One output unit
- Sigmoid activation functions

Network structure:
[Image of the network structure would be here]

1. Forward Pass:

   h = ∑ᵢ wᵢxᵢ = 0.1 × 0.4 - 0.2 × 0.3 = -0.02
   a = f(h) = sigmoid(-0.02) = 0.495
   ŷ = f(W·a) = sigmoid(0.1 × 0.495) = 0.512

2. Backward Pass:

   Output error term:
   δₒ = (y - ŷ) f'(W·a) = (1 - 0.512) × 0.512 × (1 - 0.512) = 0.122

   Hidden unit error term:
   δʰ = W δₒ f'(h) = 0.1 × 0.122 × 0.495 × (1 - 0.495) = 0.003

3. Weight Updates:

   Hidden to output weight:
   ΔW = η δₒ a = 0.5 × 0.122 × 0.495 = 0.0302

   Input to hidden weights:
   Δwᵢ = η δʰ xᵢ = (0.5 × 0.003 × 0.1, 0.5 × 0.003 × 0.3) = (0.00015, 0.00045)

## V. Implementation in NumPy

When implementing backpropagation with NumPy, we need to handle matrix operations efficiently:

```textmate
def backward_pass(x, target, learnrate, hidden_layer_out, output_layer_out, weights_hidden_to_output):
    # Calculate output error
    error = target - output_layer_out

    # Calculate error term for output layer
    output_error_term = error * output_layer_out * (1 - output_layer_out)

    # Calculate error term for hidden layer
    hidden_error_term = output_error_term * weights_hidden_to_output * \
                        hidden_layer_out * (1 - hidden_layer_out)

    # Calculate change in weights for hidden layer to output layer
    delta_w_h_o = learnrate * output_error_term * hidden_layer_out

    # Calculate change in weights for input layer to hidden layer
    delta_w_i_h = learnrate * hidden_error_term * x[:, None]

    return delta_w_h_o, delta_w_i_h
```

## VI. Key Considerations

1. Vanishing Gradient Problem: The sigmoid function's maximum derivative is 0.25, which can lead to very small weight updates 
in deep networks.

2. NumPy Broadcasting: Utilize NumPy's broadcasting capabilities for efficient matrix operations.

3. Error Propagation: The error is scaled by the weights as it propagates backwards, reflecting each unit's contribution to the final output.

## VII. Conclusion

Backpropagation is a powerful algorithm that enables the training of multilayer neural networks. By efficiently computing 
gradients for all layers, it allows networks to learn complex, non-linear mappings from inputs to outputs. Understanding 
backpropagation is crucial for implementing, optimizing, and debugging neural networks.


# Implementing Backpropagation

Now we've seen that the error term for the output layer is:

δₖ = (yₖ - ŷₖ) f'(aₖ)

and the error term for the hidden layer is:

δⱼ = ∑[wⱼₖδₖ] f'(hⱼ)

For now we'll only consider a simple network with one hidden layer and one output unit. Here's the general algorithm for updating the weights with backpropagation:

1. Set the weight steps for each layer to zero
   - The input to hidden weights Δwᵢⱼ = 0
   - The hidden to output weights ΔWⱼ = 0

2. For each record in the training data:
   - Make a forward pass through the network, calculating the output ŷ
   - Calculate the error gradient in the output unit, δₒ = (y - ŷ) f'(z) where z = ∑ⱼWⱼaⱼ, the input to the output unit.
   - Propagate the errors to the hidden layer δⱼʰ = δₒWⱼf'(hⱼ)
   - Update the weight steps, where η is the learning rate:
     - ΔWⱼ = ηδₒaⱼ
     - Δwᵢⱼ = ηδⱼʰaᵢ

3. Update the weights, where m is the number of records:
   - Wⱼ = Wⱼ + ΔWⱼ/m
   - wᵢⱼ = wᵢⱼ + Δwᵢⱼ/m

4. Repeat for e epochs.

This algorithm outlines the process of implementing backpropagation for a simple neural network with one hidden layer and 
one output unit.



```textmate
def update_weights(weights_input_to_hidden, weights_hidden_to_output, 
                 features, targets, learnrate):
    """
    Complete a single epoch of gradient descent and return updated weights
    """
    delta_w_i_h = np.zeros(weights_input_to_hidden.shape)
    delta_w_h_o = np.zeros(weights_hidden_to_output.shape)

    # Loop through all records, x is the input, y is the target
    for x, y in zip(features.values, targets):
        ## Forward pass ##

        # Calculate the output using the forward_pass function.
        hidden_layer_out, output_layer_out = forward_pass(x,
            weights_input_to_hidden, weights_hidden_to_output)

        ## Backward pass ##

        # Calculate the change in weights using the backward_pass
        # function.
        delta_w_h_o_tmp, delta_w_i_h_tmp = backward_pass(x, y, learnrate,
            hidden_layer_out, output_layer_out, weights_hidden_to_output)
        delta_w_h_o += delta_w_h_o_tmp
        delta_w_i_h += delta_w_i_h_tmp

    n_records = features.shape[0]
    # Update weights  (don't forget division by n_records or number
    # of samples).
    weights_input_to_hidden += delta_w_i_h / n_records
    weights_hidden_to_output += delta_w_h_o / n_records

    return weights_input_to_hidden, weights_hidden_to_output
```

––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


Project Summary


As a machine learning engineer, you’ve gained familiarity with deep learning, a powerful tool for a number of tasks – not 
the least of which is computer vision. As part of a new product, you’ve been asked to prototype a system for optical character 
recognition (OCR) on handwritten characters. Since the team is still collecting samples of data, you’ve been tasked with 
providing a proof of concept on the MNIST database of handwritten digits, a task with very similar input and output.

In this project, you will be given a Jupyter Notebook to do all of your coding and written explanations. You will preprocess 
a dataset for handwritten digit recognition, build a neural network, then train and tune that neural network using your data.

Before you get started, please review the Project Rubric page ahead, and ensure to see the Jupyter Notebook on the Environments 
page to get started.



