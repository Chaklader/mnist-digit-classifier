Introduction to Deep Learning


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

$$L(y, \hat{y}) = -\frac{1}{N} \sum_{i=1}^N [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$$

Where:
- $N$ is the number of samples
- $y_i$ is the true label (0 or 1)
- $\hat{y}_i$ is the predicted probability of the positive class

### B. Characteristics

1. Always positive: Log-Loss is always ≥ 0
2. Perfect prediction: Log-Loss = 0 when the model predicts the correct class with 100% confidence
3. Penalizes confident mistakes: Heavily penalizes predictions that are both confident and wrong

## II. Log-Loss in Multi-class Classification

For multi-class problems, the formula extends to:

$$L(y, \hat{y}) = -\frac{1}{N} \sum_{i=1}^N \sum_{j=1}^M y_{ij} \log(\hat{y}_{ij})$$

Where:
- $M$ is the number of classes
- $y_{ij}$ is 1 if sample $i$ belongs to class $j$, and 0 otherwise
- $\hat{y}_{ij}$ is the predicted probability that sample $i$ belongs to class $j$

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

For logistic regression (binary classification), the gradient of Log-Loss with respect to weights $w$ is:

$$\frac{\partial L}{\partial w} = \frac{1}{N} X^T (\hat{y} - y)$$

Where:
- $X$ is the input feature matrix
- $\hat{y}$ is the vector of predicted probabilities
- $y$ is the vector of true labels

### C. Parameter Update Rule

$$w_{new} = w_{old} - \alpha \frac{\partial L}{\partial w}$$

Where $\alpha$ is the learning rate.

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



### Maximum Likelihood

# Maximum Likelihood Estimation with Sigmoid Function

## I. Introduction

Maximum Likelihood Estimation (MLE) is a fundamental method in statistics and machine learning for estimating the parameters of a probability distribution. When combined with the sigmoid function, it forms the basis of logistic regression, a powerful tool for binary classification.

## II. The Sigmoid Function

### A. Definition

The sigmoid function, also known as the logistic function, is defined as:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

### B. Properties

1. Output range: (0, 1)
2. S-shaped curve
3. Symmetric around 0.5
4. Differentiable

### C. Use in Classification

In binary classification, the sigmoid function is used to map any real-valued number to a probability between 0 and 1.

## III. Logistic Regression Model

In logistic regression, we model the probability of the positive class as:

$$P(Y=1|X) = \sigma(w^T X + b)$$

Where:
- $X$ is the input feature vector
- $w$ is the weight vector
- $b$ is the bias term

## IV. Maximum Likelihood Estimation





### A. Likelihood Function



For a dataset with $n$ independent samples, the likelihood function is:

$$L(w, b) = \prod_{i=1}^n P(Y=y_i|X_i)$$

$$= \prod_{i=1}^n [\sigma(w^T X_i + b)]^{y_i} [1 - \sigma(w^T X_i + b)]^{1-y_i}$$

Where $y_i$ is the true label (0 or 1) for the i-th sample.

### B. Log-Likelihood

We typically work with the log-likelihood for computational convenience:

$$\ell(w, b) = \sum_{i=1}^n [y_i \log(\sigma(w^T X_i + b)) + (1-y_i) \log(1 - \sigma(w^T X_i + b))]$$

### C. Maximum Likelihood Estimator

The goal is to find $w$ and $b$ that maximize the log-likelihood:

$$(w^*, b^*) = \arg\max_{w, b} \ell(w, b)$$

## V. Optimization

### A. Gradient Ascent

We can use gradient ascent to find the maximum:

$$w := w + \alpha \frac{\partial \ell}{\partial w}$$
$$b := b + \alpha \frac{\partial \ell}{\partial b}$$

Where $\alpha$ is the learning rate.

### B. Gradients

The gradients of the log-likelihood with respect to $w$ and $b$ are:

$$\frac{\partial \ell}{\partial w} = \sum_{i=1}^n (y_i - \sigma(w^T X_i + b)) X_i$$

$$\frac{\partial \ell}{\partial b} = \sum_{i=1}^n (y_i - \sigma(w^T X_i + b))$$

## VI. Connection to Cross-Entropy Loss

Maximizing the log-likelihood is equivalent to minimizing the cross-entropy loss:

$$\text{Cross-Entropy} = -\frac{1}{n} \ell(w, b)$$

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



