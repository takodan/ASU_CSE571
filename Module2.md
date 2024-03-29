# Introduction to Neural Networks (Neural Networks & PyTorch)
## OVERVIEW
1. Demonstrate how an artificial neural network functions, in order to produce predicted outcomes. 
2. Evaluate the outputs of a neural network
3. Explain how a neural network is trained by adjusting the weight of each neural.
4. Recall how back propagation works. 
5. Calculate the gradient of loss function.

# Unit1


## Back Propagation
1. a tuple (x, y), x is input y is output
    1. we want to train network such that: NN(x) ≒ y, ∀ x ∈ X
    2. typically labels(output) are provided by human annotation註解
2. back propagation algorithm
    1. algorithm for training the weights
    2. usually start with radom weight, then adjust it to reduce error
    3. error(deviation) defined by loss function E  
    4. common E is quadratic error, E = 1/2∑||y_i-(y_i)hat||^2, half (sum of ((target minus output of ANN)squared))
    5. using gradient descent梯度下降 to minimizing E
3. intuition of back propagation
    1. calculate E at final layer
    2. using the chain rule to calculate change to weight layer by layer

## Back Propagation in depth
1. mathematics process


# Unit2
## Best Practices for Training ANNs
1. input normalization標準化
    1. same range of input value helps improve learning quality
    2. mean: μ = (1/N)∑x_i 平均值
    3. variance: σ^2 = (1/N)∑(x_i-μ)^2 方差/變異數
    4. (x_i)hat = (x_i-μ)/(σ^2)^(1/2)
2. Generalization
    1. make sure NN has good performance both on the training data and new data, avoiding overfitting
    2. separate the data to training set and test set 8:2 or 7:3
    3. training on the training set, but observe error on both sets
    4. Early stopping
        1. stop iteration when error on the test set going up
    5. K-Fold Cross-validation
        1. divide data in K
        2. using different batch in the data set as test set, train K times
        3. true error is average of individual errors on each training

## Example 1: Learning to Predict Collisions
1. goal was to predict collisions
2. 5 input(sensor values), 1 hidden layer(200 neurons), 1 output(collision)
3. Action-Condition predictive models
    1. take into account the robots action
    2. cannot disambiguate消除歧義 between situation where now collision is dependent on action
    3. f(s_t, a), current state and next action, predict next state(collision)
    4. add 1 input(steering angle) in to the module
    5. module has prediction because of 1 extra input

## Introduction to PyTorch
1. an open source ML library for Python
2. allows for dynamic network
3. 

