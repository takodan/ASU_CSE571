# Unit2
## Dropout
1. overfitting: model being too "obsessed著迷" with the training data
2. training data may have some noise, and the model is being to too sensitive
3. underfit -> fit -> overfit
4. solution1: Early stopping
    1. monitoring Errors on training set and test set
    2. stop training when error on test set begins increasing
5. solution2: Dropout
    1. the idea is calculated multiple trained neural networks' mean without actually training them
    2. during forward training, randomly dropout neurons
    3. choose dropout neurons by probability, we can do this by simply multiply the matrix
    4. the matrix have random values zero and one along the diagonal對角線
    5. remember use the originally network and scaled it by probability when evaluation
    6. the method prevents over-reliance on specific inputs

## Modeling Uncertainty
1. dropout can also be used for generating probability distribution
2. a typical neural network works like a function, a single mapping of an n-dimensional input to an m-dimensional output
3. we can not just generating one output but multiple output, distribution of output
4. why distribution? because the real world is non-deterministic不確定的
5. Monte Carlo Dropout
    1. using dropout at inference
    2. stochastic隨機的 forward passes
    3. dropout some of the neural makes numbers of different network configuration
    4. a input can make numbers of configuration output
    5. we can also use this method to evaluate the network is certain or not by analysis the variance
