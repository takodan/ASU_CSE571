# Introduction to Machine Learning
## OVERVIEW
1. Describe what AI is and what AI related topics we'll cover in this course.
2. Understand the three main models of Machine Learning, Unsupervised Learning, Supervised Learning and Reinforcement Learning
3. Determine which of three main models of Machine Learning would be appropriate for a given situation.
4. Describe how loss function is utilized to increase the accuracy of model predictions.  
5. Explain how object recognition generates models with few data points, using either classification or regression techniques.
6. Test Machine Learning models for optimization and determine optimal model parameters to minimize loss.


# Unit1
## Machine Learning
1. ML is part of AI. It focuses on learning structure and relationship in data.
2. ML is learning rules with out programmed by human to make a determination or prediction
3. ML compare to typical programming:
    1. typical programming paradigm: (input) -> (programming by expert) -> (output)
    2. ML(supervised learning): (data set of input and output(demonstration, example)) -> (ML), (new input) -> ML -> (right output)
4. type of ML
    1. supervised learning
    2. unsupervised learning
    3. reinforcement learning
5. vantage point
    1. learn function: f(input) -> output
    2. conditional probability distribution: p(outputs|input), more appealing when it has noise and uncertainty
6. deep learning
    1. same as artificial neural network, it's inspired by biologic
7. example applications
    1. medical classification
    2. anomaly detection in industry
    3. decision-making in autonomous driving
    4. speech recognition and generation
8. why now
    1. an abundance of data from sensors and the internet
    2. computation power has caught up
    3. new theoretical tricks and insights
    4. more investment

# Unit2
## unsupervised learning
1. only data, no labels
2. focus on finding structures and manifolds流形
3. usually by analysis distances and spatial relationships, distribution
4. unsupervised learning method
    1. clustering
        1. use distances to group individual points
        2. hard clustering: sign each points to only a single class
        3. soft clustering: using fuzzy output or probability to sign points to classes
        4. hierarchical clustering: multiple layers of class, like under class0 we can have class1 and class2
    2. manifold learning/ dimensionality reduction
        1. identify manifold structure (like a paper) and represent/project data in lower-dimensional space (make that paper as a dimension)
        2. can reprojection to higher-dimensional
        3. example: walking manifolds
            1. a man walking can have many dimension
            2. project to two dimension (like shadow on the wall)
            3. now we can draw a line on two dimension to represent walking movement
    3. unsupervised learning method: data distribution
        1. draw grids and calculate the distribution in a grid
        2. then we can have p(x|y) distribution
5. distance functions
    1. Euclidean distance: distance between two point
    2. Geodesic distance: distance alone a surface, especially when data on a manifolds

## supervised learning
1. most common form of machine learning
2. it is to learn a mapping between inputs to outputs. focus on accurate predictions
3. should work on new inputs
4. diabetes classification example
    1. training
        1. input patient's data like Blood pressure and BMI. these data are features
        2. label these data with they are sick or not as output
    2. inference
        1. input a new data with out label
        2. predict output (label)
5. object recognition example:
    1. input: image, features: pixel value
    2. output: type of object
6. types of supervised learning
    1. classification: assigning inputs to one of a discrete離散的 set of classes or categories (e.g. determining if an image shows a dog or cat).
    2. regression: learning a continuous mapping from inputs to output values (e.g. predicting a numerical value like housing prices).
7. example application of regression
    1. prediction and forecasting stock market, output of a power plant, weather
    2. creating models of real-world object
        1. also known as surrogate代理人 module
        2. trained to mimic and predict the behavior/output of a real-world process/system
        3. can test and provide the output more efficiently compare to physical tests
        4. for robot, there is a name call forward model
        5. include forward model in the instruction to control more precisely
8. loss function
    1. measures how well the machine learning model fits or describes the training data.
    2. the discrepancy差異 between labels from a model and the training data that is provided.
    3. The goal in most ML is to find a model that minimizes this loss

## reinforcement learning
1. agent trial and error by it self, teacher only provides feedback
2. similar to operant conditioning according to skinner, learning through reward and punishment
3. how reinforcement learning work
    1. agent repeatedly interacts with environment
    2. in each state the agent chooses an action
    3. reward is provided by a reward function
    4. the goal is learning a policy that maximizes sum of rewards
    5. policy take states and produces actions (a_t = π(s_t))
4. reward functions
    1. evaluates the current situation to produce reward (R(s_t))
    2. typically manually specified, so it's not always easy to do
5. challenges
    1. learn from fewer trails
    2. to specify reward functions
    3. learn in a safe way
    4. learn to solve multiple task
    5. how to understand a learned policy


# Unit3
