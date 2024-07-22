# Unit1
## Convolutional Neural Network
1. Computer Vision: detection, recognition, modification, clean-up, synthesis合成
2. basic characterization:
    1. convolution layer: convolution + ReLU
    2. pooling
3. RGB image: representation as matrices with three channels for red, green, blue
4. in computer vision we usually start with filtering
    1. process image at local level and try to extract feature to higher level
    2. some filter example:
        1. blurring
        2. sharpening
        3. denoising
        4. edge detection
        5. point of interest
        6. template matching
5. convolution
    1. a matrix represent weigh, this matrix call "mask"/"kernel"
    2. run over the entire image multiply every pixel
    3. the new value will overwrite the original pixel
6. Gaussian blur
    1. high weight at the center, decrease weight as we going to the boundary
    2. weight is generate from 2D Gaussian distribution

## generate the kernels and pooling
1. kernel traditionally is engineered by human
2. now it automatically generated in CNN
    1. the weight in a kernel can be seen as input weight in neural network
    2. input and kernel generate activation map
    3. "stride" determines the step size
    4. activation function usually use ReLU
3. pooling
    1. max pooling: only takes the maximum value in the local of activation map
4. frequent operation in CNN
    1. image(input)
    2. convolution layer: convolution + ReLU
    3. max pooling
    4. convolution layer
    5. max pooling
    6. ...
    7. flattering
    8. traditional neural network: for classification

## training a CNN
1. typically requires a large number of training set to generalize different condition
2. number of convolutional layers is critical
3. can use "dropout" or start with pre-trained network like:
    1. VGGNet
    2. InceptionNet
    3. AlexNet
4. Example
    1. face detection
    2. robot dodgeball


# Unit2
## Identifying digits with a CNN
1. using MNIST database
    1. handwritten labeled digits: 60k training set, 10k  testing set
2. input: image, output: log probability
3. log probability
    1. more efficient computation: addition replaces multiplication
    2. p' = log2(p)
    3. yelds only negative value( 2^(-x) = 1/(2^x); 2^(-1) = 1/2 = 0.5)
    4. value closer to 0 are more likely( 2^(0) = 1/1 = 1)
4. example network architecture
    1. input image: 1x28x28
    2. convolutional layer: 8x28x28
    3. max pooling: 8x14x14
    4. convolutional layer: 16x14x14
    5. max pooling: 16x7x7
    6. flatten: 1x784
    7. linear layer: 64
    8. linear layer: 10
    9. output