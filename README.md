# Neural-Network-Intermediate-Layer-Visualization

Dataset
We will use the MNIST dataset, which consists of grayscale images of handwritten digits from 0 to 9. Each image is of size 28x28 pixels. You can download the dataset from the official site (http://yann.lecun.com/exdb/mnist/) or use the MNIST dataset available in PyTorch (https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html).
Technical Aspects
Dependencies
We will use the following libraries for this project:
* NumPy: For numerical computations
* Torch: For building and training the CNN
* Torchvision: For data loading and transformations
* Matplotlib: For visualizations
Model Architecture
The CNN model used in this project is a modified version of the classic LeNet-5. The architecture is as follows:
* 		Input Image (28x28)
* 		Convolutional Layer 1: 6 (5x5) filters, ReLU activation, stride 1x1, and 2x2 padding
* 		Max Pooling Layer 1: 2x2 size, stride 2x2
* 		Convolutional Layer 2: 16 (5x5) filters, ReLU activation, stride 1x1, and no padding
* 		Max Pooling Layer 2: 2x2 size, stride 2x2
* 		Flatten
* 		Fully Connected Layer 1: 120 neurons, ReLU activation
* 		Fully Connected Layer 2: 84 neurons, ReLU activation
* 		Fully Connected Layer 3: 10 neurons (output layer), log softmax activation
The output class probabilities are obtained using a log softmax function. The input image is reshaped into 28x28 to match the model's input size.
Training
We will train the LeNet-5 model using the following hyperparameters:
* Random Seed: 42
* Learning Rate: 0.001
* Batch Size: 128
* Number of Epochs: 15
We will use the Adam optimizer and the CrossEntropyLoss as the loss function during training.
Visualization of Intermediate Activations
To gain insights into the patterns learned by each layer, we will visualize the intermediate activations of the model. This involves mapping the activations back to the pixel space of the input image using a Deconvnet approach.
The Deconvnet is attached to each layer of the Convnet. We will selectively deactivate the activations of all other layers and feed the feature maps to the corresponding attached Deconvnet layer. In the Deconvnet, we perform unpooling to determine the locations of maxima within each pooling region. This information helps us restore the appropriate locations from the layer above during the reconstruction process. After unpooling, we apply rectification to the feature maps to ensure positive values, just like in the Convnet. This process is iteratively repeated until we reach the input pixel space.
By following this approach, we can visualize the input patterns that contributed to specific activations within the Convnet layers, providing valuable insights into the learned representations.
