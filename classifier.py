# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.
import numpy as np

# Classifier
#
# The classifier which predicts/decides which move pacman will make.
class Classifier:

    # Constructor.
    def __init__(self):
        self.model = MLPNeuralNetwork(0.01)
        self.model.add(DenseLayer(25, 'sigmoid'))
        self.model.add(DenseLayer(15, 'ReLU'))
        self.model.add(DenseLayer(8, 'softmax'))

    def reset(self):
        pass
    
    # Fits the model to the data.
    def fit(self, data, target):
        X = np.array(data)
        y = np.array(target)
        self.model.fit(X, y)

    # Obtains a prediction based on the data provided.
    def predict(self, data, legal=None):
        return self.model.predict(data)
    
# MLPNeuralNetwork
#
# A multilayer perceptron neural network.
class MLPNeuralNetwork():

    # Constructor.
    def __init__(self, learning_rate):
        self.layers = []
        self.learning_rate = learning_rate

    # Adds a layer to the neural network and adjusts the shapes of
    # weight and bias matrices in the previous layer accordingly.
    def add(self, layer):
        if self.layers:
            prev_layer = self.layers[-1]
            self.layers.append(layer)
            prev_layer.reshape_weights(layer.n_nodes)
            prev_layer.reshape_biases(layer.n_nodes)
        else:
            self.layers.append(layer)

    # Fits the model to the training data (X, y) with a given number of epochs. 
    def fit(self, X, y, epochs=50):
        # Adjust the shapes of weight and bias matrices in the final
        # hidden layer to match number of classes.
        final_hidden_layer = self.layers[-1]
        number_of_classes = np.unique(y).shape[0]
        final_hidden_layer.reshape_weights(number_of_classes)
        final_hidden_layer.reshape_biases(number_of_classes)

        # Performs Forward Propagation along with Backwards Propagation
        # following Stochastic Gradient Descent for each epoch.
        for i in range(epochs):
            
            # Stores the number of correct predictions made by the model
            # for each epoch.
            matches = 0
            for sample, label in zip(X, y):
                # Convert numpy vector into a numpy matrix.
                sample = np.reshape(sample, (25, 1))
                label = np.reshape(label, (1, 1))

                output, values = self.forward_propagation(sample)

                # One hot encoding of label.
                true_dist = np.zeros((output.shape[0], 1))
                true_dist[label] = 1

                # Obtain a prediction from the probability distribution
                # (output).
                y_hat = np.argmax(output)

                matches += int(y_hat == label[0][0])
                
                # Uses the mean squared error as the loss function.
                # loss = Loss().mean_squared_error(true_dist, output)
                # print('Loss:', loss)
                
                self.back_propagation(values, output, true_dist, sample)

            print(f'Accuracy after epoch {i + 1}: {(matches/X.shape[0]) * 100}')
            
    # Obtains a prediction based on the data provided.
    def predict(self, x):
        output, values = self.forward_propagation(np.reshape(x, (25, 1)))
        return np.argmax(output)

    # Performs Forward Propagation and returns the outputs of each
    # layer in a list.
    def forward_propagation(self, inputs):
        values = []
        for layer in self.layers:
            weights = layer.weights
            biases = layer.biases
            z = biases + (weights @ inputs)
            outputs = ActivationFunction().activate(layer.activation_function, z)
            
            # Convert numpy vector into a numpy matrix.
            outputs = np.reshape(outputs, (outputs.shape[0], 1))
            values.append(outputs)

            # Outputs for one layer become the inputs for the next layer.
            inputs = outputs
        
        return outputs, values

    # Performs Backwards Propagation following Stochastic Gradient Descent.
    def back_propagation(self, values, output, true_dist, x):
        output_delta = output - true_dist
        deltas = []
        deltas.append(output_delta)

        # Updating weights and biases of final hidden layer.
        self.layers[-1].weights = self.layers[-1].weights - self.learning_rate * (deltas[-1] @ values[-2].T)
        self.layers[-1].biases = self.layers[-1].biases - self.learning_rate * deltas[-1]

        # Updating weights and biases of hidden layers between
        # the first and last hidden layers.
        for i in range(-2, -len(self.layers), -1):
            current_layer_delta = (self.layers[i + 1].weights.T @ deltas[-1]) *  ActivationFunction().derivative(self.layers[i].activation_function, values[i])
            deltas.append(current_layer_delta)
            self.layers[i].weights = self.layers[i].weights - self.learning_rate * (deltas[-1] @ values[i - 1].T)
            self.layers[i].biases = self.layers[i].biases - self.learning_rate * deltas[-1]

        # Updating weights and biases of first hidden layer.
        first_layer_delta = (self.layers[1].weights.T @ deltas[-1]) * ActivationFunction().derivative(self.layers[0].activation_function, values[0])
        deltas.append(first_layer_delta)
        self.layers[0].weights = self.layers[0].weights - self.learning_rate * (deltas[-1] @ x.T)
        self.layers[0].biases = self.layers[0].biases - self.learning_rate * deltas[-1]

# DenseLayer
#
# A single Dense Layer in the neural network. 
class DenseLayer():

    # Constructor.
    def __init__(self, n_nodes, activation_function):
        self.weights = np.random.rand(n_nodes)
        self.biases = np.random.rand(n_nodes)
        self.activation_function = activation_function
        self.n_nodes = n_nodes
    
    # Reshapes the weight matrix in order to match the number of 
    # nodes in the next layer.
    def reshape_weights(self, n_nodes_in_next_layer):
        self.weights = np.random.uniform(-1, 1, (n_nodes_in_next_layer, self.n_nodes))

    # Reshapes the bias matrix in order to match the number of 
    # nodes in the next layer. 
    def reshape_biases(self, n_nodes_in_next_layer):
        self.biases = np.zeros((n_nodes_in_next_layer, 1))

# ActivationFunction
#
# Computes different activation functions and their derivatives.
class ActivationFunction():

    # Computes the activation for the outputs of each layer.
    def activate(self, function, x):
        if function == 'ReLU':
            f = np.vectorize(lambda t: max(0, t))
            return f(x)
        elif function == 'sigmoid':
            f = np.vectorize(lambda t: 1 / (1 + np.exp(-t)))
            return f(x)
        elif function == 'softmax':
            return np.exp(x) / np.exp(x).sum()

    # Computes the derivative for the functions.
    def derivative(self, function, x):
        if function == 'ReLU':
            return x > 0
        elif function == 'sigmoid':
            return x * (1 - x)
        
# Loss
#
# Computes the loss according to the true distribution and the 
# predicted distribution.
class Loss():
    def categorical_cross_entropy_loss(self, true_dist, predicted_dist):
        loss = - (true_dist.T @ np.log(predicted_dist))
        return loss
    
    def mean_squared_error(self, true_dist, predicted_dist):
        squared_errors = (true_dist - predicted_dist) ** 2
        return (1/true_dist.shape[0]) * squared_errors.sum()