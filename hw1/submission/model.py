import numpy as np

class LogisticRegression:
    def __init__(self):
        """
        Initialize `self.weights` properly. 
        Recall that for binary classification we only need 1 set of weights (hence `num_classes=1`).
        We have given the default zero intialization with bias term (hence the `d+1`).
        You are free to experiment with various other initializations including random initialization.
        Make sure to mention your initialization strategy in your report for this task.
        """
        self.num_classes = 1 # single set of weights needed
        self.d = 2 # input space is 2D. easier to visualize
        self.weights = np.zeros((self.d+1, self.num_classes))
        self.change = self.weights
    
    def preprocess(self, input_x):
        """
        Preprocess the input any way you seem fit.
        """        

        mean = np.mean(input_x, axis=0)
        std = np.std(input_x, axis=0)
        processed_x = (input_x - mean) / std
        
        # Scale the features to a specific range (e.g., [0, 1])
        min_vals = np.min(processed_x, axis=0)
        max_vals = np.max(processed_x, axis=0)
        scaled_x = (processed_x - min_vals) / (max_vals - min_vals)
        return scaled_x

    def sigmoid(self, x):
        """
        Implement a sigmoid function if you need it. Ignore otherwise.
        """
        return np.tanh(x)

    def calculate_loss(self, input_x, input_y):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: a single scalar value corresponding to the loss.
        """

        processed_x = self.preprocess(input_x)
        extended_x = np.hstack((processed_x, np.ones((processed_x.shape[0], 1)) ))
        predictions = self.sigmoid(np.dot(extended_x, self.weights))
        loss = -np.mean(input_y * np.log(predictions) + (1 - input_y) * np.log(1 - predictions))
        return loss

    def calculate_gradient(self, input_x, input_y):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: the gradient of loss function wrt weights.
        Ensure that gradient.shape == self.weights.shape.
        """
        # print(input_y.shape)
        input_y = input_y[:, np.newaxis]
        # print(input_y.shape)
        processed_x = self.preprocess(input_x)
        extended_x = np.hstack((processed_x, np.ones((processed_x.shape[0], 1)) ))
        predictions = self.sigmoid(np.dot(extended_x, self.weights))
        gradient_w = np.dot(extended_x.T, predictions - input_y) / input_x.shape[0]
        # gradient = logreg.calculate_gradient(input_x, input_y[:, np.newaxis])
        # print(gradient_w.shape, self.weights.shape)
        return gradient_w

    def update_weights(self, grad, learning_rate, momentum):
        """
        Arguments:
        grad -- NumPy array with same shape as `self.weights`
        learning_rate -- scalar
        momentum -- scalar
        Returns: nothing
        The function should update `self.weights` with the help of `grad`, `learning_rate` and `momentum`
        """
        # self.change
        # gd = self.change*momentum * grad
        self.weights = self.weights - (self.change*momentum+learning_rate * grad)
        self.change = (self.change*momentum+learning_rate * grad)

    def get_prediction(self, input_x):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        Returns: a NumPy array with shape (N,) 
        The returned array must be the list of predicted class labels for every input in `input_x`
        """
        processed_x = self.preprocess(input_x)
        extended_x = np.hstack((processed_x, np.ones((processed_x.shape[0], 1)) ))
        predictions = self.sigmoid(np.dot(extended_x, self.weights))
        return (predictions > 0.5).astype(int).flatten()

class LinearClassifier:
    def __init__(self):
        """
        Initialize `self.weights` properly. 
        We have given the default zero intialization with bias term (hence the `d+1`).
        You are free to experiment with various other initializations including random initialization.
        Make sure to mention your initialization strategy in your report for this task.
        """
        self.num_classes = 3 # 3 classes
        self.d = 4 # 4 dimensional features
        self.weights = np.zeros((self.d+1, self.num_classes))
        # print(self.weights.shape)
        self.change = self.weights
    
    def preprocess(self, train_x):
        """
        Preprocess the input any way you seem fit.
        """
        return train_x

    def sigmoid(self, x):
        """
        Implement a sigmoid function if you need it. Ignore otherwise.
        """
        return 1 / (1 + np.exp(-x))

    def calculate_loss(self, input_x, input_y):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: a single scalar value corresponding to the loss.
        """
        processed_x = self.preprocess(input_x)
        extended_x = np.hstack((processed_x, np.ones((processed_x.shape[0], 1))))
        scores = np.dot(extended_x, self.weights)
        probabilities = self.sigmoid(scores)
        # print(input_y.shape)
        # input_y = np.eye(self.num_classes)[input_y]
        # print(input_y.shape)
        loss = -np.mean(np.log(probabilities[np.arange(probabilities.shape[0]), input_y]))
        return loss

    def calculate_gradient(self, input_x, input_y):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: the gradient of loss function wrt weights.
        Ensure that gradient.shape == self.weights.shape.
        """
        # print(input_y)
        # print(input_y.shape)
        # input_y = np.eye(self.num_classes)[input_y]
        # print(input_y.shape)
        processed_x = self.preprocess(input_x)
        
        extended_x = np.hstack((processed_x, np.ones((processed_x.shape[0], 1))))
        scores = np.dot(extended_x, self.weights)
        probabilities = self.sigmoid(scores)
        
        # Compute gradient
        gradient_w = np.dot(extended_x.T, probabilities - np.eye(self.num_classes)[input_y]) / input_x.shape[0]
        return gradient_w

    def update_weights(self, grad, learning_rate, momentum):
        """
        Arguments:
        grad -- NumPy array with same shape as `self.weights`
        learning_rate -- scalar
        momentum -- scalar
        Returns: nothing
        The function should update `self.weights` with the help of `grad`, `learning_rate` and `momentum`
        """

        self.weights = self.weights - (self.change*momentum+learning_rate * grad)
        self.change = (self.change*momentum+learning_rate * grad)

    def get_prediction(self, input_x):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        Returns: a NumPy array with shape (N,) 
        The returned array must be the list of predicted class labels for every input in `input_x`
        """
        processed_x = self.preprocess(input_x)
        extended_x = np.hstack((processed_x, np.ones((processed_x.shape[0], 1))))
        scores = np.dot(extended_x, self.weights)
        probabilities = self.sigmoid(scores)
        predictions = np.argmax(probabilities, axis=1)
        return predictions
