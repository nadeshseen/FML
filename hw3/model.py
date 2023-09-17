import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl

from fun import Gaussian, Bernoulli, Multinomial, Exponential, Laplace

class NaiveBayes:

    def fit(self, X, y):

        """Start of your code."""
        """
        X : np.array of shape (n,2)
        y : np.array of shape (n,)
        Create a variable to store number of unique classes in the dataset.
        Assume Prior for each class to be ratio of number of data points in that class to total number of data points.
        Fit a distribution for each feature for each class.
        Store the parameters of the distribution in suitable data structure, for example you could create a class for each distribution and store the parameters in the class object.
        You can create a separate function for fitting each distribution in its and call it here.
        """

        self.classes = np.unique(y)
        classes = np.unique(y)
        self.n_classes = classes.shape[0]
        P = []
        for i in range(self.n_classes):
            outcome = classes[i]
            outcome_count = sum(y == outcome)
            P.append( outcome_count/len(y))
        self.prioris = P

        # 1. (X1, X2) are drawn independently from two different univariate Gaussian distributions.
        self.x1 = Gaussian()
        self.x1.fit(X[:,0], y)
        self.x2 = Gaussian()
        self.x2.fit(X[:,1], y)

        # 2. (X3, X4) are random variables drawn independently from two different Bernoulli Distributions
        self.x3 = Bernoulli()
        self.x3.fit(X[:,2], y)
        self.x4 = Bernoulli()
        self.x4.fit(X[:,3], y)

        # 3. (X5, X6) are random variables drawn independently from two different Laplace Distributions
        self.x5 = Laplace()
        self.x5.fit(X[:,4], y)
        self.x6 = Laplace()
        self.x6.fit(X[:,5], y)

        # 4. (X7, X8) are random variables drawn independently from two different Exponential Distribution
        self.x7 = Exponential()
        self.x7.fit(X[:,6], y)
        self.x8 = Exponential()
        self.x8.fit(X[:,7], y)

        # 5. (X9, X10) are random variables drawn independently from two different Multinomial Distribution
        self.x9 = Multinomial()
        self.x9.fit(X[:,8], y)
        self.x10 = Multinomial()
        self.x10.fit(X[:,9], y)

        """End of your code."""

    def getLogLike(self, x):
        ll = []
        for i in x:
            if (i > 0):
               ll.append(np.log(i))
            else:
               ll.append(0.0)

        return np.array(ll)

    def posterior(self, X):
        Py_given_x = self.getLogLike(self.prioris) + \
                     self.getLogLike(self.x1.pdf(X[0])) + \
                     self.getLogLike(self.x2.pdf(X[1])) + \
                     self.getLogLike(self.x3.pdf(X[2])) + \
                     self.getLogLike(self.x4.pdf(X[3])) + \
                     self.getLogLike(self.x5.pdf(X[4])) + \
                     self.getLogLike(self.x6.pdf(X[5])) + \
                     self.getLogLike(self.x7.pdf(X[6])) + \
                     self.getLogLike(self.x8.pdf(X[7])) + \
                     self.getLogLike(self.x9.pdf(X[8])) + \
                     self.getLogLike(self.x10.pdf(X[9])) 

        #print("PY = ", Py_given_x)
        return Py_given_x

    def predict(self, X):
        """Start of your code."""
        """
        X : np.array of shape (n,2)

        Calculate the posterior probability using the parameters of the distribution calculated in fit function.
        Take care of underflow errors suitably (Hint: Take log of probabilities)
        Return an np.array() of predictions where predictions[i] is the predicted class for ith data point in X.
        It is implied that prediction[i] is the class that maximizes posterior probability for ith data point in X.
        You can create a separate function for calculating posterior probability and call it here.
        """
        predicted = []
        for i in range(X.shape[0]):
            indices = np.argmax(self.posterior(X[i]))
            predicted.append(self.classes[indices])

        return np.array(predicted)
        """End of your code."""

    def getParams(self):
        """
        Return your calculated priors and parameters for all the classes in the form of dictionary that will be used for evaluation
        Please don't change the dictionary names
        Here is what the output would look like:
        priors = {"0":0.2,"1":0.3,"2":0.5}
        gaussian = {"0":[mean_x1,mean_x2,var_x1,var_x2],"1":[mean_x1,mean_x2,var_x1,var_x2],"2":[mean_x1,mean_x2,var_x1,var_x2]}
        bernoulli = {"0":[p_x3,p_x4],"1":[p_x3,p_x4],"2":[p_x3,p_x4]}
        laplace = {"0":[mu_x5,mu_x6,b_x5,b_x6],"1":[mu_x5,mu_x6,b_x5,b_x6],"2":[mu_x5,mu_x6,b_x5,b_x6]}
        exponential = {"0":[lambda_x7,lambda_x8],"1":[lambda_x7,lambda_x8],"2":[lambda_x7,lambda_x8]}
        multinomial = {"0":[[p0_x9,...,p4_x9],[p0_x10,...,p7_x10]],"1":[[p0_x9,...,p4_x9],[p0_x10,...,p7_x10]],"2":[[p0_x9,...,p4_x9],[p0_x10,...,p7_x10]]}
        """

        """Start your code"""

        #print(self.prioris)
        priors = dict(zip(range(len(self.prioris)),self.prioris))

        mu, sigma = self.x1.getParams()
        mu1, sigma1 = self.x2.getParams()
        guassian = dict()
        for i in mu:
            guassian[i] = [mu[i], mu1[i], sigma[i], sigma1[i]]

        p = self.x3.getParams()
        p1= self.x4.getParams()
        tmp = [ [i, j] for i, j in zip(p, p1)]
        bernoulli = dict(zip(range(len(tmp)),tmp))

        mu, sigma = self.x5.getParams()
        mu1, sigma1 = self.x6.getParams()
        laplace = dict()
        for i in mu:
            laplace[i] = [mu[i], mu1[i], sigma[i], sigma1[i]]

        p = self.x7.getParams()
        p1= self.x8.getParams()
        tmp = [ [i, j] for i, j in zip(p, p1)]
        exponential = dict(zip(range(len(tmp)),tmp))

        p = self.x9.getParams()
        p1= self.x10.getParams()
        tmp = [ [i, j] for i, j in zip(p, p1)]
        multinomial = dict(zip(range(len(tmp)),tmp))

        
        """End your code"""
        return (priors, guassian, bernoulli, laplace, exponential, multinomial)        


def save_model(model,filename="model.pkl"):
    """

    You are not required to modify this part of the code.

    """
    file = open("model.pkl","wb")
    pkl.dump(model,file)
    file.close()

def load_model(filename="model.pkl"):
    """

    You are not required to modify this part of the code.

    """
    file = open(filename,"rb")
    model = pkl.load(file)
    file.close()
    return model

def visualise(data_points,labels, name):
    """
    datapoints: np.array of shape (n,2)
    labels: np.array of shape (n,)
    """

    plt.figure(figsize=(8, 6))
    plt.scatter(data_points[:, 0], data_points[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.title('Generated 2D Data from 5 Maussian Distributions')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


def net_f1score(predictions, true_labels):
    """Calculate the multclass f1 score of the predictions.
    For this, we calculate the f1-score for each class 

    Args:
        predictions (np.array): The predicted labels.
        true_labels (np.array): The true labels.

    Returns:
        float(list): The f1 score of the predictions for each class
    """

    def precision(predictions, true_labels, label):
        """Calculate the multclass precision of the predictions.
        For this, we take the class with given label as the positive class and the rest as the negative class.

        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.

        Returns:
            float: The precision of the predictions.
        """
        """Start of your code."""
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for pred, gt in zip(predictions, true_labels):
            if gt == label and pred == label:
               tp += 1
            elif gt != label and pred != label:
               tn += 1
            elif gt != label and pred == label:
               fp += 1
            elif gt == label and pred != label:
               fn += 1
        

        #print(tp, tn, fp, tn)
        prec = tp/(tp + fp) if tp > 0 else 0
        return prec

        """End of your code."""


    def recall(predictions, true_labels, label):
        """Calculate the multclass recall of the predictions.
        For this, we take the class with given label as the positive class and the rest as the negative class.
        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.

        Returns:
            float: The recall of the predictions.
        """
        """Start of your code."""
        
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for pred, gt in zip(predictions, true_labels):
            if gt == label and pred == label:
               tp += 1
            elif gt != label and pred != label:
               tn += 1
            elif gt != label and pred == label:
               fp += 1
            elif gt == label and pred != label:
               fn += 1
        

        recall = tp/(tp + fn)
        return recall

        """End of your code."""
        

    def f1score(predictions, true_labels, label):
        """Calculate the f1 score using it's relation with precision and recall.

        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.

        Returns:
            float: The f1 score of the predictions.
        """

        """Start of your code."""
        p = precision(predictions, true_labels, label)
        r = recall(predictions, true_labels, label)
        f1 = 2* p * r / (p + r) if p > 0 else 0

        """End of your code."""
        return f1
    

    f1s = []
    for label in np.unique(true_labels):
        f1s.append(f1score(predictions, true_labels, label))
    return f1s

def accuracy(predictions,true_labels):
    """

    You are not required to modify this part of the code.

    """
    return np.sum(predictions==true_labels)/predictions.size



if __name__ == "__main__":
    """

    You are not required to modify this part of the code.

    """

    # Load the data
    train_dataset = pd.read_csv('./data/train_dataset.csv',index_col=0).to_numpy()
    validation_dataset = pd.read_csv('./data/validation_dataset.csv',index_col=0).to_numpy()

    # Extract the data
    train_datapoints = train_dataset[:,:-1]
    train_labels = train_dataset[:, -1]
    validation_datapoints = validation_dataset[:, 0:-1]
    validation_labels = validation_dataset[:, -1]

    # Visualize the data
    visualise(train_datapoints, train_labels, "train_data.png")

    # Train the model
    model = NaiveBayes()
    model.fit(train_datapoints, train_labels)

    # Make predictions
    train_predictions = model.predict(train_datapoints)
    validation_predictions = model.predict(validation_datapoints)

    # Calculate the accuracy
    train_accuracy = accuracy(train_predictions, train_labels)
    validation_accuracy = accuracy(validation_predictions, validation_labels)

    # Calculate the f1 score
    train_f1score = net_f1score(train_predictions, train_labels)
    validation_f1score = net_f1score(validation_predictions, validation_labels)

    # Print the results
    print('Training Accuracy: ', train_accuracy)
    print('Validation Accuracy: ', validation_accuracy)
    print('Training F1 Score: ', train_f1score)
    print('Validation F1 Score: ', validation_f1score)

    #print(model.getParams())
    # Save the model
    save_model(model)

    # Visualize the predictions
    visualise(validation_datapoints, validation_predictions, "validation_predictions.png")
