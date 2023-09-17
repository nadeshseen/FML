import numpy as np
import pandas as pd

class Gaussian:
    def fit(self, x, y):
        classes = np.unique(y)
        n_classes = classes.shape[0]

        mu = dict()
        sigma = dict()

        for i in range(n_classes):
            outcome = classes[i]
            indices = np.argwhere(y == outcome).flatten()
            mean = np.mean(x[indices])
            stddev = np.std(x[indices])

            mu[i] = mean
            sigma[i] = stddev

        self.mu = mu
        self.sigma = sigma
        self.classes = classes
        self.n_classes = n_classes
        # print(mu)
        # print(sigma)
        return mu, sigma

    def getParams(self):
        return self.mu, self.sigma

    def pdf(self, value):
        pdfs = []
        for i in range(self.n_classes):
            numerator = np.exp(-np.power(float(value - self.mu[i]) / self.sigma[i], 2.0) / 2.0)
            denominator = np.sqrt(2 * np.pi) * self.sigma[i]
            pdfs.append(numerator/denominator)
        return pdfs

class Bernoulli:
    def fit(self, x, y):
        classes = np.unique(y)
        n_classes = classes.shape[0]

        count = []
        n_len = []
        for i in range(n_classes):
            outcome = classes[i]
            indices = np.argwhere(y == outcome).flatten()
            count_val = x[indices].sum(axis=0) + 1
            n_len_val = len(x[indices]) + 2
            count.append(count_val)
            n_len.append(n_len_val)

        self.p = np.array(count)/np.array(n_len)
        self.classes = classes
        self.n_classes = n_classes
        return self.p

    def getParams(self):
        return self.p

    def pdf(self, value):
        pdfs = []
        for i in range(self.n_classes):
            p_power = self.p[i] if value > 0 else 1 - self.p[i]
            pdfs.append( p_power )
        return pdfs

class Laplace:
    def fit(self, x, y):
        # 3. (X5, X6) are random variables drawn independently from two different Laplace Distributions
        classes = np.unique(y)
        n_classes = classes.shape[0]

        mu = dict()
        b = dict()

        for i in range(n_classes):
            outcome = classes[i]
            indices = np.argwhere(y == outcome).flatten()
            mean = np.median(x[indices])
            absdev = np.sum(np.abs(x[indices] - mean))/ len(x[indices])

            # correct = b*n/(n- 2)
            #absdev = absdev * len(x[indices])/(len(x[indices]) - 2)
            mu[i] = mean
            b[i] = absdev

        self.classes = classes
        self.n_classes = n_classes
        self.mu = mu
        self.b = b
        # print(mu)
        # print(b)
        return mu, b

    def getParams(self):
        return self.mu, self.b

    def pdf(self, value):
        pdfs = []
        for i in range(self.n_classes):
            numerator = np.exp(-np.abs(float(value - self.mu[i]) / self.b[i]) )
            denominator = 2 * self.b[i]
            pdfs.append(numerator/denominator)
        return pdfs

class Exponential:
    def fit(self, x, y):
        classes = np.unique(y)
        n_classes = classes.shape[0]

        mu = np.zeros(n_classes)

        for i in range(n_classes):
            outcome = classes[i]
            indices = np.argwhere(y == outcome).flatten()
            mu[i] = np.mean(x[indices])

        self.mu = mu
        self.classes = classes
        self.n_classes = n_classes
        return 1/np.array(mu)

    def getParams(self):
        return 1/self.mu

    def pdf(self, value):
        pdfs = []
        for i in range(self.n_classes):
            if value >= 0:
                one_by_mu = 1/self.mu[i]
                numerator = np.exp(-(value * one_by_mu))
                pdfs.append(numerator*one_by_mu)
            else:
                pdfs.append(0.0)
        return pdfs

class Multinomial:
    def fit(self, x, y):
        classes = np.unique(y)
        n_classes = classes.shape[0]

        n_yi = np.zeros((n_classes, 2))
        n_y = np.zeros((n_classes))
        p_k = [None] * n_classes

        self.uniques = np.unique(x)
        for i in range(n_classes):
            outcome = classes[i]
            indices = np.argwhere(y == outcome).flatten()
            pk = []
            n_sum = np.sum(x[indices])
            for x_val in self.uniques:
                pk.append(sum(x[indices] == x_val)/n_sum)
            p_k[i] = pk

        self.pk = p_k
        self.classes = classes
        self.n_classes = n_classes
        return p_k

    def getParams(self):
        return self.pk

    def pdf(self, value):
        pdfs = []
        for i in range(self.n_classes):
                idx = list(self.uniques).index(value)
                numerator = (self.pk[i])[idx]
                pdfs.append(numerator)
        return pdfs

