import numpy as np
import matplotlib.pyplot as plt


class LinearRegression(object):
    def __init__(self, thetasVector, featuresMatrix, originalAnswers):
        self.thetasVector = thetasVector
        self.featuresMatrix = featuresMatrix
        self.originalAnswers = originalAnswers

    def hypothesis(self, featuresVector):
        """
        calculates the predicted answer. Both arguments have to be a vector
        :param featuresVector: a vector with feature values
        :return: returns predicted answer
        """

        result = np.dot(self.thetasVector, featuresVector)
        return result

    def J(self, thetaVector, featureMatrix, originalAnswers):
        """
        calculates the cost function J
        :param thetaVector: a vector with theta values in it
        :param featureMatrix:  a matrix that has features in it
        :param originalAnswers:  original answers of sample data
        :return: returns the cost as ndarray
        """
        m = featureMatrix.shape[0]
        sumOfSerie = 0
        for i in range(m):
            sumOfSerie += ((self.hypothesis(thetaVector, featureMatrix[i, :]) - originalAnswers[i]) ** 2)[0]

        return (1 / 2 * m) * sumOfSerie

    def gradientDescent(self, alpha):
        m, n = self.featuresMatrix.shape
        nThetas = self.thetasVector.size

        sumOfSerie = 0
        for sub in range(nThetas):  # nThetas = number of thetas
            for super in range(m):  # for m data samples
                sumOfSerie += \
                ((self.hypothesis(featuresVector=self.featuresMatrix[super, :]) - self.originalAnswers[super]) * self.featuresMatrix[
                    super, sub])[0]

            self.thetasVector[sub] = self.thetasVector[sub] - alpha * sumOfSerie
            sumOfSerie = 0
