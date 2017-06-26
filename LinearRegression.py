from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


class LinearRegression(object):
    def __init__(self, theta, x, y):
        self.theta = theta
        self.x = x
        self.y = y
        self.n = theta.size  # number of features
        self.m = y.size  # number of training examples

    def hypothesis(self, featuresVector, currentTheta = None):
        """
        calculates the predicted answer. Both arguments have to be a vector
        :param featuresVector: a vector with feature values
        :return: returns predicted answer
        """
        if currentTheta is not None:
            return np.dot(currentTheta, featuresVector)
        else:
            return np.dot(self.theta, featuresVector)

    def J(self):
        """
        calculates the cost function J
        :param localTheta: this will be entered to hypothesis method as an argument
        :return: returns the cost as ndarray
        """
        m = self.x.shape[0]
        sumOfSerie = 0
        for i in range(m):
            sumOfSerie += ((self.hypothesis(self.x[i, :]) - self.y[i]) ** 2)

        return (1 / (2 * m)) * sumOfSerie

    def drawCostFunction(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        theta0 = np.linspace(0, 2, num=10)
        theta1 = np.linspace(0, 2, num=10)
        print(theta1.size)
        z = np.empty(theta1.size)
        for i in range(z.size):
            localTheta = np.array([theta1[i], theta0[i]])
            print("LocalTheta shape: " + str(localTheta.shape))
            print("Object's theta shape: " + str(self.theta.shape))
            z[i] = self.J()

        theta0, theta1 = np.meshgrid(theta0, theta1)

        # plot the surface.
        surf = ax.plot_surface(theta0, theta1, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        # Customize the z axis.
        #        ax.zaxis.set_major_locator(LinearLocator(10))
        #        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        #        Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

    def gradientDescent(self, alpha, iterations=1500):
        numberOfThetas = self.n
        previousCost = 0
        sumOfSerie = 0
        for i in range(iterations):
            currentTheta = self.theta.copy()
            for sub in range(numberOfThetas):
                for _super in range(self.m):
                    sumOfSerie += (self.hypothesis(self.x[_super, :], currentTheta = currentTheta) - self.y[_super]) * self.x[_super, sub]
                self.theta[sub] -= (alpha / self.m) * sumOfSerie
                sumOfSerie = 0
            currentCost = self.J()
            print("Current cost: " + str(currentCost))
            difference = previousCost - currentCost
            #print("Difference: " + str(previousCost - currentCost))
            if abs(difference) < 0.000001:
                break
            previousCost = currentCost
