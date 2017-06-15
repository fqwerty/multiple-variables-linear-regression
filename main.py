import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

#def drawCostFunction(featureMatrix, originalAnswers):
#    costsVector = np.empty((100,))
#    for i in range(100):
#        for j in range(100):
#            costsVector[i] = J(np.array([[i,j]]), featureMatrix, originalAnswers)
#
#
#    fig = plt.figure()
#    ax = fig.add_subplot()
#    u = range(100)
#    x,y= np.meshgrid(u,u)

#    plt.contour(x,y,costsVector)
#    plt.show()



def main():
    x,y = np.loadtxt("mydata1", delimiter=',', unpack=True)
    m = x.size
    x = x[np.newaxis].T
    y = y[np.newaxis].T
    # to treat x0 as a feature, we add it to the x matrix, since x0=1 by default, we create a vector
    # filled with ones and add it to the x matrix
    x0FeatureColumn = np.ones((m, 1))
    x = np.c_[x0FeatureColumn, x]  # matlab like solution to column inserting

    # theta is a vector that is initialized with zeros. (theta0 = 0 and theta1= 0)
    # later we will find the optimal theta values for our cost function J.
    theta = np.zeros((2,))

    iterations = 1500  # number of iterations
    alpha = 0.0001 # learning rate

    print("Shae of theta: " + str(theta.shape))

    lr = LinearRegression(theta,x,y)

    for i in range(iterations):
        print("Current cost is " + str(lr.J()))
        lr.gradientDescent(alpha)

    hypothesisSolutions = np.empty((m,1), dtype='float64')
    for i in range(m):
        hypothesisSolutions[i,0]= lr.hypothesis(featuresVector=x[i,:])

    print("Theta 0 is: " + str(theta[0]))
    print("Theta 1 is: " + str(theta[1]))
    lr.drawCostFunction()

#    plt.scatter(x, y, label="test scatter plot", marker="x")
#    plt.xlabel('x')
#    plt.ylabel('y')
#    #plt.title('Interesting Graph\n Check it out!')
#    #plt.legend()
#    plt.show()
if __name__ == "__main__":
    main()




