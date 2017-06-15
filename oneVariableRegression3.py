import numpy as np
import matplotlib.pyplot as plt


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





def hypothesis(thetasVector, featuresVector):
    """
    calculates the predicted answer. Both arguments have to be a vector
    :param featuresMatrix: a vector with feature values
    :param thetasVector: a vector with theta values
    :return: returns predicted answer
    """

    result = np.dot(thetasVector,featuresVector)
    return result

def J(thetaVector, featureMatrix, originalAnswers):
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
        sumOfSerie += ((hypothesis(thetaVector, featureMatrix[i,:])-originalAnswers[i])**2)[0]

    return (1/2*m)*sumOfSerie

def gradientDescent(alpha,thetaVector, featureMatrix, originalAnswers):
    m,n = featureMatrix.shape
    nThetas = thetaVector.size

    sumOfSerie = 0
    for sub in range(nThetas): #nThetas = number of thetas
        for super in range(m): # for m data samples
            sumOfSerie += ((hypothesis(thetaVector, featureMatrix[super, :]) - originalAnswers[super]) * featureMatrix[
                super, sub])[0]

        thetaVector[sub] = thetaVector[sub] -alpha*sumOfSerie
        sumOfSerie = 0


def main():
    x,y = np.loadtxt("mydata1", delimiter=',', unpack=True)
    m = x.size
    x = x[np.newaxis].T
    y = y[np.newaxis].T
    theta = np.array([0,0],dtype='float64')


    # to treat x0 as a feature, we add it to the x matrix, since x0=1 by default, we create a vector
    # filled with ones and add it to the x matrix
    x0FeatureColumn = np.ones((m, 1))
    x = np.c_[x0FeatureColumn, x]  # matlab like solution to column inserting

    # theta is a vector that is initialized with zeros. (theta0 = 0 and theta1= 0)
    # later we will find the optimal theta values for our cost function J.
    theta = np.zeros((2,))

    iterations = 1500  # number of iterations
    alpha = 0.0001 # learning rate

    for i in range(iterations):
        #print("Current cost is: " + str(J(thetaVector=theta, featureMatrix=x, originalAnswers=y)))
        gradientDescent(thetaVector=theta, alpha=alpha, featureMatrix=x, originalAnswers=y)


    hypothesisSolutions = np.empty((m,1), dtype='float64')
    for i in range(m):
        hypothesisSolutions[i,0]= hypothesis(theta,x[i,:])

    print("Theta 0 is: " + str(theta[0]))
    print("Theta 1 is: " + str(theta[1]))


#    plt.scatter(x, y, label="test scatter plot", marker="x")
#    plt.xlabel('x')
#    plt.ylabel('y')
#    #plt.title('Interesting Graph\n Check it out!')
#    #plt.legend()
#    plt.show()












if __name__ == "__main__":
    main()