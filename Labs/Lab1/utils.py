import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def gaussians():
    '''
    Generates data from 3 different (in mean) 2D multivariate Gaussian distributions.
    Means, covariances and number of samples are fixed.
    '''
    '''
	INPUTS : /
	'''
    '''
	OUTPUTS:
	- 'points' : numpy array in 2D containing the x-axis and y-axis coordinates of N points
	- 'y' : numpy array in 1D containing the labels of each one of the N points in 'points'
	'''
    
    N=50
    means = np.array([[4.5, 4.5],
                      [5.5, 2.5],
                      [6.3,3.5]])
    covs = np.array([np.diag([0.5, 0.5]),
                     np.diag([0.5, 0.5]),
                     np.diag([0.5, 0.5])])
    y=[]
    points = []
    for i in range(len(means)):
        x = np.random.multivariate_normal(means[i], covs[i], N )
        points.append(x)
        y.append(i*np.ones(N)) 
    points = np.concatenate(points)
    y=np.concatenate(y)
    
    return points, y


def comparing_plots(xx,yy, X, y, data_1, data_2, title_1, title_2):
    '''
    utilitary function to plot results from two methods side by side. 
    It displays the training data with different colours and uses the same colours to differentiate 
    the different regions defined by the decision boundaries.
    '''
    '''
    INPUTS :
    - xx : x-axis coordinates of input testing points
    - yy : y-axis coordinates of input testing points
    - X : set of coordinates of the input training points
    - y : set of labels of the training points
    - data1 : set of predicted labels using 'title_1' (e.g. scikit-learn or your implementation or logistic regression or ...)
    - data2 : set of predicted labels using 'title_2'
    - title1 : character string specifying how 'data1' are obtained
    - title2 : character string specifying how 'data2' are obtained
	'''
    '''
	OUTPUTS: /
    '''
    
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    plt.rcParams['figure.figsize'] = [20, 10]
    plt.subplot(121)
    plt.pcolormesh(xx, yy, data_1, cmap=cmap_light)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title_1)
    
    plt.subplot(122)

    plt.pcolormesh(xx, yy, data_2, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title_2)
    plt.show()