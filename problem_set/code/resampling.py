'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import matplotlib.pyplot as plt

class Resampling:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """
    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """

    def multinomial_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        X_bar_resampled =  np.zeros_like(X_bar)



        return X_bar_resampled

    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        #hi this is initial commit
        X_bar_resampled =  np.zeros_like(X_bar)
        M = len(X_bar[:,0])
        r = np.random.randint(0,M-1)
        c = X_bar[0,3]
        i = 1
        
        for m in range(1,M):
            U = r+ (m-1)*(M-1)
            for i in range(1,M-1):
                if U>c:
                    c = c+X_bar[i,3]
            X_bar_resampled[m,:] = X_bar[i,:] 
        X_bar_resampled[0,:] = X_bar_resampled[1,:]
        return X_bar_resampled

def unifromtest():
    #Resamples points 10 times
    pose = np.random.normal(180, 20, (100,3))
    weights = np.random.normal(50, 10,(100,1))
    # weights = weights/sum(weights)
    x_bar = np.append(pose,weights,1)
    sample = Resampling()
    xs1 = x_bar[:,0]
    ys1 = x_bar[:,1]
    xs = x_bar[:,0]
    ys = x_bar[:,1]
    while not np.all(xs[1:] == xs[1]):
        x_bar = sample.low_variance_sampler(x_bar)
        xs = x_bar[1:,0]
        ys = x_bar[1:,1]
        plt.scatter(xs1,ys1,c="black")
        plt.scatter(xs,ys,c="red")
        # plt.show(block=False)
        # plt.pause(1)
        # plt.close()
    print(x_bar[2,:-1])
    plt.scatter(xs1,ys1,c="black")
    plt.scatter(xs[1:],ys[1:],c="red")
    plt.show()

def normtest():
    #Resamples points 10 times

    pose = np.random.normal(180, 20, (100,3))
    weights = np.random.normal(50, 10,(100,1))
    # weights = weights/sum(weights)
    x_bar = np.append(pose,weights,1)
    sample = Resampling()
    xs1 = x_bar[:,0]
    ys1 = x_bar[:,1]
    xs = x_bar[:,0]
    ys = x_bar[:,1]
    while not np.all(xs[1:] == xs[1]):
        x_bar = sample.low_variance_sampler(x_bar)
        xs = x_bar[1:,0]
        ys = x_bar[1:,1]
        plt.scatter(xs1,ys1,c="black")
        plt.scatter(xs,ys,c="red")
        # plt.show(block=False)
        # plt.pause(1)
        # plt.close()
    print(x_bar[2,:-1])
    plt.scatter(xs1,ys1,c="black")
    plt.scatter(xs[1:],ys[1:],c="red")
    plt.show()


# for _ in range(10):
#     normtest()  
