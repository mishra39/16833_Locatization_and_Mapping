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
        X_bar_resampled =  []
        wt_arr = X_bar[:,3]

        # Normalized weights
        wt_arr = wt_arr / np.sum(wt_arr)
        M = len(X_bar[:,0])
        
        r = np.random.uniform(0,1/M)
        c = wt_arr[0]
        i = 0
        
        for m in range(0,M):
            U = r + (m/M)
            while U>c:
                i = i+1
                c += wt_arr[i]
            X_bar_resampled.append(X_bar[i])
        X_bar_resampled = np.asarray(X_bar_resampled)
        return X_bar_resampled

def test():
    #Resamples points 10 times
    X_bar = np.array([[94.234001,250.953995,-1.342158,0.025863],[294.234001,139.953995,-1.342158,0.0079745],
    	[194.234001,439.953995,1.342158,0.0013982],[594.234001,339.953995,1.342158,0.200218]])
    print (X_bar)
    print(X_bar[0,3])
    xs1 = X_bar[:,0]
    ys1 = X_bar[:,1]
    for i in range(100):
        plt.scatter(xs1,ys1,c="red")    
        sampleObj = Resampling()
        X_bar_resampled = sampleObj.low_variance_sampler(X_bar)
        print (X_bar_resampled)
        print("Iteration: " + str(i))
        xs1 = X_bar[:,0]
        ys1 = X_bar[:,1]
        #print (xs1,ys1)
        xs1 = X_bar_resampled[:,0]
        ys1 = X_bar_resampled[:,1]
        plt.axis([0, 800, 0, 800])

    #plt.show()
if __name__ == '__main__':
    test()