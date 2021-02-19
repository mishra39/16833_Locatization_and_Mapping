'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as np
import math


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._alpha1 = 0.05
        self._alpha2 = 0.05
        self._alpha3 = 0.06
        self._alpha4 = 0.04


    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        """
        TODO : Add your code here
        """
        x_est_t0, y_est_t0, theta_est_t0 = u_t0
        x_est_t1, y_est_t1, theta_est_t1 = u_t1
        x_prev, y_prev, theta_prev = x_t0

        del_rot1 = math.atan2(y_est_t1 - y_est_t0, x_est_t1 - x_est_t0) - theta_est_t0
        del_trans = math.sqrt((x_est_t0 - x_est_t1)**2 + (y_est_t0 - y_est_t1)**2)
        del_rot2 = theta_est_t1 - theta_est_t0 - del_rot1
        
        # Sampling values
        std_rot1 = math.sqrt(self._alpha1*del_rot1**2 + self._alpha2*del_trans**2)
        std_trans = math.sqrt(self._alpha3*del_trans**2 + self._alpha4*del_rot1**2 + self._alpha4*del_rot2**2)
        std_rot2 = math.sqrt(self._alpha1*del_rot2**2 + self._alpha2*del_trans**2)
        
        del_rot1_hat = del_rot1 - np.random.normal(loc=0.0,scale=std_rot1, size=None)
        del_trans_hat = del_trans - np.random.normal(loc=0.0, scale=std_trans,size=None)
        del_rot2_hat = del_rot2 - np.random.normal(loc=0.0,scale=std_rot2,size=None)
        
        x_t1 = x_prev + del_trans_hat * math.cos(theta_prev + del_rot1_hat)
        y_t1 = y_prev + del_trans_hat * math.sin(theta_prev + del_rot1_hat)
        theta_t1 = theta_prev + del_rot1_hat + del_rot2_hat
        
        return x_t1, y_t1, theta_t1
        #return np.random.rand(3)