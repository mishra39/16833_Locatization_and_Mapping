'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._z_hit = 1
        self._z_short = 0.1
        self._z_max = 0.1
        self._z_rand = 100

        self._sigma_hit = 50
        self._lambda_short = 0.1

        self._max_range = 1000
        self._min_probability = 0.35
        self._subsampling = 2

        self._norm_wts = 1.0 #ignore

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        self._z_hit = z_hit
        self._z_short = z_short
        self._z_max = z_max
        self._z_rand = z_rand
        self._sigma_hit = sig_hit
        self._lambda_short = lambda_short
        self._max_range = max_range
        self._min_probability = min_probability

        zktp = z_t1_arr + x_t1*time
        n = 1/(1-math.exp(-lambda_short*zktp))
        #p_hit
        N = 1/(sqrt(2*math.pi*sig_hit^2))*math.exp(-0.5*(z_t1_arr-zktp)^2/sig_hit^2)
        if z_t1_arr in range(0,z_max):
            p_hit = n*N
            #eq 6.4 and 6.5 the zkt when p_hit is max, is zkt* ->2 dimensional ray cast
        else:
            p_hit = o

        #p_short
        if z_t1_arr in range(0,zktp):
            p_short = n*lambda_short*math.exp(-lambda_short*z_t1_arr) #eq 6.7
        else:
            p_short = 0
        
        #p_max
        if z_t1_arr = z_max:
            p_max = 1
        else:
            p_max = 0

        #p_rand
        if z_t1_arr in range(0,z_max):
            p_rand = 1/z_max
        else:
            p_rand = 0

        q = 1
        for k=1 to K:
            p = z_hit*p_hit + z_short*p_short + z_max*p_max + z_rand*p_rand
            q = q*p
        return q

        """
        prob_zt1 = 1.0
        return prob_zt1
        """
