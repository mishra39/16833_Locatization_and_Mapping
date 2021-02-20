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

        for i in range(0,z_t1_arr.shape[0]):
            t = sqrt(x_t1[0]^2 + x_t1[1]^2)
            ur = [math.cos(x_t1[2]),math.sin(x_t1[2])]
            pr = x_t1+[25,25,0]
            zktp = pr + t*ur
            n = 1/(1-math.exp(-self._lambda_short*zktp))
            if z_t1_arr[i] in range(0,self._z_max):
                N = 1/(sqrt(2*math.pi*sig_hit^2))*math.exp(-0.5*(z_t1_arr-zktp)^2/sig_hit^2)
                p_hit = n*N
            else:
                p_hit = 0
            
            if z_t1_arr[i] in range(0,zktp):
                p_short = n*self._lambda_short*math.exp(-self._lambda_short*zktp)
            else:
                p_short = 0
            
            if z_t1_arr[i] == self._z_max:
                p_max = 1
            else:
                p_max = 0

            if z_t1_arr[i] in range(0,z_max):
                p_rand = 1/z_max
            else:
                p_rand = 0
            q = 1
            p = self._z_hit*p_hit + self._z_short*p_short + self._z_max*p_max + self._z_rand*p_rand
            q = q*p
        return q

        """
        prob_zt1 = 1.0
        return prob_zt1
        """
