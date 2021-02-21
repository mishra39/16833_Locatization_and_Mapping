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

        self._norm_wts = 1.0
        self._map = occupancy_map
        self._offest = 25 # offset for the laser
    
    def calcProb(self,z_star_k, z_t1_arr):
        if ((z_t1_arr >= 0) and (z_t1_arr <= self._max_range)):
                p_hit = np.random.normal(loc=z_star_k, scale=self._sigma_hit, size=None)
        else:
            p_hit = 0
        
        if ((z_t1_arr >= 0) and (z_t1_arr <= z_star_k)):
            nu = 1 / (1 - math.exp(-self._lambda_short*z_star_k))
            p_short = nu * self._lambda_short * math.exp(-self._lambda_short*z_t1_arr)
        else:
            p_short = 0
        
        if (z_t1_arr == self._max_range):
            p_max = 1
        else:
            p_max = 0
        
        if ((z_t1_arr >= 0) and (z_t1_arr < self._max_range)):
            p_rand = 1 / self._max_range
        else:
            p_rand = 0
        
        p = self._z_hit*p_hit + self._z_short*p_short + self._z_max*p_max + self._z_rand*p_rand
        
        return p

    def swap(self, x,y):
        temp = x
        x = y
        y = temp
        return x,y

    def rayCasting(self, x_t1, z_k):
        [x_rob,y_rob,theta_rob] = x_t1

        # compute laser pose
        x_l = x_rob + math.cos(theta_rob)*self._offest
        y_l = y_rob + math.sin(theta_rob)*self._offest

        # current location of the ray
        x_curr = x_rob + math.cos(theta_rob)*self._offest
        y_curr = y_rob + math.sin(theta_rob)*self._offest
        
        map_x = math.floor(x_curr/10)
        map_y = math.floor(y_curr/10)
        
        # extend in x first and then in y
        step_x = 1
        step_y = 0

        while (self._map[map_x][map_y] and max(x_curr,y_curr) <= 8000 and min(x_curr,y_curr) >=0):
            x_curr += step_x*math.cos(########)
            y_curr += step_y*math.sin()
            step_x, step_y = swap(step_x,step_y) # extend in the other direction next


    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        prob_zt1 = 1.0
        z_pred = rayCasting(x_t1)



        k_tot = z_t1_arr.shape[0]
        ray_dist = np.linspace(0,self._max_range,num=1000) # array of distance for vector
        new_pose  = np.asarray(x_t1)
        for k in range(k_tot):
            ang_rad = math.radians(k)
            # adjust angle for laser position
            # compute z_star_k (true measurement) using ray casting
            
            # find direction vector using theta
            ray_dirn = np.array([math.cos(ang_rad), math.sin(ang_rad)])
            
            # extend the ray
            z_star_k = float('inf')
            for t in ray_dist:
                new_pose[:2] += t*ray_dirn
                map_x = math.ceil(new_pose[0])
                map_y = math.ceil(new_pose[1])
                print(map_x,map_y)
                map_locn = self._map[[map_x][map_y]]
                # check if the location is empty on the map
                if (map_locn == 0):
                    continue

                elif (map_locn == 1):
                    z_star_k = t
                    break
                
                else:
                    # occupied with prob 0.5
                    z_star_k = t

            p = calcProb(z_star_k, z_[k])
            prob_zt1 = p * prob_zt1
            print('Laser Reading: ' + str(z_t1_arr[k])) 

        prob_zt1 = 1.0
        return prob_zt1
