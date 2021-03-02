'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm, expon
from scipy import signal
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
        self._z_hit = 1000
        self._z_short = 0.01
        self._z_max = 0.03
        self._z_rand = 100000

        self._sigma_hit = 250
        self._lambda_short = 0.01

        self._max_range = 8183
        self._min_probability = 0.25
        self._subsampling = 10

        self._norm_wts = 1.0
        self._map = occupancy_map
        self._offest = 25 # offset for the laser
        self._visualize_rays = True

    def visualize_rays(self,x_t1, x_ray, x_meas):
        """ 
        param[in] x_t1: particle state belief [x, y, theta] at time t [world_frame]
        param[in] x_ray: [x, y] predicted end point of the ray from laser to map [world_frame / 10]
        param[in] x_meas: [x, y] measured position of the obstacle [world_frame / 10]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        
        x_locs = x_t1[0] / 10.0
        y_locs = x_t1[1] / 10.0
        plt.imshow(self._map, cmap='Greys')
        pose_rob = plt.scatter(x_locs, y_locs, c='r', marker='o')
        ray_arr = plt.arrow(x_locs,y_locs,x_ray[0]-x_locs,x_ray[1]-y_locs,length_includes_head=True,head_width=20,head_length=10) # Arrow from particle to predicted point
        pred_pt = plt.scatter(x_ray[0], x_ray[1], c='b', marker='o') # Location of predicted point
        meas_l = plt.scatter(x_meas[0], x_meas[1], c='y', marker='o') # location of the measurement of from laser
        plt.pause(0.005)
        pose_rob.remove()
        ray_arr.remove()
        meas_l.remove()
        pred_pt.remove()

    def visualize_allRays(self,x_t1,x_map_arr,y_map_arr, x_meas_arr,y_meas_arr):
        """ 
        param[in] x_t1: particle state belief [x, y, theta] at time t [world_frame]
        param[in] x_ray: [x, y] predicted end point of the ray from laser to map [world_frame / 10]
        param[in] x_meas: [x, y] measured position of the obstacle [world_frame / 10]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        x_map_arr = np.asarray(x_map_arr)
        y_map_arr = np.asarray(y_map_arr)
        x_meas_arr = np.asarray(x_meas_arr)
        y_meas_arr = np.asarray(y_meas_arr)

        x_map_arr = x_map_arr[0:150:6]
        y_map_arr = y_map_arr[0:150:6]
        x_meas_arr = x_meas_arr[0:150:6]
        y_meas_arr = y_meas_arr[0:150:6]

        x_locs = x_t1[0] / 10.0
        y_locs = x_t1[1] / 10.0

        plt.imshow(self._map, cmap='Greys')
        pose_rob = plt.scatter(x_locs, y_locs, c='r', marker='o')

        #ray_arr = plt.arrow(x_locs,y_locs,x_ray[0]-x_locs,x_ray[1]-y_locs,length_includes_head=True,head_width=20,head_length=10) # Arrow from particle to predicted point
        pred_pt = plt.scatter(x_map_arr, y_map_arr, c='b', marker='o') # Location of predicted point
        meas_l = plt.scatter(x_meas_arr, y_meas_arr, c='y', marker='o') # location of the measurement of from laser
        plt.pause(0.01)
        pose_rob.remove()
        #ray_arr.remove()
        meas_l.remove()
        pred_pt.remove()

    def calcProb(self,z_t_k, z_star_k):
        if (z_t_k >=0 and z_t_k <= self._max_range):
            prob_hit = (math.exp(-(z_t_k - z_star_k)**2 / (2 * self._sigma_hit**2)))/ math.sqrt(2 * math.pi * self._sigma_hit**2)
            
        else:
            prob_hit = 0
        
        if (z_t_k >=0 and z_t_k <= z_star_k):
            exp_dist = 1 / (1 - math.exp(-self._lambda_short * z_star_k))
            prob_short = self._lambda_short * exp_dist * math.exp(-self._lambda_short * z_t_k)
        else:
            prob_short = 0

        if z_t_k == self._max_range:
            prob_max = 1.0
        else:
            prob_max = 0

        if (z_t_k >=0 and z_t_k < self._max_range):
            prob_rand = 1.0 / self._max_range
        else:
            prob_rand =  0

        p = self._z_hit*prob_hit + self._z_short*prob_short + self._z_max * prob_max + self._z_rand*prob_rand
        return p

    def rayCasting(self, k,xl_map,yl_map,theta_rob):
        ray_ang = theta_rob + math.radians(k)
        x_new_map = xl_map
        y_new_map = yl_map

        while (x_new_map>=0 and x_new_map < self._map.shape[1] and y_new_map>=0 and y_new_map < self._map.shape[0] and abs(self._map[y_new_map,x_new_map]) < 1e-6):
            x_new_map += 2*np.cos(ray_ang)
            y_new_map += 2*np.sin(ray_ang)
            x_new_map = int(round(x_new_map))
            y_new_map = int(round(y_new_map))
        
        z_true = np.sqrt((x_new_map-xl_map)**2 + (y_new_map-yl_map)**2)
        z_true = z_true*10
        return z_true

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        q = 0
        [x_rob,y_rob,theta_rob] = x_t1


        # compute laser pose in world frame
        x_l = math.cos(theta_rob)*self._offest
        y_l = math.sin(theta_rob)*self._offest

        xl_map = int(round((x_rob + x_l) / 10.0))
        yl_map = int(round((y_rob + y_l) / 10.0))

        for k in range(-90,90, self._subsampling): # k ranges from 0 to 180
            z_star_k = self.rayCasting(k,xl_map,yl_map,theta_rob)
            z_t_k = z_t1_arr[k+90]
            p = self.calcProb(z_t_k,z_star_k)# self._z_hit*self.p_hit(z_t_k,x_t1,z_star_k) + self._z_short*self.p_short(z_t_k,x_t1,z_star_k) + self._z_max * self.p_max(z_t_k,x_t1) + self._z_rand*self.p_rand(z_t_k,x_t1)
            if (p > 0):
                q += np.log(p)

        prob_zt1 = math.exp(q)
        #print("Probability Computed for " + str(x_rob) + str(y_rob) +": "+ str(prob_zt1))
        return prob_zt1