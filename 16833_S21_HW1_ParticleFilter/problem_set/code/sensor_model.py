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
        self._z_hit = 0.75
        self._z_short = 0.05
        self._z_max = 0.025
        self._z_rand = 0.8995

        self._sigma_hit = 50
        self._lambda_short = 0.1

        self._max_range = 1000
        self._min_probability = 0.35
        self._subsampling = 5

        self._norm_wts = 1.0
        self._map = occupancy_map
        self._offest = 25 # offset for the laser
        self._visualize_rays = True

    def calcProb(self,z_star_k, z_t1_arr):

        if ((z_t1_arr >= 0) and (z_t1_arr <= self._max_range)):
                cum_dist = norm.cdf(z_t1_arr, loc=z_star_k, scale=self._sigma_hit)
                if cum_dist < 0.0001:
                    p_hit = 0
                
                normalizer = 1 / cum_dist
                p_hit = normalizer * norm.pdf(z_t1_arr, loc=z_star_k, scale=self._sigma_hit)
        else:
            p_hit = 0
        
        if ((z_t1_arr >= 0) and (z_t1_arr <= z_star_k)):
            nu = 1.0/(1-math.exp(-self._lambda_short*z_star_k))
            p_short =  nu*self._lambda_short*math.exp(-self._lambda_short*z_t1_arr)
        else:
            p_short = 0
        
        if (z_t1_arr >= self._max_range):
            p_max = 1
        else:
            p_max = 0
        
        if ((z_t1_arr >= 0) and (z_t1_arr < self._max_range)):
            p_rand = 1 / self._max_range
        else:
            p_rand = 0
        
        p = self._z_hit*p_hit + self._z_short*p_short + self._z_max*p_max + self._z_rand*p_rand
        
        return p

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
        plt.pause(0.01)
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

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        prob_zt1 = 0
        [x_rob,y_rob,theta_rob] = x_t1
        k_tot = len(z_t1_arr)

        # compute laser pose in world frame
        x_l = x_rob + math.cos(theta_rob)*self._offest
        y_l = y_rob + math.sin(theta_rob)*self._offest

        z_pred_arr = []
        x_map_arr = []
        y_map_arr = []
        x_meas_arr = []
        y_meas_arr = []

        for k in range(0,k_tot, self._subsampling): # k ranges from 0 to 180
            
            # compute z_star_k (true measurement) using ray casting
            theta_l =  theta_rob + math.radians(k) - (math.pi/2) # this is in world frame the direction of the ray
            x_new = x_l 
            y_new = y_l
            map_x = int(x_new/10)
            map_y = int(y_new/10)

            #print("Location at start of ray casting: " + str(map_x) + ", " + str(map_y))
            while (max(x_new,y_new) < 8000 and min(x_new,y_new) >=0 and self._map[map_y,map_x] <  self._min_probability): # if the coordinates are within map and unoccupied, then extend the ray
                #print(map_x, map_y)
                x_new += 15*math.cos(theta_l)
                y_new += 15*math.sin(theta_l) 
                map_x = int(x_new/10)
                map_y = int(y_new/10)

                #print("Ray extended to: " + str(map_x) + ", " + str(map_y))
            
            #print("Location at end of ray casting: " + str(map_x) + ", " + str(map_y))
            z_star_k = math.sqrt((x_new - x_l)**2 + (y_new - y_l)**2)
            p = self.calcProb(z_star_k, z_t1_arr[k])
            #print("z* calc: " + str(z_star_k))
            #print("z laser: " + str(z_t1_arr[k]))
            
            prob_zt1 += math.log(p)
            z_pred_arr.append(z_star_k)
            x_map_arr.append(map_x)
            y_map_arr.append(map_y)
            
            # Location of the original measurement in the world frame
            x_meas = int((x_l + z_t1_arr[k]*math.cos(theta_l)) /10)
            y_meas = int((y_l + z_t1_arr[k]*math.sin(theta_l))/10)
            x_meas_arr.append(x_meas)
            y_meas_arr.append(y_meas)

            #self.visualize_rays(x_t1,[map_x,map_y], [x_meas,y_meas])

        #self.visualize_allRays(x_t1,x_map_arr,y_map_arr, x_meas_arr,y_meas_arr)
        prob_zt1 = math.exp(prob_zt1)

        print(prob_zt1)
        return prob_zt1