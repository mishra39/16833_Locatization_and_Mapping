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
        self._subsampling = 5

        self._norm_wts = 1.0
        self._map = occupancy_map
        self._offest = 25 # offset for the laser
        self._visualize_rays = True
    def calcProb(self,z_star_k, z_t1_arr):
        if ((z_t1_arr >= 0) and (z_t1_arr <= self._max_range)):
                normal_dist = 1 / (norm(z_star_k,self._sigma_hit**2).cdf(self._max_range) - norm(z_star_k,self._sigma_hit**2).cdf(0))
                p_hit = normal_dist * norm(z_star_k,self._sigma_hit**2).pdf(z_t1_arr)
        else:
            p_hit = 0
        
        if ((z_t1_arr >= 0) and (z_t1_arr <= z_star_k)):
            nu = 1 / (1 - math.exp(-self._lambda_short*z_star_k))
            p_short = nu * self._lambda_short * math.exp(-self._lambda_short*z_t1_arr)
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

    def rayCasting(self, x_t1):
        [x_rob,y_rob,theta_rob] = x_t1

        # compute laser pose
        x_l = x_rob + math.cos(theta_rob)*self._offest
        y_l = y_rob + math.sin(theta_rob)*self._offest

        # current location of the ray
        x_new = x_rob + math.cos(theta_rob)*self._offest
        y_new = y_rob + math.sin(theta_rob)*self._offest
        
        # for visualization
        z_pred_arr = [] # list of all the extended rays
        x_all = []
        y_all = []

        for theta_l in range(-90,90,self._subsampling):
            x_new = x_rob +  math.cos(theta_rob)
            y_new = y_rob + math.sin(theta_rob)
            theta_new = theta_rob + theta_l/180*math.pi
            map_x = int(x_new/10)
            map_y = int(y_new/10)
            
            while (max(x_new,y_new) < 8000 and min(x_new,y_new) >=0 and self._map[map_x,map_y]):
                x_new += 10*math.cos(theta_new)
                y_new += 10*math.sin(theta_new) 
                map_x = int(x_new/10)
                map_y = int(y_new/10)
            z_pred = math.sqrt((x_new - x_l)**2 + (y_new - y_l)**2)
            z_pred_arr.append(z_pred)
            x_all.append(map_x)
            y_all.append(map_y)
        return z_pred_arr

    def visualize_rays(self,x_t1, x_ray):
        plt.imshow(self._map,cmap='Greys')
        x_locs = x_t1[0] / 10.0
        y_locs = x_t1[1] / 10.0
        x_rloc = x_ray[0]/10
        y_rloc = x_ray[1]/10
        pose_rob = plt.plot(x_locs, y_locs, c='r', marker='.')
        pose_ray = plt.plot(x_rloc, y_rloc, c='g', marker='.')
        ray_arr = plt.arrow(x_locs,y_locs,(x_ray[0]/10)-x_locs,(x_ray[1]/10)-y_locs,length_includes_head=True,head_width=20,head_length=10)
        #ray_arr = plt.plot([x_locs,x_ray[0]],[y_locs,x_ray[1]])
        plt.pause(0.001)
        temp = pose_rob.pop(0)
        temp.remove()
        temp2 = pose_ray.pop(0)
        temp2.remove()
        ray_arr.remove()


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
        x_new_arr = []
        y_new_arr = []

        for k in range(0,k_tot): # k ranges from 0 to 180
            
            # compute z_star_k (true measurement) using ray casting
            theta_l =  int(theta_rob + math.radians(k) - (math.pi/2)) # this is in world frame the direction of the ray
            x_new = x_l 
            y_new = y_l
            theta_new = theta_rob + math.radians(k)
            map_x = int(x_new/10)
            map_y = int(y_new/10)
            
            while (max(x_new,y_new) < 8000 and min(x_new,y_new) >=0 and self._map[map_x,map_y] <  self._min_probability): # if the coordinates are within map and unoccupied, then extend the ray
                #print(self._min_probability)
                #print(map_x, map_y)
                x_new += 25*math.cos(theta_l)
                y_new += 25*math.sin(theta_l) 
                map_x = int(x_new/10)
                map_y = int(y_new/10)
            z_star_k = math.sqrt((x_new - x_l)**2 + (y_new - y_l)**2)
            p = self.calcProb(z_star_k, z_t1_arr[k])
            prob_zt1 += math.log(p)

            z_pred_arr.append(z_star_k)
            x_new_arr.append(map_x)
            y_new_arr.append(map_y)
            print(k,'append')

            self.visualize_rays(x_t1,[x_new,y_new])
            #print(math.log(p))
        #print(prob_zt1)
        prob_zt1 = math.exp(prob_zt1)
        
        return prob_zt1
