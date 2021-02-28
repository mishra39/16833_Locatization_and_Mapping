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
        self._z_hit = 75
        self._z_short = 0.50
        self._z_max = 0.25
        self._z_rand = 500

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

                else:
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

        downsample = 2
        x_map_arr = x_map_arr[0:150:downsample]
        y_map_arr = y_map_arr[0:150:downsample]
        x_meas_arr = x_meas_arr[0:150:downsample]
        y_meas_arr = y_meas_arr[0:150:downsample]

        x_locs = x_t1[0] / 10.0
        y_locs = x_t1[1] / 10.0

        plt.imshow(self._map, cmap='Greys')
        pose_rob = plt.scatter(x_locs, y_locs, c='r', marker='o')

        #ray_arr = plt.arrow(x_locs,y_locs,x_ray[0]-x_locs,x_ray[1]-y_locs,length_includes_head=True,head_width=20,head_length=10) # Arrow from particle to predicted point
        pred_pt = plt.scatter(x_map_arr, y_map_arr, c='b', marker='o') # Location of true point
        meas_l = plt.scatter(x_meas_arr, y_meas_arr, c='y', marker='o') # location of the measurement of from laser
        plt.pause(0.01)
        pose_rob.remove()
        #ray_arr.remove()
        meas_l.remove()
        pred_pt.remove()

    def rayCasting(self, x_l, y_l, theta_l):
        """ 
        param[in] x_l,y_l: laser position in world frame (cm)
        param[in] theta_l: direction of ray from laser in world frame
        param[out] z_star_k : true measurement from the map
        """
        rayDirX = math.cos(theta_l)
        rayDirY = math.sin(theta_l)

        deltaDistX = float("inf")
        deltaDistY = float("inf")

        if rayDirY==0:
            deltaDistX=0
        else:
            if rayDirX==0:
                deltaDistX=1
            else:
                deltaDistX = abs(1/rayDirX)
        
        if rayDirX ==0:
            deltaDistX = 0
        else:
            if rayDirY==0:
                deltaDistY = 1
            else:
                deltaDistY = abs(1/rayDirY)

        # current location in the map in decimeters
        posX = x_l/10
        posY = y_l/10

        # which box of the map we're in
        mapX = int(x_l/10)
        mapY = int(y_l/10)

        # length of ray from current position to next x or y-side
        sideDistX = 0
        sideDistY = 0

        # what direction to step in x or y-direction (either +1 or -1)
        stepX = 1
        stepY = 1

        side = 0 #  If an x-side was hit, side is set to 0, if an y-side was hit, side will be 1

        if rayDirX < 0:
            stepX = -1
            sideDistX = (posX - mapX)*deltaDistX

        else:
            stepX = 1
            sideDistX = (mapX + 1.0 - posX)*deltaDistX
        
        if rayDirY < 0:
            stepY = -1
            sideDistY = (posY - mapY) * deltaDistY

        else:
            stepY = 1
            sideDistY = (mapY + 1.0 - posY)*deltaDistY
        
        #print(mapX,mapY)
        while (max(mapX,mapY) < 800 and min(mapX,mapY) >=0 and self._map[mapY,mapX] < self._min_probability):
            if sideDistX < sideDistY:
                sideDistX += deltaDistX
                mapX += stepX
                side = 0
            else:
                sideDistY += deltaDistY
                mapY += stepY
                side = 1
            #print(mapX,mapY)

        # Calculate distance projected on camera direction (Euclidean distance will give fisheye effect!)
        if (side == 0):
            perpWallDist = (mapX - posX + (1 - stepX) / 2) / rayDirX
        else:
            perpWallDist = (mapY - posY + (1 - stepY) / 2) / rayDirY;
        
        finalDist = perpWallDist*10 # convert distance into cm from decimeters

        return [mapX, mapY, finalDist]
    
    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        prob_zt1 = 0
        [x_rob,y_rob,theta_rob] = x_t1
        k_tot = len(z_t1_arr)
        x_t1_occ = False # flag if the given position in a wall

        if (self._map[int(y_rob/10)][int(x_rob/10)] >= self._min_probability):
            x_t1_occ = True

        # compute laser pose in world frame
        x_l = x_rob + math.cos(theta_rob)*self._offest
        y_l = y_rob + math.sin(theta_rob)*self._offest

        mapX_arr = []
        mapY_arr = []
        
        x_meas_arr = []
        y_meas_arr = []

        zStarK_arr = []
        for k in range(0,k_tot, self._subsampling): # k ranges from 0 to 180
            
            # compute z_star_k (true measurement) using ray casting
            theta_l =  theta_rob + math.radians(k) - (math.pi/2) # the direction of the ray in world frame 
            mapX, mapY, z_star_k = self.rayCasting(x_l, y_l,theta_l)

            p = self.calcProb(z_star_k, z_t1_arr[k])
            #print("z* calc: " + str(z_star_k))
            #print("z laser: " + str(z_t1_arr[k]))
            if p!=0:
                prob_zt1 += math.log(p)
            else:
                return -float("inf")

            mapX_arr.append(mapX)
            mapY_arr.append(mapY)

            # Location of the original measurement in the world frame
            x_meas = int((x_l + z_t1_arr[k]*math.cos(theta_l)) /10)
            y_meas = int((y_l + z_t1_arr[k]*math.sin(theta_l))/10)
            x_meas_arr.append(x_meas)
            y_meas_arr.append(y_meas)

        #self.visualize_allRays(x_t1,mapX_arr,mapY_arr, x_meas_arr,y_meas_arr)
        #prob_zt1 = math.exp(prob_zt1)
        #prob_zt1 += 50
        print(prob_zt1)
        return prob_zt1

if __name__ == '__main__':
    map = [\
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],\
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],\
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],\
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],\
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]
    
    map = (np.asarray(map)).reshape(5,-1)
    sensor_model = SensorModel(map)
    mapX, mapY, z_star_k = sensor_model.rayCasting(20, 30, -1.57/2)
    print(mapX, mapY)
    print(map[mapY,mapX])