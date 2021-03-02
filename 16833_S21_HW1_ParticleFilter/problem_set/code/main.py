'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import argparse
import numpy as np
import sys, os

import multiprocessing
from multiprocessing import Pool
from map_reader import MapReader
from motion_model import MotionModel
from sensor_model import SensorModel
from resampling import Resampling
import math
from matplotlib import pyplot as plt
from matplotlib import figure as fig
import time


def visualize_map(occupancy_map):
    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    plt.ion()
    plt.imshow(occupancy_map, cmap='Greys')
    plt.axis([0, 800, 0, 800])


def visualize_timestep(X_bar, tstep, output_path):
    x_locs = X_bar[:, 0] / 10.0
    y_locs = X_bar[:, 1] / 10.0
    x_dir = np.cos(X_bar[:,2])
    y_dir = np.sin(X_bar[:,2])
    dirn = plt.quiver(x_locs,y_locs,x_dir,y_dir,scale=10)
    scat = plt.scatter(x_locs, y_locs, c='r', marker='o')
    #plt.savefig('{}/{:04d}.png'.format(output_path, tstep))
    plt.pause(0.00001)
    scat.remove()
    dirn.remove()


def init_particles_random(num_particles, occupancy_map):
    
    # initialize [x, y, theta] positions in world_frame for all particles
    y0_vals = np.random.uniform(0, 7000, (num_particles, 1))
    x0_vals = np.random.uniform(3000, 7000, (num_particles, 1))
    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))

    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))
    #print(X_bar_init)
    return X_bar_init


def init_particles_freespace(num_particles, occupancy_map):
    
    # initialize [x, y, theta] positions in world_frame for all particles
    """
    TODO : Add your code here
    This version converges faster than init_particles_random
    """
    # initialize [x, y, theta] positions in world_frame for all particles
    x0_vals = np.zeros((num_particles, 1))
    y0_vals = np.zeros((num_particles, 1))
    theta0_vals = np.random.uniform(-3.14,3.14, (num_particles,1))
    
    x_pos = 0
    y_pos = 0
    
    if num_particles is 1:
        x0_vals[0] = 3940
        y0_vals[0] = 3840
        print(occupancy_map[394,384])

    else:
        for i in range(num_particles):
            more_particles = True # indicates if more particles needed
            while more_particles:

                if (i < num_particles/4): # long corridor region
                    x_pos = np.random.randint(3800,4280)
                    y_pos = np.random.randint(2300,7470)
                
                elif (i >= num_particles/4 and i < num_particles/2): # room in the middle of the corridor
                    x_pos = np.random.randint(4200,5000)
                    y_pos = np.random.randint(3700,4500)
                
                elif (i >= num_particles/2 and i < 3*num_particles/4): # bottom right region
                    x_pos =np.random.randint(5800,6800)
                    y_pos = np.random.randint(0,2500)
                
                else: # bottom region
                    x_pos = np.random.randint(4000,6000)
                    y_pos = np.random.randint(0,2500)

                map_x = (int)(x_pos/10)
                map_y = (int)(y_pos/10)
                
                if (occupancy_map[map_y, map_x] <= 0.35 and occupancy_map[map_y, map_x] >=0): # check if the initialized points are in freespace
                    more_particles = False
                    #print(occupancy_map[map_y,map_x])
                
            x0_vals[i] = x_pos
            y0_vals[i] = y_pos

    # initialize weights for all particles
    w0_vals = np.ones((num_particles,1),dtype=np.float64)
    w0_vals = w0_vals / num_particles
    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))
    
    # initalize in the middle of the map at 400,400
    return X_bar_init

def init_particles_custom(num_particles, occupancy_map):
    
    # initialize [x, y, theta] positions in world_frame for all particles
    """
    TODO : Add your code here
    This version converges faster than init_particles_random
    """
    # initialize [x, y, theta] positions in world_frame for all particles
    X_bar_init = np.array([[4150,4040,3.00,1.0],
                            [5800,1500,1.50,1.0],
                            [4560,1000,1.00,1.0],
                            [6500,1480,-3.00,1.0],
                            [5310, 2230, -3.0,1.0]])
    return X_bar_init

if __name__ == '__main__':
    """
    Description of variables used
    u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
    u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
    x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
    x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
    z_t : array of 180 range measurements for each laser scan
    """
    """
    Initialize Parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='../data/map/wean.dat')
    parser.add_argument('--path_to_log', default='../data/log/robotdata1.log')
    parser.add_argument('--output', default='results')
    parser.add_argument('--num_particles', default=500, type=int)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    src_path_map = args.path_to_map
    src_path_log = args.path_to_log
    os.makedirs(args.output, exist_ok=True)

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    logfile = open(src_path_log, 'r')

    motion_model = MotionModel()
    sensor_model = SensorModel(occupancy_map)
    resampler = Resampling()

    num_particles = args.num_particles
    #X_bar = init_particles_random(num_particles, occupancy_map)
    X_bar = init_particles_freespace(num_particles, occupancy_map)
    #X_bar = init_particles_custom(num_particles, occupancy_map)
    """
    Monte Carlo Localization Algorithm : Main Loop 
    """
    if args.visualize:
        visualize_map(occupancy_map)

    first_time_idx = True
    for time_idx, line in enumerate(logfile):

        # Read a single 'line' from the log file (can be either odometry or laser measurement)
        # L : laser scan measurement, O : odometry measurement
        meas_type = line[0]

        # convert measurement values from string to double
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')

        # odometry reading [x, y, theta] in odometry frame
        odometry_robot = meas_vals[0:3]
        time_stamp = meas_vals[-1]

        # ignore pure odometry measurements for (faster debugging)
        # if ((time_stamp <= 0.0) | (meas_type == "O")):
        #     continue

        if (meas_type == "L"):
            # [x, y, theta] coordinates of laser in odometry frame
            odometry_laser = meas_vals[3:6]
            # 180 range measurement values from single laser scan
            ranges = meas_vals[6:-1]

        #print("Processing time step {} at time {}s".format(
           #time_idx, time_stamp))

        if first_time_idx:
            u_t0 = odometry_robot
            first_time_idx = False
            continue

        X_bar_new = np.zeros((num_particles, 4), dtype=np.float64)
        u_t1 = odometry_robot

        if sum(abs(u_t1[:2]-u_t0[:2])) < 1 and abs(u_t1[2]-u_t0[2]) < (math.pi/20):
            #print("Particles did not move. So skipping this step")
            continue
        # Note: this formulation is intuitive but not vectorized; looping in python is SLOW.
        # Vectorized version will receive a bonus. i.e., the functions take all particles as the input and process them in a vector.
        for m in range(0, num_particles):
            """
            MOTION MODEL
            """
            x_t0 = X_bar[m, 0:3]
            x_t1 = motion_model.update(u_t0, u_t1, x_t0)
            
            ########## Only for Debugging Motion Model. Delte Afterwards#######################
            #X_bar_new[m, :] = np.hstack((x_t1, X_bar[m, 3]))
            ###################################################################################
            """
            SENSOR MODEL
            """
            if (meas_type == "L"):
                z_t = ranges
                w_t = sensor_model.beam_range_finder_model(z_t, x_t1)
                print("Updated weight: " + str(w_t))
                X_bar_new[m, :] = np.hstack((x_t1, w_t))

            else:
                pass
                X_bar_new[m, :] = np.hstack((x_t1, X_bar[m, 3]))

        X_bar = X_bar_new
        u_t0 = u_t1
        """
        RESAMPLING
        """
        X_bar = resampler.low_variance_sampler(X_bar)
        if args.visualize:
            visualize_timestep(X_bar, time_idx, args.output)