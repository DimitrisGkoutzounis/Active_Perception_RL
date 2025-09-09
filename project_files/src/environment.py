# src/environment.py
# Functions for creating and interacting with the simulation environment.

import numpy as np
import math
import config

def point_cloud(n_samples, a, b, c):
    Sigma = np.array([a, b, c])
    rand3d = np.random.normal(loc=0.0, scale=Sigma[:, None], size=(3, n_samples))
    return rand3d

def project_point(point_3d, fx, fy):
    cx, cy = 632.69, 379.6325 
    denom = point_3d[:, 2].copy()
    denom[denom == 0] = 1e-12
    u = (point_3d[:, 0] * fx) / denom + cx
    v = (point_3d[:, 1] * fy) / denom + cy
    return np.vstack([u, v, point_3d[:, 2]]).T

def in_fov(points, image_w, image_h):
    if points.shape[1] == 3:
        points = points[:, :2]
    mask = (points[:, 0] >= 0) & (points[:, 0] <= image_w) & \
           (points[:, 1] >= 0) & (points[:, 1] <= image_h)
    return points[mask], mask 

def is_occluded(pcd2d, cam_center_2d, obstacle_center_2d, obstacle_radius):
    occluded_indices = []
    for i, Pi in enumerate(pcd2d):
        d = Pi - cam_center_2d
        cam_to_obs = obstacle_center_2d - cam_center_2d
        
      
        if np.dot(d, cam_to_obs) <= 0:
            continue
            
        A = np.dot(d, d)
        if A < 1e-12: continue
        
        B = 2 * np.dot(d, cam_center_2d - obstacle_center_2d)
        C = np.dot(cam_to_obs, cam_to_obs) - obstacle_radius**2
        
        discriminant = B**2 - 4 * A * C
        
        if discriminant >= 0:
            sqrt_D = math.sqrt(discriminant)
            t1 = (-B - sqrt_D) / (2 * A)
            t2 = (-B + sqrt_D) / (2 * A)
            if (0 < t1 < 1) or (0 < t2 < 1):
                occluded_indices.append(i)
                
    return np.array(occluded_indices)

def env_setup():
    # Generate Obstacles
    obstacle_list = []
    for i in range(config.NUM_OBSTACLES):
        obstacle_dist = np.random.uniform(config.OBSTACLE_MIN_DIST_FROM_ROI, config.OBSTACLE_MAX_DIST_FROM_ROI)
        obstacle_polar_angle = np.random.uniform(0, 2 * math.pi)
        
        obs_polar_coords = np.array([obstacle_dist, obstacle_polar_angle])
        obstacle_cart_pos = obs_polar_coords[0] * np.array([math.cos(obs_polar_coords[1]), math.sin(obs_polar_coords[1])])
        obstacle_radius = np.random.uniform(config.OBSTACLE_MIN_RADIUS, config.OBSTACLE_MAX_RADIUS)
        obstacle_list.append((obstacle_cart_pos, obstacle_radius, i))
        
    camera_dist = np.random.uniform(config.AGENT_MIN_START_DIST_FROM_ROI, config.AGENT_MAX_START_DIST_FROM_ROI)
    camera_polar_angle = np.random.uniform(0, 2 * math.pi)
    
    cam_polar_coords = np.array([camera_dist, camera_polar_angle])
    camera_cart_pos = cam_polar_coords[0] * np.array([math.cos(cam_polar_coords[1]), math.sin(cam_polar_coords[1])])
    
    return obstacle_list, camera_cart_pos