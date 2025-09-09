# src/environment.py
# Functions for creating and interacting with the simulation environment.

import numpy as np
import math
import config
import torch

def point_cloud(n_samples, a, b, c):
    """Generates a 3D point cloud with a given covariance."""
    Sigma = np.array([a, b, c])
    rand3d = np.random.normal(loc=0.0, scale=Sigma[:, None], size=(3, n_samples))
    return rand3d

def project_point(point_3d_t: torch.Tensor, fx: float, fy: float) -> torch.Tensor:
    """Projects 3D points to 2D pixel coordinates."""
    cx, cy = 632.69, 379.6325
    
    # Add a small epsilon to the z-coordinate to prevent division by zero
    z = point_3d_t[:, 2]
    denom = z + 1e-12
    
    u = (point_3d_t[:, 0] * fx) / denom + cx
    v = (point_3d_t[:, 1] * fy) / denom + cy
    
    return torch.stack([u, v, z], dim=1)

def in_fov(points_t: torch.Tensor, image_w: int, image_h: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Filters points that are within the camera's field of view."""
     # Create a boolean mask for points within the image boundaries
    mask = (points_t[:, 0] >= 0) & (points_t[:, 0] < image_w) & \
           (points_t[:, 1] >= 0) & (points_t[:, 1] < image_h)
           
    return points_t[mask], mask

def is_occluded(pcd2d_t: torch.Tensor, cam_center_2d_t: torch.Tensor, 
                      obstacle_center_2d_t: torch.Tensor, obstacle_radius: float) -> torch.Tensor:
   
    d_t = pcd2d_t - cam_center_2d_t
    cam_to_obs_t = obstacle_center_2d_t - cam_center_2d_t

    A_t = torch.sum(d_t * d_t, dim=1)
    
    B_t = 2 * torch.sum(d_t * (cam_center_2d_t - obstacle_center_2d_t), dim=1)
    C = torch.dot(cam_to_obs_t, cam_to_obs_t) - obstacle_radius**2
    
    discriminant = B_t**2 - 4 * A_t * C

    intersect_mask = (discriminant >= 0) & (A_t > 1e-9)
    if not torch.any(intersect_mask):
        return torch.tensor([], dtype=torch.long, device=pcd2d_t.device)

    A_intersect = A_t[intersect_mask]
    B_intersect = B_t[intersect_mask]
    D_intersect = discriminant[intersect_mask]
    
    sqrt_D = torch.sqrt(D_intersect)
    
    # Calculate the two potential solutions for t
    t1 = (-B_intersect - sqrt_D) / (2 * A_intersect)
    t2 = (-B_intersect + sqrt_D) / (2 * A_intersect)
    
    occlusion_condition = ((t1 > 0) & (t1 < 1)) | ((t2 > 0) & (t2 < 1))
    
    original_indices = torch.arange(pcd2d_t.shape[0], device=pcd2d_t.device)[intersect_mask]
    occluded_indices = original_indices[occlusion_condition]
    
    return occluded_indices

def env_setup():
    
    obstacle_positions = []
    obstacle_radii = []
    
    # Generate Obstacles
    for i in range(config.NUM_OBSTACLES):
        obstacle_dist = np.random.uniform(config.OBSTACLE_MIN_DIST_FROM_ROI, config.OBSTACLE_MAX_DIST_FROM_ROI)
        obstacle_polar_angle = np.random.uniform(0, 2 * math.pi)
        
        # Convert polar to cartesian coordinates
        obstacle_cart_pos = obstacle_dist * np.array([math.cos(obstacle_polar_angle), math.sin(obstacle_polar_angle)])
        obstacle_radius = np.random.uniform(config.OBSTACLE_MIN_RADIUS, config.OBSTACLE_MAX_RADIUS)
        
        obstacle_positions.append(obstacle_cart_pos)
        obstacle_radii.append(obstacle_radius)
        
    # Convert the lists into single, efficient NumPy arrays
    obstacle_positions_np = np.array(obstacle_positions, dtype=np.float32)
    obstacle_radii_np = np.array(obstacle_radii, dtype=np.float32)

    # Generate Agent Start Position (this part remains the same)
    camera_dist = np.random.uniform(config.AGENT_MIN_START_DIST_FROM_ROI, config.AGENT_MAX_START_DIST_FROM_ROI)
    camera_polar_angle = np.random.uniform(0, 2 * math.pi)
    camera_cart_pos = camera_dist * np.array([math.cos(camera_polar_angle), math.sin(camera_polar_angle)])
    
    return obstacle_positions_np, obstacle_radii_np, camera_cart_pos