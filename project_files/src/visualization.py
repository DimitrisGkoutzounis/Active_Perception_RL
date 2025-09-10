# src/visualization.py
# All functions related to plotting and creating visualizations.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import time

import config

from src.environment import project_point, in_fov, is_occluded
from src.utils import compute_reward, compute_reward_for_training

def show_map_multiple_obstacles(obstacle_list, mu, pcd3d, d_min, trajectory=None, reward_map=None, reward_map_extent=None):
    clicked_point = []
    
    def onclick(event):
        if event.inaxes:
            ix, iy = event.xdata, event.ydata
            print(f'Starting position selected at: x={ix:.2f}, y={iy:.2f}')
            clicked_point.extend([ix, iy])
            plt.close(event.canvas.figure)

    fig, ax = plt.subplots(figsize=(9, 9))
    
    if reward_map is not None and reward_map_extent is not None:
        vmin = np.nanmin(reward_map)
        vmax = np.nanmax(reward_map)
        im = ax.imshow(reward_map, cmap='viridis', origin='lower',
                       extent=reward_map_extent, alpha=0.5, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Reward')

    color_center = np.mean(pcd3d, axis=1).reshape(3, -1)
    color = -np.linalg.norm(pcd3d - color_center, axis=0)
    ax.scatter(pcd3d[0,:], pcd3d[1,:], c=color, cmap='inferno', label='ROI')
    ax.scatter(mu[0], mu[1], c='blue', marker='x', s=200, label='ROI Center')

    for obs_pos_2d, obs_radius, obs_id in obstacle_list:
        ax.add_patch(plt.Circle(obs_pos_2d, obs_radius, fill=False, color='red', linewidth=2))
        ax.text(obs_pos_2d[0], obs_pos_2d[1], str(obs_id), color='red', fontsize=12, ha='center', va='center')
    ax.plot([], [], color='r', linewidth=2, label='Obstacles')[0].set_visible(False)

    camera_min_area_circle = plt.Circle(mu[:2], config.AGENT_MIN_START_DIST_FROM_ROI, fill=False, color='green', label='Camera Area')
    camera_max_area_circle = plt.Circle(mu[:2], config.AGENT_MAX_START_DIST_FROM_ROI, fill=False, color='green')
    obstacle_min_area_circle = plt.Circle(mu[:2], config.OBSTACLE_MIN_DIST_FROM_ROI, fill=False, color='red', label='Obstacle Area', linestyle='dashed')
    obstacle_max_area_circle = plt.Circle(mu[:2], config.OBSTACLE_MAX_DIST_FROM_ROI, fill=False, color='red', linestyle='dashed')
    min_distance_circle = plt.Circle(mu[:2], config.DIST_MIN, fill=False, color='black', label='Minimum Distance')
    
    ax.add_patch(camera_min_area_circle)
    ax.add_patch(camera_max_area_circle)
    ax.add_patch(obstacle_min_area_circle)
    ax.add_patch(obstacle_max_area_circle)
    ax.add_patch(min_distance_circle)

    if isinstance(trajectory, np.ndarray) and trajectory.size > 0:
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'g-o', label='Agent Trajectory', markersize=5)
        plt.scatter(trajectory[0, 0], trajectory[0, 1], c='orange', marker='s', s=150, label='Start Position', zorder=10)
        plt.scatter(trajectory[-1, 0], trajectory[-1, 1], c='green', marker='*', s=300, label='Final Position', zorder=10)
        ax.set_title("Trajectory", fontsize=16)
    else:
        ax.set_title("Click to start the beast", fontsize=16)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    # ax.legend() 
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(config.ENV_BOUNDS_X[0], config.ENV_BOUNDS_X[1])
    ax.set_ylim(config.ENV_BOUNDS_Y[0], config.ENV_BOUNDS_Y[1])
    ax.set_aspect('equal', adjustable='box')

    # If no trajectory is provided, enable click to select start position
    # if trajectory is None:
    #     fig.canvas.mpl_connect('button_press_event', onclick)
    #     print("Click to start the beast")
    #     plt.show()
    #     return clicked_point
    # else:
    #     plt.show()
    #     return None
    
    fig.canvas.mpl_connect('button_press_event', onclick)
    print("Please click on the map to set the start position.")
    plt.show()
    
    return clicked_point

def plot_run_diagnostics(rewards, trajectory, n_steps):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('Reward and position', fontsize=16)
    steps = np.arange(len(rewards))

    ax1.plot(steps, rewards, 'b-o', markersize=4)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Reward')
    ax1.set_title('Reward')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(steps, trajectory[:n_steps, 0], 'r-o', markersize=4, label='X')
    ax2.plot(steps, trajectory[:n_steps, 1], 'g-s', markersize=4, label='Y')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Position')
    ax2.set_title('Agent Position')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()

def plot_frame_comparison(step_title, gnt_points, obs_points, H_gnt, H_obs, image_w, image_h, hist_range, gnt_intensity=None, obs_intensity=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    fig.suptitle(step_title, fontsize=16)
    
    # --- Top-left: Ground Truth Points ---
    ax = axes[0, 0]
    ax.set_title('Ground-Truth View (FOV)')
    if gnt_points.shape[0] > 0:
        sc = ax.scatter(gnt_points[:, 0], gnt_points[:, 1], s=5, c=gnt_intensity, cmap='inferno')
        fig.colorbar(sc, ax=ax, label='Intensity')
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-50, image_w + 50)
    ax.set_ylim(-50, image_h + 50)

    # --- Top-right: Observed Points ---
    ax = axes[0, 1]
    ax.set_title('Observed View (FOV)')
    if obs_points.shape[0] > 0:
        sc = ax.scatter(obs_points[:, 0], obs_points[:, 1], s=5, c=obs_intensity, cmap='inferno')
        fig.colorbar(sc, ax=ax, label='Intensity')
    rect = patches.Rectangle((0, 0), image_w, image_h, linewidth=1.5, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-50, image_w + 50)
    ax.set_ylim(-50, image_h + 50)
    
    # --- Bottom-left: Ground Truth Heatmap ---
    ax = axes[1, 0]
    ax.set_title('Ground Truth Heatmap (FOV)')
    im = ax.imshow(H_gnt, interpolation='nearest', origin='lower', extent=[hist_range[0][0], hist_range[0][1], hist_range[1][0], hist_range[1][1]])
    fig.colorbar(im, ax=ax)

    # --- Bottom-right: Observed Heatmap ---
    ax = axes[1, 1]
    ax.set_title('Observed Heatmap (FOV)')
    im = ax.imshow(H_obs, interpolation='nearest', origin='lower', extent=[hist_range[0][0], hist_range[0][1], hist_range[1][0], hist_range[1][1]])
    fig.colorbar(im, ax=ax)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()

def generate_reward_map(grid_size, bounds, pcd3d_world, obstacle_list, mu, fx, fy, image_w, image_h, BINS, hist_range, DIST_MIN):
    """Computes the reward for each point in a grid to create a reward heatmap."""
    print("Generating reward map...")
    start_time = time.time()
    
    if grid_size is None:
        return None, None

    x_coords = np.linspace(bounds[0], bounds[1], grid_size)
    y_coords = np.linspace(bounds[2], bounds[3], grid_size)
    reward_map = np.full((grid_size, grid_size), np.nan) 

    pcd2d_world = pcd3d_world[:2, :].T
    all_indices = np.arange(pcd2d_world.shape[0])

    for iy, y_pos in enumerate(y_coords):
        for ix, x_pos in enumerate(x_coords):
            cam_center_i = np.array([x_pos, y_pos, 0.0])
            dist_i = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]) @ (mu - cam_center_i)
            norm_dist_i = np.linalg.norm(dist_i)
            if norm_dist_i < 1e-6: continue 
            
            zc_i = dist_i / norm_dist_i
            xc_i_cand = np.cross(zc_i, [0, 0, 1])
            norm_xc_i = np.linalg.norm(xc_i_cand)
            if norm_xc_i < 1e-6: continue
            
            xc_i = xc_i_cand / norm_xc_i
            yc_i = np.cross(zc_i, xc_i)
            R_i = np.vstack([xc_i, yc_i, zc_i])
            pc_i = R_i @ (pcd3d_world - cam_center_i.reshape(3, 1))

            gnt_point_cloud_px_i = project_point(pc_i.T, fx, fy)
            gnt_point_cloud_px_fov_i, _ = in_fov(gnt_point_cloud_px_i, image_w, image_h)

            visible_indices = all_indices.copy()

            for obs_center_2d, obs_radius, _ in obstacle_list:
                if visible_indices.size == 0:
                    break

                currently_visible_pcd = pcd2d_world[visible_indices]

                newly_occluded_local_indices = is_occluded(
                    currently_visible_pcd, cam_center_i[:2], obs_center_2d, obs_radius
                )

                if newly_occluded_local_indices.size > 0:
                    newly_occluded_global_indices = visible_indices[newly_occluded_local_indices]

                    
                    visible_indices = np.setdiff1d(
                        visible_indices, newly_occluded_global_indices, assume_unique=True
                    )

            still_visible_indices = visible_indices
            
            observed_pc_camera_i = pc_i[:, still_visible_indices]
            observed_point_cloud_px_i = project_point(observed_pc_camera_i.T, fx, fy)
            observed_point_cloud_px_fov_i, _ = in_fov(observed_point_cloud_px_i, image_w, image_h)

            H_obs, _, _ = np.histogram2d(
                observed_point_cloud_px_fov_i[:, 0], observed_point_cloud_px_fov_i[:, 1], 
                bins=BINS, range=hist_range
            )
            
            H_gnt, _, _ = np.histogram2d(
                gnt_point_cloud_px_fov_i[:, 0], gnt_point_cloud_px_fov_i[:, 1], 
                bins=BINS, range=hist_range
            )
            
            distance_to_roi = np.linalg.norm(cam_center_i[:2] - mu[:2])
            
            min_dist_obs = float('inf')
            for obs_center_2d, obs_radius, _ in obstacle_list:
                distance_to_obs = np.linalg.norm(cam_center_i[:2] - obs_center_2d)

                if distance_to_obs < min_dist_obs:
                    min_dist_obs = distance_to_obs
                    
                if distance_to_obs < obs_radius:
                    crashed = True
                    break
            
            # distance to obs is the closest distance
            distance_to_obs = min_dist_obs 
            
            # reward, _ = compute_reward(H_gnt.flatten(), H_obs.flatten(), distance_to_roi, DIST_MIN)
            reward , _ = compute_reward_for_training(H_gnt.flatten(), H_obs.flatten(),
                                                    distance_to_roi, DIST_MIN, distance_to_obs)
            reward_map[iy, ix] = reward
            
    end_time = time.time()
    print(f"Reward map generated in {end_time - start_time:.2f} seconds.")
    return reward_map, [bounds[0], bounds[1], bounds[2], bounds[3]]