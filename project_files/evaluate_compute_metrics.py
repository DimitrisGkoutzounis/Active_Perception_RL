import numpy as np
import torch
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import time
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import config
from src.models import ActorCritic
from src.environment import point_cloud, project_point, in_fov, is_occluded, env_setup
from src.utils import compute_reward_for_training

# --- Evaluation Configuration ---
N_EVAL_EPISODES = 100
MAX_STEPS_PER_EPISODE = 70
LOG_DIR = f"runs/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(LOG_DIR)

FORCE_BAD_START = True
BAD_START_THRESHOLD = 0.5
fixed_obstacle_config = True

def calculate_occlusion_info(pcd3d_world, cam_pos_2d, obstacle_list):
    
    all_indices = np.arange(pcd3d_world.shape[1])
    pcd2d_world = pcd3d_world[:2, :].T
    cam_center = np.append(cam_pos_2d, 0.0)

    # --- Camera Orientation ---
    dist = np.array([[1,0,0],[0,1,0],[0,0,0]]) @ (config.MU - cam_center)
    norm_dist = np.linalg.norm(dist)
    if norm_dist < 1e-6: return 0.0, None, None 
    
    zc = dist / norm_dist
    xc_cand = np.cross(zc, [0, 0, 1])
    norm_xc = np.linalg.norm(xc_cand)
    if norm_xc < 1e-6: return 0.0, None, None
    
    xc = xc_cand / norm_xc
    yc = np.cross(zc, xc)
    R = np.vstack([xc, yc, zc])

    # --- Point Cloud Projection ---
    pc_camera_frame = R @ (pcd3d_world - cam_center.reshape(3, 1))
    gnt_px, _ = in_fov(project_point(pc_camera_frame.T, config.FX, config.FY), config.IMAGE_W, config.IMAGE_H)

    # --- Occlusion Calculation ---
    visible_indices = all_indices.copy()
    for obs_center_2d, obs_radius, _ in obstacle_list:
        if visible_indices.size == 0: break
        currently_visible_pcd = pcd2d_world[visible_indices]
        newly_occluded_local = is_occluded(currently_visible_pcd, cam_pos_2d, obs_center_2d, obs_radius)
        if newly_occluded_local.size > 0:
            newly_occluded_global = visible_indices[newly_occluded_local]
            visible_indices = np.setdiff1d(visible_indices, newly_occluded_global, assume_unique=True)

    obs_pc_camera_frame = pc_camera_frame[:, visible_indices]
    obs_px, _ = in_fov(project_point(obs_pc_camera_frame.T, config.FX, config.FY), config.IMAGE_W, config.IMAGE_H)

    # --- State Histogram Creation ---
    H_obs, _, _ = np.histogram2d(obs_px[:, 0], obs_px[:, 1], bins=config.BINS, range=config.HIST_RANGE)
    H_gnt, _, _ = np.histogram2d(gnt_px[:, 0], gnt_px[:, 1], bins=config.BINS, range=config.HIST_RANGE)
    
    # --- Ratio Calculation ---
    sum_gnt = np.sum(H_gnt)
    sum_obs = np.sum(H_obs)
    ratio = sum_obs / sum_gnt if sum_gnt > 0 else 0.0
    
    return ratio, H_gnt, H_obs

def run_mass_evaluation():
 
    print(f"Starting mass evaluation for {N_EVAL_EPISODES} episodes.")
    print(f"TensorBoard logs will be saved in: {LOG_DIR}")

    # --- Setup ---
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # --- Load Policy ---
    policy = ActorCritic(state_vec_dim=2, action_dim=2).to(device)
    try:
        policy.load_state_dict(torch.load(config.POLICY_PATH, map_location=device))
    except FileNotFoundError:
        print(f"ERROR: Policy file not found at {config.POLICY_PATH}. Exiting.")
        return
    policy.eval()

    # --- Statistics Accumulators ---
    all_episode_rewards = []
    total_crashes = 0
    occlusion_outcomes = {'improved': 0, 'worsened': 0, 'stayed_good': 0, 'stayed_bad': 0, 'overall_good':0, 'overall_bad':0}
    # NEW: Store a list of trajectories (each trajectory is a list of positions)
    all_episode_trajectories = []
    
    start_time = time.time()
    
    episodes_completed = 0
    total_attempts = 0
    
    # --- Main Evaluation Loop ---
    while episodes_completed < N_EVAL_EPISODES:
        total_attempts += 1
        
        a,b,c = 2.5, 2.5, 2.5
        pcd3d_world = point_cloud(config.PCD_N_SAMPLES, a, b, c) + config.MU.reshape(3, 1)

        obstacle_list, initial_camera_pos = env_setup(fixed_obstacle_config=fixed_obstacle_config)
        state_vec = initial_camera_pos
        
        initial_ratio, _, _ = calculate_occlusion_info(pcd3d_world, state_vec, obstacle_list)

        if FORCE_BAD_START and initial_ratio >= BAD_START_THRESHOLD:
            continue
        
        # NEW: Track the path for the current episode
        current_trajectory = [initial_camera_pos.copy()]
        crashed = False
        episode_total_reward = 0
        
        for step in range(MAX_STEPS_PER_EPISODE):
            _, H_gnt, H_obs = calculate_occlusion_info(pcd3d_world, state_vec, obstacle_list)
            
            state_img_tensor = torch.FloatTensor(H_obs.T).unsqueeze(0).unsqueeze(0).to(device)
            state_vec_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(device)

            with torch.no_grad():
                action_mean, _ = policy.forward(state_img_tensor, state_vec_tensor)
            
            action_np = action_mean.cpu().numpy().flatten()
            state_vec = state_vec + action_np * config.ACTION_SCALING

            # NEW: Append the new position to the current episode's trajectory
            current_trajectory.append(state_vec.copy())

            min_dist_obs = float('inf')
            for obs_center_2d, obs_radius, _ in obstacle_list:
                dist_to_obs = np.linalg.norm(state_vec - obs_center_2d)
                min_dist_obs = min(min_dist_obs, dist_to_obs)
                if dist_to_obs < obs_radius:
                    crashed = True
                    break
            
            distance_to_roi = np.linalg.norm(state_vec - config.MU[:2])
            reward, ratio = compute_reward_for_training(H_gnt.flatten(), H_obs.flatten(), distance_to_roi, config.DIST_MIN, min_dist_obs)

            if crashed:
                reward = -3000 

            episode_total_reward += reward
            
            if crashed:
                break
        
        # --- End of Episode: Log Metrics ---
        all_episode_rewards.append(episode_total_reward)
        final_ratio, _, _ = calculate_occlusion_info(pcd3d_world, state_vec, obstacle_list)
        
        occlusion_threshold = 0.85 
        is_initial_good = initial_ratio >= occlusion_threshold
        is_final_good = final_ratio >= occlusion_threshold
        
        if not is_initial_good and is_final_good:
            occlusion_outcomes['improved'] += 1
        elif is_initial_good and not is_final_good:
            occlusion_outcomes['worsened'] += 1
        elif is_initial_good and is_final_good:
            occlusion_outcomes['stayed_good'] += 1
        else:
            occlusion_outcomes['stayed_bad'] += 1
                    
        if crashed:
            total_crashes += 1
        
        # NEW: Add the completed trajectory to our list of all trajectories
        all_episode_trajectories.append(current_trajectory)

        print(f"Finished Episode {episodes_completed+1}/{N_EVAL_EPISODES} | Reward: {episode_total_reward:.2f} | Crashed: {crashed} | Ratio: {initial_ratio:.2f} -> {final_ratio:.2f} | status: {occlusion_outcomes}")
        episodes_completed += 1

    # --- Final Summary ---
    end_time = time.time()
    print("\n--- Mass Evaluation Summary ---")
    
    avg_reward = np.mean(all_episode_rewards)
    std_reward = np.std(all_episode_rewards)
    crash_rate = (total_crashes / N_EVAL_EPISODES) * 100
    
    if FORCE_BAD_START:
        print(f"Found and evaluated {N_EVAL_EPISODES} episodes with bad start after {total_attempts}")

    print(f"Average Episode Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Crash Rate: {crash_rate:.2f}% ({total_crashes}/{N_EVAL_EPISODES})")
    
    occlusion_outcomes['overall_good'] = occlusion_outcomes['improved'] + occlusion_outcomes['stayed_good']
    occlusion_outcomes['overall_bad'] = occlusion_outcomes['worsened'] + occlusion_outcomes['stayed_bad']
    
    print("\nOcclusion Outcomes:")
    for key, value in occlusion_outcomes.items():
        rate = (value / N_EVAL_EPISODES) * 100
        print(f"  - {key.replace('_', ' ').title()}: {value}/{N_EVAL_EPISODES} ({rate:.2f}%)")
    
    total_time = end_time - start_time
    print(f"\nTotal evaluation time: {total_time:.2f} seconds.")
    
    hparam_dict = {
        'n_episodes': N_EVAL_EPISODES,
        'policy': config.POLICY_PATH,
        'max_steps_per_episode': MAX_STEPS_PER_EPISODE
    }
    metric_dict = {
        'summary/average_reward': avg_reward,
        'summary/crash_rate': crash_rate,
        'summary/occlusion_improved_rate': (occlusion_outcomes['improved'] / N_EVAL_EPISODES) * 100,
        'summary/occlusion_stayed_good_rate': (occlusion_outcomes['stayed_good'] / N_EVAL_EPISODES) * 100,
        'summary/occlusion_stayed_bad_rate': (occlusion_outcomes['stayed_bad'] / N_EVAL_EPISODES) * 100,
        'summary/occlusion_worsened_rate': (occlusion_outcomes['worsened'] / N_EVAL_EPISODES) * 100,
        'summary/occlusion_overall_good_rate': (occlusion_outcomes['overall_good'] / N_EVAL_EPISODES) * 100,
        'summary/occlusion_overall_bad_rate': (occlusion_outcomes['overall_bad'] / N_EVAL_EPISODES) * 100
    }

    bad_start_dict = {
        'force_bad_start': FORCE_BAD_START,
        'bad_start_threshold': BAD_START_THRESHOLD
        
    }
    # Combine hparams and metrics into one dictionary
    summary_dict = {
        'hparams': hparam_dict,
        'metrics': metric_dict,
        'bad_start_config': bad_start_dict
    }
    

    # Save to json
    metrics_path = os.path.join(writer.log_dir, 'summary_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(summary_dict, f, indent=2)
        
        
    
    
    print(f"--- Evaluation Finished. Logs saved to {LOG_DIR} ---")
    
    # --- MODIFIED PLOTTING SECTION ---
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each episode's trajectory as a line
    for trajectory in all_episode_trajectories:
        traj_np = np.array(trajectory)
        ax.plot(traj_np[:, 0], traj_np[:, 1], marker='.', markersize=4, alpha=0.7)
        # Mark the starting point of each trajectory
        # ax.plot(traj_np[0, 0], traj_np[0, 1], 'go', markersize=6, label='Start' if 'Start' not in [l.get_label() for l in ax.get_legend().get_texts()] else "")

    # Plot the last known point cloud
    ax.scatter(pcd3d_world[0,:], pcd3d_world[1,:], label='Point Cloud', s=5, alpha=0.3, color='gray')

    # Plot obstacles
    for obs_pos_2d, obs_radius, obs_id in obstacle_list:
        circle = plt.Circle(obs_pos_2d, obs_radius, fill=False, color='r', linewidth=2)
        ax.add_patch(circle)
        ax.text(obs_pos_2d[0], obs_pos_2d[1], str(obs_id), color='r', fontsize=12, ha='center', va='center')

    # Add invisible plot for the legend entry
    ax.plot([], [], color='r', linewidth=2, label='Obstacles')[0].set_visible(False)
    
    # Plot ROI center
    ax.scatter(config.MU[0], config.MU[1], c='b', marker='x', s=150, label='ROI center')

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_title(f"Agent Trajectories for {N_EVAL_EPISODES} episodes")
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True)
    
    # Save the figure
    if FORCE_BAD_START:
        fig.savefig(os.path.join(LOG_DIR, "evaluation_trajectories_bad_starts.png" ))
    else:
        fig.savefig(os.path.join(LOG_DIR, "evaluation_trajectories.png"))
    plt.show()

if __name__ == '__main__':
    
    
    all_cam_positions = []
    
    
    
    run_mass_evaluation()