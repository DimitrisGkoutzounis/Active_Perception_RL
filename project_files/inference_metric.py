import numpy as np
import torch
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import time

import config
from src.models import ActorCritic
from src.environment import point_cloud, project_point, in_fov, is_occluded, env_setup
from src.utils import compute_reward_for_training

# --- Evaluation Configuration ---
N_EVAL_EPISODES = 100
MAX_STEPS_PER_EPISODE = 50
# Create a distinct directory for this evaluation run
LOG_DIR = f"runs/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def calculate_occlusion_info(pcd3d_world, cam_pos_2d, obstacle_list):
    """
    Helper function to compute the visibility ratio for a given camera position.
    This encapsulates the simulation logic needed to get the H_gnt and H_obs histograms.
    Returns the visibility ratio and the state histograms.
    """
    all_indices = np.arange(pcd3d_world.shape[1])
    pcd2d_world = pcd3d_world[:2, :].T
    cam_center = np.append(cam_pos_2d, 0.0)

    # --- Camera Orientation ---
    dist = np.array([[1,0,0],[0,1,0],[0,0,0]]) @ (config.MU - cam_center)
    norm_dist = np.linalg.norm(dist)
    if norm_dist < 1e-6: return 0.0, None, None # Should not happen in valid positions
    
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
    """
    Main function to run multiple evaluation episodes and log performance metrics.
    """
    print(f"Starting mass evaluation for {N_EVAL_EPISODES} episodes.")
    print(f"TensorBoard logs will be saved in: {LOG_DIR}")

    # --- Setup ---
    writer = SummaryWriter(log_dir=LOG_DIR)
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
    occlusion_outcomes = {'improved': 0, 'worsened': 0, 'stayed_good': 0, 'stayed_bad': 0}
    
    start_time = time.time()

    # --- Main Evaluation Loop ---
    for episode in range(N_EVAL_EPISODES):
        # Create a new random environment for each episode for robust evaluation
        a = np.random.uniform(config.PCD_A_MIN, config.PCD_A_MAX)
        b = np.random.uniform(config.PCD_B_MIN, config.PCD_B_MAX)
        c = np.random.uniform(config.PCD_C_MIN, config.PCD_C_MAX)
        pcd3d_world = point_cloud(config.PCD_N_SAMPLES, a, b, c) + config.MU.reshape(3, 1)
        
        obstacle_list, initial_camera_pos = env_setup()
        state_vec = initial_camera_pos
        
        # --- Assess Initial Position ---
        initial_ratio, _, _ = calculate_occlusion_info(pcd3d_world, state_vec, obstacle_list)

        # --- Per-Episode Tracking ---
        crashed = False
        episode_total_reward = 0
        rewards_per_step = []

        for step in range(MAX_STEPS_PER_EPISODE):
            # Calculate current state and visibility
            _, H_gnt, H_obs = calculate_occlusion_info(pcd3d_world, state_vec, obstacle_list)
            
            # Create state tensors for the policy
            state_img_tensor = torch.FloatTensor(H_obs.T).unsqueeze(0).unsqueeze(0).to(device)
            state_vec_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(device)

            # --- Agent Action (Inference) ---
            with torch.no_grad():
                action_mean, _ = policy.forward(state_img_tensor, state_vec_tensor)
            
            action_np = action_mean.cpu().numpy().flatten()
            state_vec = state_vec + action_np * config.ACTION_SCALING

            # --- Crash Detection ---
            min_dist_obs = float('inf')
            for obs_center_2d, obs_radius, _ in obstacle_list:
                dist_to_obs = np.linalg.norm(state_vec - obs_center_2d)
                min_dist_obs = min(min_dist_obs, dist_to_obs)
                if dist_to_obs < obs_radius:
                    crashed = True
                    break
            
            # --- Reward Calculation ---
            distance_to_roi = np.linalg.norm(state_vec - config.MU[:2])
            reward, ratio = compute_reward_for_training(H_gnt.flatten(), H_obs.flatten(), distance_to_roi, config.DIST_MIN, min_dist_obs)

            if crashed:
                reward = -3000 

            episode_total_reward += reward
            rewards_per_step.append(reward)
            
            writer.add_scalars('Per_Step_Metrics/Reward', {f'episode_{episode}': reward}, step)

            if crashed:
                break
        
        # --- End of Episode: Log Metrics ---
        
        # a. Reward over step for each episode (logged inside the loop)
        all_episode_rewards.append(episode_total_reward)
        
        # b. Average reward (will be computed at the end)
        writer.add_scalar('Per_Episode_Metrics/Total_Reward', episode_total_reward, episode)
        
        # c. Initial vs. Final Occlusion Ratio
        final_ratio, _, _ = calculate_occlusion_info(pcd3d_world, state_vec, obstacle_list)
        writer.add_scalar('Per_Episode_Metrics/Initial_Visibility_Ratio', initial_ratio, episode)
        writer.add_scalar('Per_Episode_Metrics/Final_Visibility_Ratio', final_ratio, episode)
        
        # Categorize occlusion outcome
        occlusion_threshold = 0.85 # Define what counts as "good" visibility
        is_initial_good = initial_ratio >= occlusion_threshold
        is_final_good = final_ratio >= occlusion_threshold
        
        if not is_initial_good and is_final_good:
            occlusion_outcomes['improved'] += 1
        elif is_initial_good and not is_final_good:
            occlusion_outcomes['worsened'] += 1
        elif is_initial_good and is_final_good:
            occlusion_outcomes['stayed_good'] += 1
        else: # not initial_good and not final_good
            occlusion_outcomes['stayed_bad'] += 1
        
        # d. Crash Logging
        if crashed:
            total_crashes += 1
        writer.add_scalar('Per_Episode_Metrics/Crashed', int(crashed), episode)

        # e. Other Necessary Metrics
        final_dist_to_roi = np.linalg.norm(state_vec - config.MU[:2])
        writer.add_scalar('Per_Episode_Metrics/Final_Distance_to_ROI', final_dist_to_roi, episode)
        
        final_dist_to_obs = min_dist_obs
        writer.add_scalar('Per_Episode_Metrics/Final_Distance_to_Obstacle', final_dist_to_obs, episode)

        print(f"Finished Episode {episode+1}/{N_EVAL_EPISODES} | Reward: {episode_total_reward:.2f} | Crashed: {crashed} | Ratio: {initial_ratio:.2f} -> {final_ratio:.2f}")

    # --- Final Summary ---
    end_time = time.time()
    print("\n--- Mass Evaluation Summary ---")
    
    avg_reward = np.mean(all_episode_rewards)
    std_reward = np.std(all_episode_rewards)
    crash_rate = (total_crashes / N_EVAL_EPISODES) * 100
    
    print(f"Average Episode Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Crash Rate: {crash_rate:.2f}% ({total_crashes}/{N_EVAL_EPISODES})")
    
    print("\nOcclusion Outcomes:")
    for key, value in occlusion_outcomes.items():
        rate = (value / N_EVAL_EPISODES) * 100
        print(f"  - {key.replace('_', ' ').title()}: {value}/{N_EVAL_EPISODES} ({rate:.2f}%)")
    
    total_time = end_time - start_time
    print(f"\nTotal evaluation time: {total_time:.2f} seconds.")

    # Log summary metrics
    # CORRECTED CODE
    writer.add_hparams(
        hparam_dict={
            'n_episodes': N_EVAL_EPISODES,
            'policy': config.POLICY_PATH
        },
        metric_dict={
            'summary/average_reward': avg_reward,
            'summary/crash_rate': crash_rate,
            'summary/occlusion_improved_rate': (occlusion_outcomes['improved'] / N_EVAL_EPISODES) * 100,
            'summary/occlusion_stayed_good_rate': (occlusion_outcomes['stayed_good'] / N_EVAL_EPISODES) * 100,
            'summary/occlusion_stayed_bad_rate': (occlusion_outcomes['stayed_bad'] / N_EVAL_EPISODES) * 100,
            'summary/occlusion_worsened_rate': (occlusion_outcomes['worsened'] / N_EVAL_EPISODES) * 100
        }
    )
    
    writer.close()
    print(f"--- Evaluation Finished. Logs saved to {LOG_DIR} ---")


if __name__ == '__main__':
    run_mass_evaluation()