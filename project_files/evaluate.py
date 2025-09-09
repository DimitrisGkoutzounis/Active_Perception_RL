
import numpy as np
import torch
import matplotlib.pyplot as plt

# Import project-specific modules
import config
from src.models import ActorCritic
from src.environment import point_cloud, project_point, in_fov, is_occluded, env_setup
from src.utils import compute_reward, compute_reward_for_training
from src.visualization import show_map_multiple_obstacles, plot_run_diagnostics, plot_frame_comparison, generate_reward_map

def run_evaluation():
   
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device} for inference.")
   
    # --- Environment Setup ---
    obstacle_list, _ = env_setup()
    
    a = np.random.uniform(config.PCD_A_MIN, config.PCD_A_MAX)
    b = np.random.uniform(config.PCD_B_MIN, config.PCD_B_MAX)
    c = np.random.uniform(config.PCD_C_MIN, config.PCD_C_MAX)
    pcd3d_world = point_cloud(config.PCD_N_SAMPLES, a, b, c) + config.MU.reshape(3, 1)
    pcd2d_world = pcd3d_world[:2, :].T
    all_indices = np.arange(pcd2d_world.shape[0])

    # --- Generate and Display Reward Map ---
    reward_map, reward_map_extent = generate_reward_map(
        grid_size=config.EVAL_GRID_SIZE, bounds=config.EVAL_MAP_BOUNDS, 
        pcd3d_world=pcd3d_world, obstacle_list=obstacle_list, mu=config.MU, 
        fx=config.FX, fy=config.FY, image_w=config.IMAGE_W, image_h=config.IMAGE_H, 
        BINS=config.BINS, hist_range=config.HIST_RANGE, DIST_MIN=config.DIST_MIN
    )
    # reward_map = None
    # reward_map_extent= None
    clicked_point = show_map_multiple_obstacles(
        obstacle_list=obstacle_list, mu=config.MU, pcd3d=pcd3d_world, d_min=config.DIST_MIN,
        trajectory=None, reward_map=reward_map, reward_map_extent=reward_map_extent
    )
    
    if not clicked_point:
        print("No point clicked. Exiting evaluation.")
        return

    # --- Load Policy and Initialize ---
    state_vec = np.array(clicked_point)
    policy = ActorCritic(state_vec_dim=2, action_dim=2).to(device)
    policy.load_state_dict(torch.load(config.POLICY_PATH, map_location=device))
    policy.eval()

    trajectory = [state_vec.copy()]
    rewards_over_time = []
    initial_frame_data = {}
    
    # --- Compute Point Cloud Intensity ---
    intensity_center = np.mean(pcd3d_world, axis=1).reshape(3,-1)
    intensity = -np.linalg.norm(pcd3d_world - intensity_center, axis=0)

    # --- Run Simulation Loop ---
    for step in range(config.EVAL_N_STEPS):
        cam_center_i = np.append(state_vec, 0.0)
        
        # Camera Orientation
        dist_i = np.array([[1,0,0],[0,1,0],[0,0,0]]) @ (config.MU - cam_center_i)
        norm_dist_i = np.linalg.norm(dist_i)
        if norm_dist_i < 1e-6: continue
        zc_i = dist_i / norm_dist_i
        xc_i_cand = np.cross(zc_i, [0, 0, 1])
        norm_xc_i = np.linalg.norm(xc_i_cand)
        if norm_xc_i < 1e-6: continue
        xc_i = xc_i_cand / norm_xc_i
        yc_i = np.cross(zc_i, xc_i)
        R_i = np.vstack([xc_i, yc_i, zc_i])
        
        # --- Point Cloud Projection and Occlusion ---
        pc_i = R_i @ (pcd3d_world - cam_center_i.reshape(3, 1))
        
        gnt_point_cloud_px_i = project_point(pc_i.T, config.FX, config.FY)
        gnt_point_cloud_px_fov_i, gnt_mask = in_fov(gnt_point_cloud_px_i, config.IMAGE_W, config.IMAGE_H)
        gnt_weights_in_fov = intensity[gnt_mask]

        still_visible_indices = all_indices.copy()
        for obs_center_2d, obs_radius, _ in obstacle_list:
            occluded_by_obs = is_occluded(pcd2d_world, cam_center_i[:2], obs_center_2d, obs_radius)
            still_visible_indices = np.setdiff1d(still_visible_indices, occluded_by_obs, assume_unique=True)
        
        observed_pc_camera_i = pc_i[:, still_visible_indices]
        observed_intensity = intensity[still_visible_indices]
        
        observed_point_cloud_px_i = project_point(observed_pc_camera_i.T, config.FX, config.FY)
        observed_point_cloud_px_fov_i, obs_mask = in_fov(observed_point_cloud_px_i, config.IMAGE_W, config.IMAGE_H)
        obs_weights_in_fov = observed_intensity[obs_mask]

        # State Creation and Reward Calculation
        H_obs, _, _ = np.histogram2d(
            observed_point_cloud_px_fov_i[:, 0], observed_point_cloud_px_fov_i[:, 1], 
            bins=config.BINS, range=config.HIST_RANGE
        )
        H_gnt, _, _ = np.histogram2d(
            gnt_point_cloud_px_fov_i[:, 0], gnt_point_cloud_px_fov_i[:, 1], 
            bins=config.BINS, range=config.HIST_RANGE
        )
        
        distance_to_roi = np.linalg.norm(state_vec - config.MU[:2])
        print("DIST", distance_to_roi)
        distance_to_obs = None
        crashed = False
        # reward, ratio = compute_reward_for_training(H_gnt.flatten(), H_obs.flatten(), distance_to_roi, config.DIST_MIN)
        reward, ratio = compute_reward_for_training(H_gnt.flatten(), H_obs.flatten(), distance_to_roi, config.DIST_MIN, distance_to_obs if not crashed else 0.0)

        rewards_over_time.append(reward)
        print(f"Step: {step+1}/{config.EVAL_N_STEPS} | Position: [{state_vec[0]:.2f}, {state_vec[1]:.2f}] | Reward: {reward:.4f} | Ratio: {ratio:.4f}")

        # Store data for initial frame plot
        if step == 0:
            initial_frame_data['gnt_points'] = gnt_point_cloud_px_fov_i.copy()
            initial_frame_data['obs_points'] = observed_point_cloud_px_fov_i.copy()
            initial_frame_data['H_gnt'] = H_gnt.T.copy()
            initial_frame_data['H_obs'] = H_obs.T.copy()
            initial_frame_data['gnt_intensity'] = gnt_weights_in_fov.copy()
            initial_frame_data['obs_intensity'] = obs_weights_in_fov.copy()

        # Agent Action
        state_img_tensor = torch.FloatTensor(H_obs.T).unsqueeze(0).unsqueeze(0).to(device)
        state_vec_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(device)

        with torch.no_grad():
            action_mean, _ = policy.forward(state_img_tensor, state_vec_tensor)
            action_tensor = action_mean # Use the deterministic action_mean

        # action_np = action_tensor.cpu().numpy().flatten()
        # state_vec = state_vec + action_np * config.ACTION_SCALING
        # trajectory.append(state_vec.copy())

        action_np = action_mean.cpu().numpy().flatten()
        state_vec = state_vec + action_np * config.ACTION_SCALING
        trajectory.append(state_vec.copy())
        
    trajectory = np.array(trajectory)

    # --- Generate Final Plots ---
    plot_run_diagnostics(rewards_over_time, trajectory, config.EVAL_N_STEPS)
    plot_frame_comparison(
        step_title='Initial Frame (Step 0)', **initial_frame_data,
        image_w=config.IMAGE_W, image_h=config.IMAGE_H, hist_range=config.HIST_RANGE
    )
    plot_frame_comparison(
        step_title=f'Final Frame (Step {config.EVAL_N_STEPS})',
        gnt_points=gnt_point_cloud_px_fov_i, obs_points=observed_point_cloud_px_fov_i,
        H_gnt=H_gnt.T, H_obs=H_obs.T,
        image_w=config.IMAGE_W, image_h=config.IMAGE_H, hist_range=config.HIST_RANGE,
        gnt_intensity=gnt_weights_in_fov, obs_intensity=obs_weights_in_fov
    )
    show_map_multiple_obstacles(
        obstacle_list=obstacle_list, mu=config.MU, pcd3d=pcd3d_world, d_min=config.DIST_MIN,
        trajectory=trajectory, reward_map=reward_map, reward_map_extent=reward_map_extent
    )
    
if __name__ == '__main__':
    while True:
        run_evaluation()
        # again = input("Run another evaluation? (y/n): ")
        # if again.lower() != 'y':
        #     break