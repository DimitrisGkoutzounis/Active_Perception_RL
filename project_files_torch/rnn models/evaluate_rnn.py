# Add this import to the top of your evaluation script
from src.models_rnn import ActorCriticRNN
import numpy as np
import torch
import matplotlib.pyplot as plt

# Import project-specific modules
import config
from src.models import ActorCritic
from src.environment import point_cloud, project_point, in_fov, is_occluded, env_setup
from src.utils import compute_reward, compute_reward_for_training
from src.visualization import show_map_multiple_obstacles, plot_run_diagnostics, plot_frame_comparison, generate_reward_map

def run_evaluation_rnn():
    """
    Runs evaluation for the STATEFUL ActorCriticRNN model.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device} for RNN inference.")
   
    # --- Environment Setup (same as before) ---
    obstacle_list, _ = env_setup()
    
    a = np.random.uniform(config.PCD_A_MIN, config.PCD_A_MAX)
    b = np.random.uniform(config.PCD_B_MIN, config.PCD_B_MAX)
    c = np.random.uniform(config.PCD_C_MIN, config.PCD_C_MAX)
    pcd3d_world = point_cloud(config.PCD_N_SAMPLES, a, b, c) + config.MU.reshape(3, 1)

    # --- Generate Reward Map (optional, same as before) ---
    reward_map, reward_map_extent = generate_reward_map(
        grid_size=config.EVAL_GRID_SIZE, bounds=config.EVAL_MAP_BOUNDS, 
        pcd3d_world=pcd3d_world, obstacle_list=obstacle_list, mu=config.MU, 
        fx=config.FX, fy=config.FY, image_w=config.IMAGE_W, image_h=config.IMAGE_H, 
        BINS=config.BINS, hist_range=config.HIST_RANGE, DIST_MIN=config.DIST_MIN
    )
    clicked_point = show_map_multiple_obstacles(
        obstacle_list=obstacle_list, mu=config.MU, pcd3d=pcd3d_world, d_min=config.DIST_MIN,
        trajectory=None, reward_map=reward_map, reward_map_extent=reward_map_extent
    )
    
    if not clicked_point:
        print("No point clicked. Exiting evaluation.")
        return

    # --- 1. Load the RNN Policy and Initialize ---
    state_vec = np.array(clicked_point)
    rnn_policy_path = config.POLICY_PATH.replace('.pth', '_rnn.pth')
    print(f"Loading RNN policy from: {rnn_policy_path}")
    
    # Instantiate the ActorCriticRNN model
    policy = ActorCriticRNN(state_vec_dim=2, action_dim=2, rnn_hidden_size=64).to(device)
    policy.load_state_dict(torch.load(rnn_policy_path, map_location=device))
    policy.eval()

    # --- 2. Initialize the Hidden State ---
    # Create the initial "memory" for the RNN before the episode starts
    hidden_state = ActorCriticRNN.init_hidden(batch_size=1, rnn_hidden_size=64, n_layers=1, device=device)

    trajectory = [state_vec.copy()]
    rewards_over_time = []
    initial_frame_data = {}
    
    # --- Run Simulation Loop ---
    for step in range(config.EVAL_N_STEPS):
        # The environment simulation part remains exactly the same
        cam_center_i = np.append(state_vec, 0.0)
        
        # ... (all the code for camera orientation, projection, occlusion, etc. remains unchanged)
        dist_i=np.array([[1,0,0],[0,1,0],[0,0,0]])@(config.MU-cam_center_i);norm_dist_i=np.linalg.norm(dist_i)
        if norm_dist_i < 1e-6: continue
        zc_i=dist_i/norm_dist_i;xc_i_cand=np.cross(zc_i,[0,0,1]);norm_xc_i=np.linalg.norm(xc_i_cand)
        if norm_xc_i < 1e-6: continue
        xc_i=xc_i_cand/norm_xc_i;yc_i=np.cross(zc_i,xc_i);R_i=np.vstack([xc_i,yc_i,zc_i])
        pc_i=R_i@(pcd3d_world-cam_center_i.reshape(3,1))
        gnt_point_cloud_px_i=project_point(pc_i.T,config.FX,config.FY)
        gnt_point_cloud_px_fov_i,_=in_fov(gnt_point_cloud_px_i,config.IMAGE_W,config.IMAGE_H)
        pcd2d_world=pcd3d_world[:2,:].T;all_indices=np.arange(pcd2d_world.shape[0])
        still_visible_indices=all_indices.copy()
        for obs_center_2d,obs_radius,_ in obstacle_list:
            occluded_by_obs=is_occluded(pcd2d_world,cam_center_i[:2],obs_center_2d,obs_radius)
            still_visible_indices=np.setdiff1d(still_visible_indices,occluded_by_obs,assume_unique=True)
        observed_pc_camera_i=pc_i[:,still_visible_indices]
        observed_point_cloud_px_i=project_point(observed_pc_camera_i.T,config.FX,config.FY)
        observed_point_cloud_px_fov_i,_=in_fov(observed_point_cloud_px_i,config.IMAGE_W,config.IMAGE_H)
        H_obs,_,_=np.histogram2d(observed_point_cloud_px_fov_i[:,0],observed_point_cloud_px_fov_i[:,1],bins=config.BINS,range=config.HIST_RANGE)
        H_gnt,_,_=np.histogram2d(gnt_point_cloud_px_fov_i[:,0],gnt_point_cloud_px_fov_i[:,1],bins=config.BINS,range=config.HIST_RANGE)
        distance_to_roi=np.linalg.norm(state_vec-config.MU[:2])
        min_dist_obs=float('inf')
        for obs_center_2d,obs_radius,_ in obstacle_list:
            distance_to_obs=np.linalg.norm(cam_center_i[:2]-obs_center_2d)
            if distance_to_obs<min_dist_obs: min_dist_obs=distance_to_obs
        distance_to_obs=min_dist_obs if np.isfinite(min_dist_obs)else 0.0
        reward,ratio=compute_reward_for_training(H_gnt.flatten(),H_obs.flatten(),distance_to_roi,config.DIST_MIN,distance_to_obs)
        rewards_over_time.append(reward)
        print(f"Step: {step+1}/{config.EVAL_N_STEPS} | Position: [{state_vec[0]:.2f}, {state_vec[1]:.2f}] | Reward: {reward:.4f}")

        # Agent Action
        state_img_tensor = torch.FloatTensor(H_obs.T).unsqueeze(0).unsqueeze(0).to(device)
        state_vec_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(device)

        with torch.no_grad():
            # --- 3. Pass and Update the Hidden State ---
            # Add a sequence dimension of 1 for single-step evaluation
            state_img_seq = state_img_tensor.unsqueeze(1)
            state_vec_seq = state_vec_tensor.unsqueeze(1)
            
            # The model now takes the hidden_state as input and returns the new one
            action_mean, _, hidden_state = policy.forward(state_img_seq, state_vec_seq, hidden_state)
            
            # Remove the sequence dimension from the action
            action_tensor = action_mean.squeeze(1)

        action_np = action_tensor.cpu().numpy().flatten()
        state_vec = state_vec + action_np * config.ACTION_SCALING
        trajectory.append(state_vec.copy())

    # --- Visualization (same as before) ---
    # ... (all your plotting functions remain unchanged)
    show_map_multiple_obstacles(
        obstacle_list=obstacle_list, mu=config.MU, pcd3d=pcd3d_world, d_min=config.DIST_MIN,
        trajectory=np.array(trajectory), reward_map=reward_map, reward_map_extent=reward_map_extent
    )

# In your main execution block, call the new function
if __name__ == '__main__':
    while True:
        # run_evaluation()       # Call this for the original stateless agent
        run_evaluation_rnn()   # Call this for the new stateful RNN agent
        