
import numpy as np
import torch
import matplotlib.pyplot as plt

# Import project-specific modules
import config
from src.agent import PPO, Memory
from src.environment import point_cloud, project_point, in_fov, is_occluded, env_setup
from src.utils import compute_reward_for_training

def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal (MPS) for training.")
    else:
        device = torch.device("cpu")
        print("MPS not available. Using CPU for training.")

    # --- PPO Agent ---
    state_vec_dim = 2 # (x, y) position
    action_dim = 2
    ppo_agent = PPO(state_vec_dim, action_dim, config.LR_ACTOR, config.LR_CRITIC, 
                    config.GAMMA, config.K_EPOCHS, config.EPS_CLIP, device, config.ACTION_STD_INIT)
    memory = Memory()

    # --- Logging ---
    all_rewards = []
    all_cam_positions = []
    all_mse_loss = []
    all_start_pos = []
    all_end_pos = []
    illegal_start_pos = []
    all_dist_to_roi = []
    all_dist_to_obs = []
    illegal_start_obs_pos = []
    
    timestep = 0

    print("Starting Training...")
    for episode in range(config.N_EPISODES):
        # Create a new random environment for each episode
        a = np.random.uniform(config.PCD_A_MIN, config.PCD_A_MAX)
        b = np.random.uniform(config.PCD_B_MIN, config.PCD_B_MAX)
        c = np.random.uniform(config.PCD_C_MIN, config.PCD_C_MAX)
        pcd3d_world = point_cloud(config.PCD_N_SAMPLES, a, b, c) + config.MU.reshape(3, 1)
        pcd3d_world_t = torch.from_numpy(pcd3d_world).float().to(device) #torch version
        
        pcd2d_world = pcd3d_world[:2, :].T
        pcd2d_world_t = torch.from_numpy(pcd2d_world).float().to(device) #torch version
        
        
        obstacle_positions_np, obstacle_radii_np, initial_camera_pos = env_setup()
        
        
        mu_t = torch.from_numpy(config.MU).float().to(device) #torch version
        obstacle_centers_t = torch.from_numpy(obstacle_positions_np).to(device)
        obstacle_radii_t = torch.from_numpy(obstacle_radii_np).to(device)
        
        all_start_pos.append(initial_camera_pos)
        all_indices = torch.arange(pcd2d_world_t.shape[0], device=device)

        
        state_vec = initial_camera_pos
        current_ep_reward = 0

        for t in range(config.MAX_TIMESTEPS):
            timestep += 1
            
            state_vec_t = torch.from_numpy(state_vec).float().to(device) #torch version
            cam_center_t = torch.cat((state_vec_t, torch.tensor([0.0], device=device))) #torch version

            # --- Camera Orientation ---
            
            dist_t = mu_t - cam_center_t
            dist_t[2] = 0 # Equivalent to the matrix multiplication
            norm_dist_t = torch.linalg.norm(dist_t)
            
            
            if norm_dist_t < 1e-6: continue


            zc_t = dist_t / norm_dist_t
            
            z_world = torch.tensor([0.0, 0.0, 1.0], device=device)
            
            
            # xc_i_cand = np.cross(zc_t, [0, 0, 1])
            
            xc_t = torch.linalg.cross(zc_t, z_world)
            
            norm_xc_t = torch.linalg.norm(xc_t)
            
            
            if norm_xc_t < 1e-6: continue

            xc_t = xc_t / norm_xc_t
            yc_t = torch.linalg.cross(zc_t, xc_t)
            R_t = torch.stack([xc_t, yc_t, zc_t])

            # --- Point Cloud Projection and Occlusion ---
            pc_t = R_t @ (pcd3d_world_t - cam_center_t.view(3, 1))
            gnt_point_cloud_px_t = project_point(pc_t.T, config.FX, config.FY)
            gnt_point_cloud_px_fov_t, _ = in_fov(gnt_point_cloud_px_t, config.IMAGE_W, config.IMAGE_H)

            # still_visible_indices = all_indices.copy()
            # for obs_center_2d, obs_radius, _ in obstacle_list:
            #     occluded_by_obs = is_occluded(pcd2d_world, cam_center_i[:2], obs_center_2d, obs_radius)
            #     still_visible_indices = np.setdiff1d(still_visible_indices, occluded_by_obs, assume_unique=True)
            
            # --- Point Cloud Projection and Occlusion (Optimized) ---
            visible_indices = all_indices.clone()

            # CORRECTED LOOP: Iterate over the NumPy arrays and create tensors inside
            for i in range(len(obstacle_positions_np)):
                obs_center_2d_t = torch.from_numpy(obstacle_positions_np[i]).to(device)
                obs_radius = obstacle_radii_np[i]

                # If no points are left, no need to check further.
                if visible_indices.numel() == 0:
                    break

                # Create a point cloud of only the currently visible points.
                currently_visible_pcd = pcd2d_world_t[visible_indices]

                # Run occlusion check on this smaller point cloud.
                newly_occluded_local_indices = is_occluded(
                    currently_visible_pcd, cam_center_t[:2], obs_center_2d_t, obs_radius
                )

                # If any points were occluded by this obstacle, update the list.
                if newly_occluded_local_indices.numel() > 0:
                    # Convert local indices from the small array back to the global indices.
                    newly_occluded_global_indices = visible_indices[newly_occluded_local_indices]
                    
                    # Create a boolean mask to filter out the occluded indices
                    # This is more efficient for PyTorch tensors than np.setdiff1d
                    mask = torch.ones_like(visible_indices, dtype=torch.bool)
                    all_occluded_indices = torch.cat([
                        torch.where(visible_indices == idx)[0] for idx in newly_occluded_global_indices
                    ])
                    mask[all_occluded_indices] = False
                    visible_indices = visible_indices[mask]

            observed_pc_camera_i = pc_t[:, visible_indices]
            observed_point_cloud_px_i = project_point(observed_pc_camera_i.T, config.FX, config.FY)
            observed_point_cloud_px_fov_i, _ = in_fov(observed_point_cloud_px_i, config.IMAGE_W, config.IMAGE_H)
            
           # --- State Creation ---
            # First, move the tensor to the CPU
            observed_cpu = observed_point_cloud_px_fov_i.cpu()
            H_obs, _, _ = np.histogram2d(
                observed_cpu[:, 0], observed_cpu[:, 1], 
                bins=config.BINS, range=config.HIST_RANGE
            )
            
            gnt_cpu = gnt_point_cloud_px_fov_t.cpu()
            H_gnt, _, _ = np.histogram2d(
                gnt_cpu[:, 0], gnt_cpu[:, 1], 
                bins=config.BINS, range=config.HIST_RANGE
            )
            
            state_img = torch.FloatTensor(H_obs.T).unsqueeze(0).unsqueeze(0).to(device)
            state_vec_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(device)

            # --- Agent Action ---
            action, log_prob = ppo_agent.act(state_img, state_vec_tensor)
            action_np = action.cpu().numpy().flatten()
            state_vec = state_vec + action_np * config.ACTION_SCALING
            
            state_vec_t = torch.from_numpy(state_vec).float().to(device) #torch version

            # --- Crash Detection and Reward ---
            crashed = False
            distances_to_obs = torch.linalg.norm(state_vec_t - obstacle_centers_t, axis=1)
            if torch.any(distances_to_obs < obstacle_radii_t):
                crashed = True
            else:
                crashed = False

            distance_to_roi = torch.linalg.norm(state_vec_t - mu_t[:2])
            distance_to_roi_np = distance_to_roi.item()
            reward, ratio = compute_reward_for_training(H_gnt.flatten(), H_obs.flatten(), distance_to_roi_np, config.DIST_MIN)
            
            if crashed:
                reward = -100
                print(f"Episode {episode+1} crashed at timestep {t+1}. Reward: {reward:.4f}")
                done = True
            else:
                done = t == config.MAX_TIMESTEPS - 1

            # --- Store and Update ---
            memory.states_img.append(state_img)
            memory.states_vec.append(state_vec_tensor)
            memory.actions.append(action)
            memory.logprobs.append(log_prob)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            if timestep % config.BATCH_UPDATE_TIMESTEP == 0:
                # Update the PPO agent
                print("Updating PPO agent... at timestep:", timestep)
                ppo_agent.update(memory)
                
                memory.clear()
                timestep = 0

            current_ep_reward += reward
            all_cam_positions.append(state_vec.copy())
            
            if done or crashed:
                all_end_pos.append(state_vec.copy())
                break

        all_rewards.append(current_ep_reward / config.MAX_TIMESTEPS)
        all_dist_to_roi.append(distance_to_roi)
        print(f"Episode {episode+1}/{config.N_EPISODES}: Avg Reward = {all_rewards[-1]:.4f}")

    print("Training finished.")
    torch.save(ppo_agent.policy.state_dict(), config.POLICY_PATH)
    print(f"Policy saved to {config.POLICY_PATH}")

    # --- Visualization ---
    plt.figure(figsize=(10, 5))
    plt.plot(all_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward per Episode")
    plt.title("Training Progress")
    plt.grid(True)
    # plt.show()
    
    # --- Visualization ---
    cam_centers = np.array(all_cam_positions)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ax1.plot(all_rewards, c='b', label="Average reward per episode")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Rewards")
    ax1.legend()
    ax1.grid(True)

    # Use a colormap to show the steps over time.
    num_steps_total = cam_centers.shape[0]
    step_indices = np.arange(num_steps_total)
    all_start_pos_np = np.array(all_start_pos)
    all_end_pos_np = np.array(all_end_pos)

    sc = ax2.scatter(cam_centers[:, 0], cam_centers[:, 1], c=step_indices, s=10, alpha=0.7, label='Camera centers')
    ax2.scatter(pcd3d_world[0,:], pcd3d_world[1,:], label='Point cloud (last ep.)')

    for i in range(len(obstacle_positions_np)):
        obs_pos_2d = obstacle_positions_np[i]
        obs_radius = obstacle_radii_np[i]
        ax2.add_patch(plt.Circle(obs_pos_2d, obs_radius, fill=False, color='r', linewidth=2))
        ax2.text(obs_pos_2d[0], obs_pos_2d[1], str(i), color='r', fontsize=12, ha='center', va='center')

    ax2.plot([], [], color='r', linewidth=2, label='Obstacles (last ep.)')[0].set_visible(False)
    ax2.scatter(config.MU[0], config.MU[1], c='b', marker='x', s=150, label='ROI center')
    ax2.scatter(all_start_pos_np[:,0], all_start_pos_np[:,1], c='g', label='Start')
    ax2.scatter(all_end_pos_np[:,0], all_end_pos_np[:,1], c='r', label="End")

    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title("Camera Trajectories Over Training")
    ax2.set_aspect('equal')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    cbar = plt.colorbar(sc, ax=ax2)
    cbar.set_label("Timestep")
    fig.savefig("trajectories_and_reward.png")  # Save the first figure

    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 7))
    ax3.plot(all_dist_to_obs)
    ax3.set_ylabel("Distance")
    ax3.set_xlabel("Episode")
    ax3.set_title("Final Distance to Nearest Obstacle")
    ax3.grid(True)

    ax4.plot(all_dist_to_roi)
    ax4.set_ylabel("Distance")
    ax4.set_xlabel("Episode")
    ax4.set_title("Final Distance to ROI")
    ax4.grid(True)

    plt.tight_layout()
    fig2.savefig("distances_over_episodes.png")  # Save the second figure

    plt.show()

if __name__ == "__main__":
    main()