
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import config
from src.agent import PPO, Memory
from project_files.src.agent_rnn import PPO_RNN, MemoryRNN
from src.environment import point_cloud, project_point, in_fov, is_occluded, env_setup
from src.utils import compute_reward_for_training

def main_rnn():
    """Main function to train the PPO agent with an RNN policy."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal (MPS) for RNN training.")
    else:
        device = torch.device("cpu")
        print("MPS not available. Using CPU for RNN training.")

    writer = SummaryWriter(log_dir="runs/PPO_RNN_Experiment")
    
    # --- PPO Agent with RNN ---
    state_vec_dim = 2
    action_dim = 2
    rnn_hidden_size = 64
    rnn_n_layers = 1
    
    ppo_agent = PPO_RNN(
        state_vec_dim, action_dim, config.LR_ACTOR, config.LR_CRITIC, 
        config.GAMMA, config.K_EPOCHS, config.EPS_CLIP, device, 
        config.ACTION_STD_INIT, config.ACTION_STD_DECAY_RATE, config.MIN_ACTION_STD,
        rnn_hidden_size, rnn_n_layers  
    )
    memory = MemoryRNN()

    # --- Logging ---
    all_rewards = []
    update_step_count = 0

    print("Starting Training with RNN Agent...")
    for episode in range(config.N_EPISODES):
        # --- Environment Setup (same as before) ---
        a = np.random.uniform(config.PCD_A_MIN, config.PCD_A_MAX)
        b = np.random.uniform(config.PCD_B_MIN, config.PCD_B_MAX)
        c = np.random.uniform(config.PCD_C_MIN, config.PCD_C_MAX)
        pcd3d_world = point_cloud(config.PCD_N_SAMPLES, a, b, c) + config.MU.reshape(3, 1)
        pcd2d_world = pcd3d_world[:2, :].T
        all_indices = np.arange(pcd2d_world.shape[0])
        obstacle_list, initial_camera_pos = env_setup()
        
        state_vec = initial_camera_pos
        current_ep_reward = 0
        
        hidden_state = ppo_agent.init_hidden()

        for t in range(config.MAX_TIMESTEPS):
            # --- State Creation (same as before) ---
            cam_center_i = np.append(state_vec, 0.0)
            dist_i = np.array([[1,0,0],[0,1,0],[0,0,0]]) @ (config.MU - cam_center_i)
            norm_dist_i = np.linalg.norm(dist_i);
            if norm_dist_i < 1e-6: continue
            zc_i = dist_i / norm_dist_i
            xc_i_cand = np.cross(zc_i, [0, 0, 1]); norm_xc_i = np.linalg.norm(xc_i_cand)
            if norm_xc_i < 1e-6: continue
            xc_i = xc_i_cand / norm_xc_i; yc_i = np.cross(zc_i, xc_i); R_i = np.vstack([xc_i, yc_i, zc_i])
            pc_i = R_i @ (pcd3d_world - cam_center_i.reshape(3, 1))
            gnt_point_cloud_px_i = project_point(pc_i.T, config.FX, config.FY)
            gnt_point_cloud_px_fov_i, _ = in_fov(gnt_point_cloud_px_i, config.IMAGE_W, config.IMAGE_H)
            visible_indices = all_indices.copy()
            for obs_center_2d, obs_radius, _ in obstacle_list:
                if visible_indices.size == 0: break
                currently_visible_pcd = pcd2d_world[visible_indices]
                newly_occluded_local_indices = is_occluded(currently_visible_pcd, cam_center_i[:2], obs_center_2d, obs_radius)
                if newly_occluded_local_indices.size > 0:
                    newly_occluded_global_indices = visible_indices[newly_occluded_local_indices]
                    visible_indices = np.setdiff1d(visible_indices, newly_occluded_global_indices, assume_unique=True)
            observed_pc_camera_i = pc_i[:, visible_indices]
            observed_point_cloud_px_i = project_point(observed_pc_camera_i.T, config.FX, config.FY)
            observed_point_cloud_px_fov_i, _ = in_fov(observed_point_cloud_px_i, config.IMAGE_W, config.IMAGE_H)
            H_obs, _, _ = np.histogram2d(observed_point_cloud_px_fov_i[:, 0], observed_point_cloud_px_fov_i[:, 1], bins=config.BINS, range=config.HIST_RANGE)
            H_gnt, _, _ = np.histogram2d(gnt_point_cloud_px_fov_i[:, 0], gnt_point_cloud_px_fov_i[:, 1], bins=config.BINS, range=config.HIST_RANGE)
            state_img = torch.FloatTensor(H_obs.T).unsqueeze(0).unsqueeze(0).to(device)
            state_vec_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(device)

            h_in, c_in = hidden_state[0].detach(), hidden_state[1].detach()
            action, log_prob, hidden_state = ppo_agent.act(state_img, state_vec_tensor, hidden_state)
            
            state_vec = state_vec + action.cpu().numpy().flatten() * config.ACTION_SCALING

            crashed = False; distance_to_obs = float('inf')
            for obs_center_2d, obs_radius, _ in obstacle_list:
                current_dist_to_obs = np.linalg.norm(state_vec - obs_center_2d)
                distance_to_obs = min(distance_to_obs, current_dist_to_obs)
                if current_dist_to_obs < obs_radius: crashed = True; break
            distance_to_roi = np.linalg.norm(state_vec - config.MU[:2])
            
            reward, _ = compute_reward_for_training(H_gnt.flatten(), H_obs.flatten(), distance_to_roi, config.DIST_MIN, distance_to_obs)
            
            
            
            if crashed:
                reward = -3000
                print(f"Episode {episode+1} crashed at timestep {t+1}. Reward: {reward:.4f}")
                done = True
            else:
                done = t == config.MAX_TIMESTEPS - 1


            memory.store_timestep(action, state_img, state_vec_tensor, log_prob, reward, done, h_in, c_in)
            
            current_ep_reward += reward
            if done: break
        
        memory.finish_trajectory()
        avg_episode_reward = current_ep_reward / (t + 1)
        all_rewards.append(avg_episode_reward)
        writer.add_scalar('Reward/Average_Episode_Reward_RNN', avg_episode_reward, episode)
        writer.add_scalar('Metrics/Final_Distance_to_ROI_RNN', distance_to_roi, episode)
        print(f"Episode {episode+1}/{config.N_EPISODES}: Avg Reward = {all_rewards[-1]:.4f}, Total Timesteps in Mem: {len(memory)}")

        # --- Update Policy ---
        if len(memory) >= config.BATCH_UPDATE_TIMESTEP:
            print(f"Updating PPO-RNN agent... at update step: {update_step_count}")
            p_loss, v_loss, entropy, avg_value = ppo_agent.update(memory)
            ppo_agent.decay_action_std()
            writer.add_scalar('Loss/Policy_Loss_RNN', p_loss, update_step_count)
            writer.add_scalar('Loss/Value_Loss_RNN', v_loss, update_step_count)
            writer.add_scalar('Metrics/Policy_Entropy_RNN', entropy, update_step_count)
            update_step_count += 1
            memory.clear()

    print("Training finished.")
    rnn_policy_path = config.POLICY_PATH.replace('.pth', '_rnn.pth')
    torch.save(ppo_agent.policy.state_dict(), rnn_policy_path)
    print(f"RNN Policy saved to {rnn_policy_path}")
    writer.close()
    
    # --- Visualization (add your plots here if needed) ---
    plt.figure(figsize=(10, 5))
    plt.plot(all_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward per Episode")
    plt.title("RNN Training Progress")
    plt.grid(True)
    plt.savefig("rnn_training_rewards.png")
    plt.show()
    
    