
import numpy as np

# --- Scene and Camera Parameters ---
FX = 533.895
FY = 534.07
IMAGE_W = 1280
IMAGE_H = 720
MU = np.array([0.0, 0.0, 0.0]) # ROI center
DIST_MIN = 7 # Minimum allowed distance to ROI center for the agent

# --- Environment Generation ---
# Point Cloud parameters (randomized range)
PCD_A_MIN, PCD_A_MAX = 1.0, 3.5
PCD_B_MIN, PCD_B_MAX = 1.0, 3.5
PCD_C_MIN, PCD_C_MAX = 1.0, 3.5
PCD_N_SAMPLES = 500

# Obstacle generation parameters
NUM_OBSTACLES = 8
OBSTACLE_MIN_DIST_FROM_ROI = 15#25#15
OBSTACLE_MAX_DIST_FROM_ROI = 40#130#40
OBSTACLE_MIN_RADIUS = 2.0#1.0 #2
OBSTACLE_MAX_RADIUS = 5.0 #5

# Agent starting position generation parameters
AGENT_MIN_START_DIST_FROM_ROI = 50#70
AGENT_MAX_START_DIST_FROM_ROI = 80#150 #150, reward 120
ENV_BOUNDS_X = [-100, 100]
ENV_BOUNDS_Y = [-100, 100]


# --- State Representation ---
BINS = 30
HIST_RANGE = [[0, IMAGE_W], [0, IMAGE_H]]


# --- PPO Agent Hyperparameters (for training) ---
POLICY_PATH = "saved_models/ppo_policy_camera_cnn_whole_circle.pth"
N_EPISODES = 6000
MAX_TIMESTEPS = 50
BATCH_UPDATE_TIMESTEP = 2048#1050
LR_ACTOR = 0.0001
LR_CRITIC = 0.001
GAMMA = 0.99
K_EPOCHS = 3
EPS_CLIP = 0.2
ACTION_STD_INIT = 0.2
ACTION_SCALING = 2.0

ACTION_STD_INIT = 0.6  #             
ACTION_STD_DECAY_RATE = 0.005      
MIN_ACTION_STD = 0.1    

# --- Evaluation Parameters ---
EVAL_N_STEPS = 50
EVAL_GRID_SIZE = 25 # For reward map generation
EVAL_MAP_BOUNDS = [ENV_BOUNDS_X[0], ENV_BOUNDS_X[1], ENV_BOUNDS_Y[0], ENV_BOUNDS_Y[1]]