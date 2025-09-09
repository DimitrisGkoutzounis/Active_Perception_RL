# Active_Perception_RL


This project uses a Proximal Policy Optimization (PPO) agent to determine the optimal placement of a camera for viewing a 3D point cloud (Simulated the behavior of a single fire source) while avoiding occlusions from obstacles.

## Project Structure

- `src/`: Contains all core Python modules.
  - `agent.py`: PPO agent and memory buffer.
  - `environment.py`: Simulation logic (point clouds, projection, occlusion).
  - `models.py`: PyTorch Actor-Critic model definition.
  - `utils.py`: Reward functions and metrics.
  - `visualization.py`: Plotting and visualization functions.
- `saved_models/`: Directory to store trained model weights (`.pth` files).
- `config.py`: Central file for all hyperparameters and constants.
- `train.py`: Script to train the agent.
- `evaluate.py`: Script to evaluate a trained agent and generate visuals.
- `requirements.txt`: Python dependencies.

## Setup

1.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training

To train a new agent, run the `train.py` script. The trained model will be saved in the `saved_models/` directory, as specified in `config.py`.

```bash
python train.py