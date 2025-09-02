import pickle
import numpy as np
import torch
from grid_world_env import gridworld_env
# from path_planning_model import PathPlanningModel
from attention_model.path_planning_attention_model import PathPlanningAttentionModel

from utils.run_utils import setup_logger_kwargs
from utils.logx import EpochLogger

import torch.nn.functional as F
from torch.optim import Adam

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger_kwargs = setup_logger_kwargs('agents-visit-targets', 0)
logger = EpochLogger(**logger_kwargs)

# Load the buffer
with open('final_buffer.pkl', 'rb') as f:
    buffer = pickle.load(f)

# Extract data from the buffer
obs_buf = buffer.obs_buf
action_mask_buf = buffer.action_mask_buf

print(f"action_mask_buf: {action_mask_buf.shape}")

target_mask_buf = buffer.target_mask_buf
action_buf = buffer.action_buf
route_buf = buffer.route_buf

# Define batch size
batch_size = 1024

# Shuffle the data
indices = np.arange(len(obs_buf))
np.random.shuffle(indices)

obs_buf = obs_buf[indices]
action_mask_buf = action_mask_buf[indices]
target_mask_buf = buffer.target_mask_buf
action_buf = action_buf[indices]
route_buf = route_buf[indices]

target_weight = 0.5
action_weight = 0.5

# Instantiate environment
env = gridworld_env()
num_targets = env.num_targets
obs_size = 2 * env.num_obstacles + 2 * env.num_agents + 2 * env.num_targets + 1 + env.num_agents

hidden_sizes = 1024
path_model = PathPlanningAttentionModel(obs_dim=obs_size, n_targets=env.num_targets + 1, n_actions=5).to(device)
# baseline_model = PathPlanningModel(obs_dim=obs_dim, n_acts=4).to(device)

pi_lr=1e-4
path_model_optimizer = Adam(path_model.parameters(), lr=pi_lr)

# Iterate through the data in batches
for i in range(0, len(obs_buf), batch_size):
    print(f"i: {i}")
    obs_batch = torch.as_tensor(obs_buf[i:i + batch_size], dtype=torch.float32).to(device)
    action_mask_batch = torch.as_tensor(action_mask_buf[i:i + batch_size], dtype=torch.bool).to(device)
    target_mask_batch = torch.as_tensor(target_mask_buf[i:i + batch_size], dtype=torch.bool).to(device)
    action_batch = torch.as_tensor(action_buf[i:i + batch_size], dtype=torch.long).to(device)
    route_batch = torch.as_tensor(route_buf[i:i + batch_size], dtype=torch.long).to(device)

    # Forward pass through the policy network

    target_dist, action_dist = path_model(obs_batch, target_mask_batch, action_mask_batch)

    # Compute loss and update the network
    target_loss = F.cross_entropy(target_dist.probs, route_batch)
    action_loss = F.cross_entropy(action_dist.probs, action_batch)
    imitation_loss = target_weight * target_loss + action_weight * action_loss
    logger.store(Loss=imitation_loss.item())

    path_model_optimizer.zero_grad()
    imitation_loss.backward()
    torch.nn.utils.clip_grad_norm_(path_model.parameters(), 0.5)
    path_model_optimizer.step()

    # Log the loss
    logger.store(Loss=imitation_loss.item())

    logger.log_tabular('Epoch', i)
    # logger.log_tabular('EpRet', average_only=True)
    # logger.log_tabular('OptPathLength', average_only=True)
    logger.log_tabular('Loss', average_only=True)
    logger.dump_tabular()