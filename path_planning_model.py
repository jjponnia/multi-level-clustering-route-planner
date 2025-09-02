import torch
# import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions.categorical import Categorical


class PathPlanningModel(nn.Module):
    # num_targets includes the dummy target
    # num_actions includes the null action
    def __init__(self, obs_dim, n_targets, n_actions, hidden_sizes=128):
        super(PathPlanningModel, self).__init__()
        # Shared layers
        print(f"obs_dim: {obs_dim}")
        self.shared_layer1 = nn.Linear(obs_dim, hidden_sizes)
        self.shared_layer2 = nn.Linear(hidden_sizes, hidden_sizes)
        self.shared_layer3 = nn.Linear(hidden_sizes, hidden_sizes)

        # Output layers
        self.target_output_layer = nn.Linear(hidden_sizes, n_targets)  # For target softmax
        self.action_output_layer = nn.Linear(hidden_sizes, n_actions)  # For action softmax

        # self.output_layer = nn.Linear(hidden_sizes, n_acts)
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, obs, target_mask, action_mask):
        # Shared layers
        shared_out = F.relu(self.shared_layer1(obs))
        shared_out = F.relu(self.shared_layer2(shared_out))
        shared_out = F.relu(self.shared_layer3(shared_out))

        # Target output
        target_logits = self.target_output_layer(shared_out)
        masked_target_logits = target_logits.masked_fill(target_mask, -float('inf'))
        target_dist = Categorical(logits=masked_target_logits)

        # Action output
        action_logits = self.action_output_layer(shared_out)
        # print(f"action_mask (path_planning_model): {action_mask}")
        masked_action_logits = action_logits.masked_fill(action_mask, -float('inf'))
        action_dist = Categorical(logits=masked_action_logits)

        return target_dist, action_dist

        # out_layer1 = torch.relu(self.first_layer(obs))
        # out_layer2 = torch.relu(self.hidden_layer(out_layer1))
        # out_layer3 = self.output_layer(out_layer1)
        # print(f"mask: {mask.shape}")
        # masked_logits = out_layer3.masked_fill(mask, -float('inf'))

        # return Categorical(logits=out_layer3)
        # return Categorical(logits=masked_logits)