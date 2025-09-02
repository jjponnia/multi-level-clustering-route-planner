import torch
# import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions.categorical import Categorical
from attention_model.entity_encoder import EntityEncoder
from attention_model.cross_attention_decoder import CrossAttentionDecoder
from attention_model.config import config
cfg = config()
# from cross_attention_decoder import CrossAttentionDecoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PathPlanningAttentionModel(nn.Module):
    def __init__(self, n_targets, n_actions):
        super(PathPlanningAttentionModel, self).__init__()

        self.encoder = EntityEncoder(cfg)
        self.decoder = CrossAttentionDecoder(n_targets, n_actions, cfg)

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, ego_agent_input, other_agents_input, targets_input, obstacles_input, action_mask, target_mask):
        # Shared layers

        ego_agent_latent_vector, other_agents_latent_vectors, obstacles_latent_vectors, targets_latent_vectors = self.encoder(
            ego_agent_input, other_agents_input, obstacles_input, targets_input
        )

        target_logits, action_logits = self.decoder(ego_agent_latent_vector,
                                                    other_agents_latent_vectors,
                                                    obstacles_latent_vectors,
                                                    targets_latent_vectors)

        # print(f"target_logits: {target_logits.shape}")
        # print(f"action_logits: {action_logits.shape}")

        masked_target_logits = target_logits.masked_fill(target_mask, -float(1e9))
        threshold = 1e-6  # Define a threshold for "close to zero"
        close_to_zero_indices = torch.isclose(target_logits, torch.tensor(0.0, device=device), atol=threshold).nonzero(as_tuple=True)[0]

        if close_to_zero_indices.numel() > 0:
            print(f"Batch indices with target_logits close to zero: {close_to_zero_indices}")

        # target_dist = Categorical(logits=masked_target_logits)

        masked_action_logits = action_logits.masked_fill(action_mask, -float(1e9))
        # action_dist = Categorical(logits=masked_action_logits)

        # shared_out = F.relu(self.shared_layer1(obs))
        # shared_out = F.relu(self.shared_layer2(shared_out))

        # Target output
        # target_logits = self.target_output_layer(shared_out)
        # masked_target_logits = target_logits.masked_fill(target_mask, -float('inf'))
        # target_dist = Categorical(logits=masked_target_logits)

        # Action output
        # action_logits = self.action_output_layer(shared_out)
        # print(f"action_mask (path_planning_model): {action_mask}")
        # masked_action_logits = action_logits.masked_fill(action_mask, -float('inf'))
        # action_dist = Categorical(logits=masked_action_logits)

        return masked_target_logits, masked_action_logits

        # out_layer1 = torch.relu(self.first_layer(obs))
        # out_layer2 = torch.relu(self.hidden_layer(out_layer1))
        # out_layer3 = self.output_layer(out_layer1)
        # print(f"mask: {mask.shape}")
        # masked_logits = out_layer3.masked_fill(mask, -float('inf'))

        # return Categorical(logits=out_layer3)
        # return Categorical(logits=masked_logits)