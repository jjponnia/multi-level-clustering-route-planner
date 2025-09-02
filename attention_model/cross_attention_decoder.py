import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.distributions.categorical import Categorical

# from config import config
# from agentEncoder import AgentEncoder
# from targetEncoder import TargetEncoder

# cfg = config()


class SingleHeadAttention(nn.Module):
    def __init__(self, cfg):
        super(SingleHeadAttention, self).__init__()
        self.input_dim = cfg.embedding_dim
        self.embedding_dim = cfg.embedding_dim
        self.value_dim = self.embedding_dim
        self.key_dim = self.value_dim
        self.tanh_clipping = cfg.tanh_clipping
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
                :param q: queries (batch_size, n_query, input_dim)
                :param h: data (batch_size, graph_size, input_dim)
                :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
                Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
                :return:
                """
        if h is None:
            h = q

        batch_size, target_size, input_dim = h.size()
        n_query = q.size(1)  # n_query = target_size in tsp

        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim

        h_flat = h.reshape(-1, input_dim)  # (batch_size*graph_size)*input_dim
        q_flat = q.reshape(-1, input_dim)  # (batch_size*n_query)*input_dim

        shape_k = (batch_size, target_size, -1)
        shape_q = (batch_size, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)  # batch_size*n_query*key_dim
        K = torch.matmul(h_flat, self.w_key).view(shape_k)  # batch_size*targets_size*key_dim

        U = self.norm_factor * torch.matmul(Q, K.transpose(1, 2))  # batch_size*n_query*targets_size
        U = self.tanh_clipping * torch.tanh(U)

        if mask is not None:
            mask = mask.view(batch_size, 1, target_size).expand_as(U)  # copy for n_heads times
            # U = U-1e8*mask  # ??
            U[mask.bool()] = -1e8

        attention = torch.log_softmax(U, dim=-1)  # batch_size*n_query*targets_size

        out = attention

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg, n_heads=16):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.query_dim = cfg.embedding_dim  # step_context_size
        self.input_dim = cfg.embedding_dim
        self.embedding_dim = cfg.embedding_dim
        self.value_dim = self.embedding_dim // self.n_heads
        self.key_dim = self.value_dim
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.n_heads, self.query_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_value = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.value_dim))
        self.w_out = nn.Parameter(torch.Tensor(self.n_heads, self.value_dim, self.embedding_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
                :param q: queries (batch_size, n_query, input_dim)
                :param h: data (batch_size, graph_size, input_dim)
                :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
                Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
                :return:
                """
        if h is None:
            h = q

        batch_size, target_size, input_dim = h.size()
        n_query = q.size(1)  # n_query = target_size in tsp
        assert q.size(0) == batch_size
        # assert q.size(2) == input_dim
        assert input_dim == self.input_dim

        h_flat = h.contiguous().view(-1, input_dim)  # (batch_size*graph_size)*input_dim
        q_flat = q.contiguous().view(-1, self.query_dim)  # (batch_size*n_query)*input_dim
        shape_v = (self.n_heads, batch_size, target_size, -1)
        shape_k = (self.n_heads, batch_size, target_size, -1)
        shape_q = (self.n_heads, batch_size, n_query, -1)
        Q = torch.matmul(q_flat, self.w_query).view(shape_q)  # n_heads*batch_size*n_query*key_dim
        K = torch.matmul(h_flat, self.w_key).view(shape_k)  # n_heads*batch_size*targets_size*key_dim
        V = torch.matmul(h_flat, self.w_value).view(shape_v)  # n_heads*batch_size*targets_size*value_dim
        U = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))  # n_heads*batch_size*n_query*targets_size

        if mask is not None:
            mask = mask.view(1, batch_size, 1, target_size).expand_as(U)  # copy for n_heads times
            U[mask.bool()] = -np.inf

        attention = torch.softmax(U, dim=-1)  # n_heads*batch_size*n_query*targets_size

        if mask is not None:
            attnc = attention.clone()
            attnc[mask.bool()] = 0
            attention = attnc

        heads = torch.matmul(attention, V)  # n_heads*batch_size*n_query*value_dim
        out = torch.mm(
            heads.permute(1, 2, 0, 3).reshape(-1, self.n_heads * self.value_dim),
            # batch_size*n_query*n_heads*value_dim
            self.w_out.view(-1, self.embedding_dim)
            # n_heads*value_dim*embedding_dim
        ).view(batch_size, n_query, self.embedding_dim)

        return out  # batch_size*n_query*embedding_dim


class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, inputs):
        return inputs + self.module(inputs)


class Normalization(nn.Module):
    def __init__(self, cfg):
        super(Normalization, self).__init__()
        self.normalizer = nn.LayerNorm(cfg.embedding_dim)

    def forward(self, input):
        return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())


class AttentionLayer(nn.Module):
    # For not self attention
    def __init__(self, cfg):
        super(AttentionLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(cfg)
        self.normalization1 = Normalization(cfg)
        self.feedForward = nn.Sequential(nn.Linear(cfg.embedding_dim, 512),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(512, cfg.embedding_dim))
        self.normalization2 = Normalization(cfg)

    def forward(self, q, h=None, mask=None):
        if h is None:
            h = q

        h0 = q
        h = self.multiHeadAttention(q=q, h=h,mask=mask)
        h = h+h0
        h=self.normalization1(h)
        h1=h
        h = self.feedForward(h)
        h2 = h+h1
        h=self.normalization2(h2)
        return h


class CrossAttentionDecoder(nn.Module):
    def __init__(self, n_targets, n_actions, cfg):
        super(CrossAttentionDecoder, self).__init__()

        self.ego_to_others = AttentionLayer(cfg)
        self.self_attention_round_one = AttentionLayer(cfg)
        self.ego_to_obstacles = AttentionLayer(cfg)
        self.self_attention_round_two = AttentionLayer(cfg)
        self.ego_to_targets = AttentionLayer(cfg)
        self.self_attention_round_three = AttentionLayer(cfg)
        self.shared_layer_one = nn.Linear(128, 2048)
        self.shared_layer_one_norm = nn.LayerNorm(2048)
        self.shared_layer_two = nn.Linear(2048, 2048)
        self.shared_layer_two_norm = nn.LayerNorm(2048)
        self.target_logits_one = nn.Linear(2048, 2048)
        self.target_logits_one_norm = nn.LayerNorm(2048)
        self.target_logits_two = nn.Linear(2048, n_targets)
        self.action_logits_one = nn.Linear(2048, 2048)
        self.action_logits_one_norm = nn.LayerNorm(2048)
        self.action_logits_two = nn.Linear(2048, n_actions)


    def forward(self, ego_agent_latent_vector, other_agents_latent_vector, obstacles_latent_vector, targets_latent_vector):
        ego_agent_latent_vector = self.ego_to_others(q=ego_agent_latent_vector, h=other_agents_latent_vector)
        ego_agent_latent_vector = self.self_attention_round_one(ego_agent_latent_vector)
        ego_agent_latent_vector = self.ego_to_obstacles(q=ego_agent_latent_vector, h=obstacles_latent_vector)
        ego_agent_latent_vector = self.self_attention_round_two(ego_agent_latent_vector)
        ego_agent_latent_vector = self.ego_to_targets(q=ego_agent_latent_vector, h = targets_latent_vector)
        ego_agent_latent_vector = self.self_attention_round_three(ego_agent_latent_vector)

        ego_agent_latent_vector = ego_agent_latent_vector.squeeze(1)  # (batch, 1, embedding_dim)

        shared_layer_one = self.shared_layer_one(ego_agent_latent_vector)
        activation_layer_one = F.relu(shared_layer_one)
        activation_layer_one = self.shared_layer_one_norm(activation_layer_one)
        shared_layer_two = self.shared_layer_two(activation_layer_one)
        activation_layer_two = F.relu(shared_layer_two)
        activation_layer_two = self.shared_layer_two_norm(activation_layer_two)
        target_logits_one = self.target_logits_one(activation_layer_two)
        target_activation_one = F.relu(target_logits_one)
        target_activation_one = self.target_logits_one_norm(target_activation_one)
        target_logits_two = self.target_logits_two(target_activation_one)
        action_activation_one = self.action_logits_one(activation_layer_one)
        action_activation_one = F.relu(action_activation_one)
        action_logits_two = self.action_logits_two(action_activation_one)

        return target_logits_two, action_logits_two


    def get_score_function(self, _log_p, pi):
        """	args:
            _log_p: (batch, city_t, city_t)
            pi: (batch, city_t), predicted tour
            return: (batch) sum of the log probability of the chosen targets
        """
        log_p = torch.gather(input=_log_p, dim=2, index=pi[:, :, None])
        return torch.sum(log_p.squeeze(-1), 1)

    def sum_distance(self, inputs, route):
        d = torch.gather(input=inputs, dim=1, index=route[:, :, None].repeat(1, 1, 2))
        return (torch.sum((d[:, 1:] - d[:, :-1]).norm(p=2, dim=2), dim=1)
                + (d[:, 0] - d[:, -1]).norm(p=2, dim=1))  # distance from last node to first selected node)


