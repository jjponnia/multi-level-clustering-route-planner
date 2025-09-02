import torch
import numpy as np
import torch.nn as nn
import math
import torchsummary
# from config import config

# cfg = config()


class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, inputs):
        return inputs + self.module(inputs)


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg, n_heads=16):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = cfg.embedding_dim
        self.embedding_dim = cfg.embedding_dim
        self.value_dim = self.embedding_dim // self.n_heads
        self.key_dim = self.value_dim
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
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
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim

        h_flat = h.contiguous().view(-1, input_dim)  # (batch_size*graph_size)*input_dim
        q_flat = q.contiguous().view(-1, input_dim)  # (batch_size*n_query)*input_dim
        shape_v = (self.n_heads, batch_size, target_size, -1)
        shape_k = (self.n_heads, batch_size, target_size, -1)
        shape_q = (self.n_heads, batch_size, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)  # n_heads*batch_size*n_query*key_dim
        K = torch.matmul(h_flat, self.w_key).view(shape_k)  # n_heads*batch_size*targets_size*key_dim
        V = torch.matmul(h_flat, self.w_value).view(shape_v)  # n_heads*batch_size*targets_size*value_dim

        U = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))  # n_heads*batch_size*n_query*targets_size

        if mask is not None:
            mask = mask.view(1, batch_size, n_query, target_size).expand_as(U)  # copy for n_heads times
            U[mask] = -np.inf  # ??

        attention = torch.softmax(U, dim=-1)  # n_heads*batch_size*n_query*targets_size

        if mask is not None:
            attnc = attention.clone()
            attnc[mask] = 0
            attention = attnc

        heads = torch.matmul(attention, V)  # n_heads*batch_size*n_query*value_dim

        out = torch.mm(
            heads.permute(1, 2, 0, 3).reshape(-1, self.n_heads * self.value_dim),
            # batch_size*n_query*n_heads*value_dim
            self.w_out.view(-1, self.embedding_dim)
            # n_heads*value_dim*embedding_dim
        ).view(batch_size, n_query, self.embedding_dim)

        return out  # batch_size*n_query*embedding_dim


class Normalization(nn.Module):
    def __init__(self, cfg):
        super(Normalization, self).__init__()
        self.normalizer = nn.LayerNorm(cfg.embedding_dim, elementwise_affine=True)

    def forward(self, input):
        return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, cfg):
        super(MultiHeadAttentionLayer, self).__init__()
        self.layer = nn.Sequential(
            SkipConnection(nn.Sequential(Normalization(cfg=cfg), MultiHeadAttention(cfg))),
            SkipConnection(nn.Sequential(Normalization(cfg=cfg), nn.Linear(cfg.embedding_dim, 512),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(512, cfg.embedding_dim)))
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class EntityEncoder(nn.Module):
    def __init__(self, cfg):
        super(EntityEncoder, self).__init__()
        self.ego_agent_embedder = nn.Linear(2, cfg.embedding_dim)  # ego agent is also an other agent
        self.other_agents_embedder = nn.Linear(2, cfg.embedding_dim)
        self.obstacles_embedder = nn.Linear(8, cfg.embedding_dim)
        self.targets_embedder = nn.Linear(2, cfg.embedding_dim)

        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(cfg=cfg)
            for _ in range(3)
        ))  # * can capture the elements in the iteration

    def forward(self, ego_agent_input, other_agents_input, obstacles_input, targets_input, mask=None):
        assert mask is None, "mask is None"
        # print(f"targets_input: {targets_input.shape}")
        # print(f"obstacles_input: {obstacles_input.shape}")
        # print(f"other_agents_input: {other_agents_input.shape}")
        # print(f"ego_agent_input: {ego_agent_input.shape}")

        ego_agent_embedding = self.ego_agent_embedder(ego_agent_input)
        other_agents_embedding = self.other_agents_embedder(other_agents_input)
        obstacles_embedding = self.obstacles_embedder(obstacles_input)
        targets_embedding = self.targets_embedder(targets_input)

        combined_other_agents_embedding = torch.mean(other_agents_embedding, dim=1)
        combined_obstacles_embedding = torch.mean(obstacles_embedding, dim=1)
        combined_targets_embedding = torch.mean(targets_embedding, dim=1)

        combined_other_agents_embedding = combined_obstacles_embedding.unsqueeze(1)
        combined_obstacles_embedding = combined_obstacles_embedding.unsqueeze(1)
        combined_targets_embedding = combined_targets_embedding.unsqueeze(1)

        # print(f"combined_other_agents_embedding: {combined_other_agents_embedding.shape}")
        # print(f"combined_obstacles_embedding: {combined_obstacles_embedding.shape}")
        # print(f"combined_targets_embedding: {combined_targets_embedding.shape}")
        # print(f"other_agents_embedding: {other_agents_embedding.shape}")
        # print(f"obstacles_embedding: {obstacles_embedding.shape}")
        # print(f"targets_embedding: {targets_embedding.shape}")

        other_agents_embedding = other_agents_embedding + combined_obstacles_embedding + combined_targets_embedding
        obstacles_embedding = obstacles_embedding + combined_other_agents_embedding + combined_targets_embedding
        targets_embedding = targets_embedding + combined_other_agents_embedding + combined_obstacles_embedding

        # h = torch.cat([self.other_agents_embedding(other_agents_input), self.ego_agent_embedding(ego_agent_input)],dim=1)
        # print(f"ego_agent_embedding: {ego_agent_embedding.shape}")
        ego_agent_embedding = ego_agent_embedding.unsqueeze(1)

        ego_agent_latent_vector = self.layers(ego_agent_embedding)

        # print(f"other_agents_embedding: {other_agents_embedding.shape}")
        other_agents_latent_vectors = self.layers(other_agents_embedding)
        obstacles_latent_vectors = self.layers(obstacles_embedding)
        targets_latent_vectors = self.layers(targets_embedding)

        # h = self.layers(h)
        return ego_agent_latent_vector, other_agents_latent_vectors, obstacles_latent_vectors, targets_latent_vectors

