# import ppo_attention_core
import torch
import numpy as np

class RoutePlanningBuffer:

    def __init__(self, num_targets):
        self.num_targets = num_targets
        # self.obs_dim = 2*(num_targets + 1) # 2*(num_targets + num_agents)
        self.node_dim = 2
        self.target_mask_buf = np.zeros((0, num_targets), dtype=np.bool_)
        # self.agent_position_buf = np.zeros((0, 1, 2), dtype=np.float32)
        # self.target_position_buf = np.zeros((0, num_targets, 2), dtype=np.float32)
        # self.obs_buf = np.zeros((0, self.num_targets + 1, 2), dtype=np.float32) #num_targets + num_agents
        self.target_positions_buf = np.zeros((0, num_targets, 2), dtype=np.float32)
        self.agent_positions_buf = np.zeros((0, 1, 2), dtype=np.float32)
        self.tour_length_rew_buf = np.zeros(0, dtype=np.float32)
        self.target_action_buf = np.zeros(0, dtype=np.float32)
        self.target_mask_buf = np.zeros((0, num_targets), dtype=np.bool_)
        self.tour_buf = np.zeros((0, num_targets), dtype=np.int64)

    # def store(self, agent_positions, target_positions, target_mask, target_action):
    def store(self, agent_positions, target_positions, target_mask, target_action):
        # print(f"agent_positions: {agent_positions}")
        # print(f"agent_position_buf: {self.agent_position_buf}")
        # self.agent_position_buf = np.append(self.agent_position_buf, agent_positions, axis=0)
        # self.target_position_buf = np.append(self.target_position_buf, target_positions, axis = 0)
        # self.obs_buf = np.append(self.obs_buf, obs, axis = 0)
        self.target_positions_buf = np.append(self.target_positions_buf, target_positions, axis = 0)
        self.agent_positions_buf = np.append(self.agent_positions_buf, agent_positions, axis = 0)
        self.target_mask_buf = np.append(self.target_mask_buf, target_mask, axis = 0)
        self.target_action_buf = np.append(self.target_action_buf, target_action)
        # self.tour_length_rew_buf = np.append(self.tour_length_rew_buf, rew)
        # self.tour_buf = np.append(self.tour_buf, tour, axis=0)

    def update_rew(self, tour_length_rew):
        self.tour_length_rew_buf = np.append(self.tour_length_rew_buf, [tour_length_rew]*self.num_targets)

    def get(self):
        # data = dict(agent_state=self.agent_position_buf, target_state=self.target_position_buf, target_mask=self.target_mask_buf, actions=self.target_action_buf, rew=self.tour_length_rew_buf)
        # data = dict(obs=self.obs_buf, target_mask=self.target_mask_buf, actions=self.target_action_buf, rew=self.tour_length_rew_buf)
        data = dict(agent_positions=self.agent_positions_buf, target_positions=self.target_positions_buf, target_mask=self.target_mask_buf, actions=self.target_action_buf, rew=self.tour_length_rew_buf)
        return data
        # return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

    def reset(self):
        # self.obs_buf = np.zeros((0, self.num_targets + 1, 2), dtype=np.float32)
        self.agent_positions_buf = np.zeros((0, 1, 2), dtype=np.float32)
        self.target_positions_buf = np.zeros((0, self.num_targets, 2), dtype=np.float32)
        self.target_mask_buf = np.zeros((0, self.num_targets), dtype=np.bool_)
        self.tour_length_rew_buf = np.zeros(0, dtype=np.float32)
        self.target_action_buf = np.zeros(0, dtype=np.float32)
        self.tour_buf = np.zeros((0, self.num_targets), dtype=np.int64)


class PathPlanningBuffer:

    def __init__(self, obs_dim, num_targets):
        self.obs_dim = obs_dim
        self.num_targets = num_targets
        self.mask_buf = np.zeros((0, 4), dtype=np.bool_)
        self.obs_buf = np.zeros((0, self.obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(0, dtype=np.float32)
        self.route_buf = np.zeros(0, dtype=np.int64)
        self.route_mask_buf = np.zeros((0, num_targets), dtype=np.bool_)

    def store(self, obs, mask, actions, target, route_mask):
        # print(f'obs: {obs.shape}')
        # print(f'obs_buf: {self.obs_buf.shape}')
        self.obs_buf = np.append(self.obs_buf, obs, axis=0)
        self.mask_buf = np.append(self.mask_buf, mask, axis=0)
        self.act_buf = np.append(self.act_buf, actions)
        self.route_buf = np.append(self.route_buf, target, axis=0)
        self.route_mask_buf = np.append(self.route_mask_buf, route_mask, axis=0)

    def get(self):
        data = dict(obs=self.obs_buf, mask=self.mask_buf, act=self.act_buf, route=self.route_buf, route_mask=self.route_mask_buf)
        return data

    def reset(self):
        # self.obs_buf = np.zeros((0, self.num_targets + 1, 2), dtype=np.float32)
        self.obs_buf = np.zeros((0, self.obs_dim), dtype=np.float32)
        self.mask_buf = np.zeros((0, 4), dtype=np.bool_)
        self.act_buf = np.zeros(0, dtype=np.float32)
        self.route_buf = np.zeros(0, dtype=np.int64)
        self.route_mask_buf = np.zeros((0, self.num_targets), dtype=np.bool_)



# class MultiAgentPathPlanningBuffer:
#     def __init__(self, num_agents, num_targets, num_obstacles):
#         self.num_agents = num_agents
#         self.num_targets = num_targets
#         self.num_obstacles = num_obstacles
#         # self.cat_input = cat_input
#         cat_input = num_agents
#         self.obs_dim = 2 * (num_agents + num_targets + num_obstacles) + 1 + cat_input # positions for agents, targets, and obstacles + 1 for the obstacle dimension + categorical input
#
#         self.action_mask_buf = np.zeros((0, 5), dtype=np.bool_)
#         self.target_mask_buf = np.zeros((0, self.num_targets + 1), dtype=np.bool_) #null target
#         self.action_buf = np.zeros(0, dtype=np.float32)
#         self.route_buf = np.zeros(0, dtype=np.int64)
#
#     def store(self, ego_agent_input, other_agents_input, targets_input, obstacles_input, action_mask, target_mask, action, target):
#         self.obs_buf = np.append(self.obs_buf, obs, axis=0)
#         self.action_mask_buf = np.append(self.action_mask_buf, action_mask, axis=0)
#         self.target_mask_buf = np.append(self.target_mask_buf, target_mask, axis=0)
#         self.action_buf = np.append(self.action_buf, action)
#         self.route_buf = np.append(self.route_buf, target, axis=0)
#
#     def get(self):
#         data = dict(obs=self.obs_buf, mask=self.action_mask_buf, target_mask=self.target_mask_buf, act=self.action_buf, route=self.route_buf)
#         return data
#
#     def reset(self):
#         self.obs_buf = np.zeros((0, self.obs_dim), dtype=np.float32)
#         self.action_mask_buf = np.zeros((0, 5), dtype=np.bool_)
#         self.target_mask_buf = np.zeros((0, self.num_targets), dtype=np.bool_)
#         self.action_buf = np.zeros(0, dtype=np.float32)
#         self.route_buf = np.zeros(0, dtype=np.int64)


class MultiAgentPathPlanningBuffer:
    def __init__(self, num_agents, num_targets, num_obstacles):
        self.num_agents = num_agents
        self.num_targets = num_targets
        self.num_obstacles = num_obstacles
        # self.cat_input = cat_input
        cat_input = num_agents

        self.ego_agent_buf = np.zeros((0, 2), dtype=np.float32)  # ego agent position
        self.other_agents_buf = np.zeros((0, num_agents - 1, 2), dtype=np.float32)  # other agents positions
        self.targets_buf = np.zeros((0, num_targets, 2), dtype=np.float32)  # target positions
        self.obstacles_buf = np.zeros((0, num_obstacles, 8), dtype=np.float32)  # obstacle positions and dimensions

        self.action_mask_buf = np.zeros((0, 5), dtype=np.bool_)
        self.target_mask_buf = np.zeros((0, self.num_targets + 1), dtype=np.bool_)  # null target
        self.action_buf = np.zeros(0, dtype=np.float32)
        self.route_buf = np.zeros(0, dtype=np.int64)

    def store(self, ego_agent_input, other_agents_input, targets_input, obstacles_input, action_mask, target_mask,
              action, target):
        self.ego_agent_buf = np.append(self.ego_agent_buf, ego_agent_input, axis=0)
        self.other_agents_buf = np.append(self.other_agents_buf, other_agents_input, axis=0)
        self.targets_buf = np.append(self.targets_buf, targets_input, axis=0)
        self.obstacles_buf = np.append(self.obstacles_buf, obstacles_input, axis=0)
        self.action_mask_buf = np.append(self.action_mask_buf, action_mask, axis=0)
        self.target_mask_buf = np.append(self.target_mask_buf, target_mask, axis=0)
        self.action_buf = np.append(self.action_buf, action)
        self.route_buf = np.append(self.route_buf, target, axis=0)

    def get(self):
        data = dict(ego_agent=self.ego_agent_buf, other_agents=self.other_agents_buf, targets=self.targets_buf, obstacles=self.obstacles_buf, mask=self.action_mask_buf, target_mask=self.target_mask_buf, act=self.action_buf,
                    route=self.route_buf)
        return data

    def reset(self):
        # self.obs_buf = np.zeros((0, self.obs_dim), dtype=np.float32)

        self.ego_agent_buf = np.zeros((0, 2), dtype=np.float32)  # ego agent position
        self.other_agents_buf = np.zeros((0, self.num_agents - 1, 2), dtype=np.float32)  # other agents positions
        self.targets_buf = np.zeros((0, self.num_targets, 2), dtype=np.float32)  # target positions
        self.obstacles_buf = np.zeros((0, self.num_obstacles, 8), dtype=np.float32)  # obstacle positions and dimensions

        self.action_mask_buf = np.zeros((0, 5), dtype=np.bool_)
        self.target_mask_buf = np.zeros((0, self.num_targets), dtype=np.bool_)
        self.action_buf = np.zeros(0, dtype=np.float32)
        self.route_buf = np.zeros(0, dtype=np.int64)