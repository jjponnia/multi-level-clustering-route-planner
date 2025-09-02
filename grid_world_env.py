import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from scipy.spatial.distance import euclidean
from torch_geometric.utils import normalized_cut

from agent import Agent
from params import allParams
# from multi_agent_route_planner import

from cluster_token import ClusterToken

from display_grid import display_grid
from display_grid import display_agents
from display_grid import display_targets
from display_grid import display_obstacles
from display_grid import display_cluster_lines
from display_grid import display_cluster_heads
from plotter import Plotter
# from bfs import shortest_tour
# from train_w_expertise import held_karp
# from model.config import config as cfg
import itertools
import networkx as nx
from astar import a_star
import copy
from additional_functions import chebyshev_distance
from additional_functions import euclidean

class gridworld_env:
    def __init__(self, seed=None, params=allParams):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        # THE NUMBERS USED TO GENERATE TRAINING DATA
        # self.grid_dim = 25
        # self.num_agents = 4
        # self.num_targets = 8
        # self.num_obstacles = 4
        # self.obstacle_size = 9

        self.grid_dim = 150
        self.num_agents = 70
        self.num_targets = 140
        self.num_obstacles = 0
        self.obstacle_size = 9


        self.list_of_agents = [Agent(i) for i in range(self.num_agents)]

        self.grid_state = np.zeros((self.grid_dim, self.grid_dim))

        self.max_num_obstacle_attempts = 8
        self.obstacle_positions = np.empty((0, 2))

        self.place_obstacles()
        self.obstacle_corners = self.create_obstacle_corners()

        self.agent_positions, self.target_positions = self.init_agents_and_targets_positions()
        self.reshaped_target_positions = self.target_positions.reshape(1, self.num_targets, 2)
        self.reset_agents()

        self.target_position = self.target_positions[0]

        self.obs_dim = 9 # (agent_position, target_position, obstacle_positions, obstacle_size)

        self.target_mask = [0 for _ in range(self.num_targets)]

        # self.cluster_params = cluster_params
        self.params = params
        self.clusterParams = self.params['agentParams']['clusterParams']
        self.threshold = self.clusterParams['l1_distance']
        self.cluster_counter = 0

        self.plotter = Plotter(self.params['envParams'])

    def are_agents_connected(self):
        highest_level = 0
        highest_agent = None
        for agent in self.list_of_agents:
            current_level = agent.clusterHeadToken.level_ if agent.clusterHeadToken else 0
            if highest_level < current_level:
                highest_level = current_level
                highest_agent = agent

        if highest_level == 0:
            return False, highest_level
        elif len(highest_agent.clusterHeadToken.get_all_members()) == len(self.list_of_agents):
            return True, highest_level
        else:
            print(f"Not all agents are connected. Highest level: {highest_level}, number of members: {len(highest_agent.clusterHeadToken.get_all_members())}, number of agents: {len(self.list_of_agents)}")
            return False, highest_level

    def create_obstacle_corners(self):
        obstacle_corners = np.empty((0, 8))
        for position in self.obstacle_positions:
            x, y = position
            corners = [
                (x, y),
                (x + self.obstacle_size - 1, y),
                (x, y + self.obstacle_size - 1),
                (x + self.obstacle_size - 1, y + self.obstacle_size - 1)
            ]
            corners = np.array(corners).flatten() / self.grid_dim  # Normalize to grid dimension
            corners = corners.reshape(1, 8)
            obstacle_corners = np.concatenate((obstacle_corners, corners), axis=0)

        obstacle_corners = obstacle_corners.reshape(1, self.num_obstacles, 8)
        return obstacle_corners


    def reset_agents(self):
        index = 0
        for agent in self.list_of_agents:
            agent.position = self.agent_positions[index]
            agent.grid_state = self.grid_state
            agent.reset()
            index += 1

    def update_agents_positions(self):
        self.agent_positions = [agent.position for agent in self.list_of_agents]


    def invert_actions(self, actions):
        # Define the mapping of actions to their opposites
        opposite_action = {0: 1, 1: 0, 2: 3, 3: 2, 4: 4}

        # Invert the sequence of actions
        inverted_actions = [opposite_action[action] for action in reversed(actions)]

        return inverted_actions


    def get_pairwise_distance_and_path_dict(self):
        grid = self.grid_state
        pairwise_distances = {}
        pairwise_paths = {}
        pairwise_action_sequences = {}

        self.update_agents_positions()

        # Filter out visited targets using the target_mask
        unvisited_target_positions = [
            self.target_positions[i] for i, visited in enumerate(self.target_mask) if not visited
        ]

        # Convert agent and unvisited target positions to grid cells
        agent_positions = [self.convert_position_to_grid_cell(pos) for pos in self.agent_positions]
        target_positions = [self.convert_position_to_grid_cell(pos) for pos in unvisited_target_positions]

        # Combine agent and unvisited target positions
        positions = agent_positions + target_positions

        num_positions = len(positions)

        for i in range(num_positions):
            for j in range(i + 1, num_positions):
                # print(f"i: {i}, j: {j}")
                start = positions[i]
                goal = positions[j]

                action_sequence, path = a_star(grid, start, goal)
                if path:
                    distance = len(path) - 1
                    pairwise_paths[(i, j)] = path
                    pairwise_paths[(j, i)] = path[::-1]
                    pairwise_action_sequences[(i, j)] = action_sequence
                    pairwise_action_sequences[(j, i)] = self.invert_actions(action_sequence)
                else:
                    distance = float('inf')
                    pairwise_paths[(i, j)] = None
                    pairwise_paths[(j, i)] = None
                    pairwise_action_sequences[(i, j)] = None
                    pairwise_action_sequences[(j, i)] = None

                pairwise_distances[(i, j)] = distance
                pairwise_distances[(j, i)] = distance

        return pairwise_distances, pairwise_paths, pairwise_action_sequences, positions


    def place_obstacles(self):
        num_obstacles_placed = 0
        self.grid_state.fill(0)
        self.obstacle_positions = np.empty((0, 2))

        while num_obstacles_placed < self.num_obstacles:

            max_attempts = 1000
            attempts = 0
            while attempts < max_attempts:
                attempts += 1
                if self.place_obstacle():
                    # num_obstacle_attempts += 1
                    num_obstacles_placed += 1
                    # print(f"num_obstacles_placed: {num_obstacles_placed}")

                if num_obstacles_placed >= self.num_obstacles:
                    break

            if attempts == max_attempts:
                self.grid_state.fill(0)
                self.obstacle_positions = np.empty((0, 2))
                num_obstacles_placed = 0


    def place_obstacle(self):
        max_attempts = 100
        attempts = 0

        obstacle_size = self.obstacle_size

        while attempts < max_attempts:
            start_x = np.random.randint(0, self.grid_dim - obstacle_size + 1)
            start_y = np.random.randint(0, self.grid_dim - obstacle_size + 1)
            # print(f"grid_state: {self.grid_state}")
            if self.is_valid_obstacle_position(start_x, start_y, obstacle_size):
                for i in range(obstacle_size):
                    for j in range(obstacle_size):
                        self.grid_state[start_x + i,start_y+j] = 1
                # print(f"self.obstacle_positions: {self.obstacle_positions}")
                self.obstacle_positions = np.concatenate((self.obstacle_positions, np.array([start_x, start_y]).reshape(1, 2)), axis=0)
                return True
            attempts += 1

        return False


    def is_valid_obstacle_position(self, start_x, start_y, obstacle_size):
        for i in range(obstacle_size):
            for j in range(obstacle_size):
                if self.grid_state[start_x + i, start_y + j] != 0: # or delta_agent < 0.001 or delta_target < 0.001:
                    return False
        return True


    def is_valid_target_position(self, agent_position, target_position):
        agent_position = self.convert_position_to_grid_cell(agent_position)
        target_position = self.convert_position_to_grid_cell(target_position)
        grid = self.grid_state
        result = a_star(grid, agent_position, target_position)

        if result is None:
            return False
        else:
            return True


    def get_path_model_obs_input(self):
        agent_positions = np.array([agent.position for agent in self.list_of_agents])
        agent_positions = agent_positions.reshape(1, len(self.list_of_agents), 2)

        return agent_positions, self.reshaped_target_positions, self.obstacle_corners


    def convert_to_one_hot_vector(self, agent_id):
        one_hot_vector = np.zeros(self.num_agents)
        one_hot_vector[agent_id] = 1
        return one_hot_vector

    # doesn't seem to be used
    def get_distance_to_target(self):
        target_position = self.target_position
        agent_position = self.agent_position
        distance = abs(target_position[0] - agent_position[0]) + abs(target_position[1] - agent_position[1])
        return distance


    # don't think I use this
    def get_agent_position(self):
        return self.agent_position


    def init_agents_and_targets_positions(self):
        unique_positions = set()
        offset = 1 / self.grid_dim

        while len(unique_positions) < self.num_targets + self.num_agents:

            x = np.random.randint(0, self.grid_dim)
            y = np.random.randint(0, self.grid_dim)

            if self.grid_state[x, y] == 0:
                x = x * offset + offset / 2
                y = y * offset + offset / 2
                target_position = (x, y)

                if len(unique_positions) == 0:
                    valid_position = True
                else:
                    agent_position = np.array(list(unique_positions)[0])
                    # target_position = np.array(target_position)
                    valid_position = self.is_valid_target_position(agent_position, target_position)

                if valid_position:
                    unique_positions.add(target_position)

        # target_positions = np.array(list(unique_positions))
        num_agents = self.num_agents
        agent_positions = np.array(list(unique_positions)[:num_agents])
        target_positions = np.array(list(unique_positions)[num_agents:])

        index = 0
        for agent in self.list_of_agents:
            agent.position = agent_positions[index]
            index += 1
            # print(f"agent.position: {agent.position}")

        return agent_positions, target_positions


    def convert_position_to_grid_cell(self, position):
        x, y = position
        x_cell = int((x * self.grid_dim))
        y_cell = int((y * self.grid_dim))
        return x_cell, y_cell


    def update_target_mask(self):
        for agent in self.list_of_agents:
            agent_cell = self.convert_position_to_grid_cell(agent.position)
            # print(f"agent_cell (grid_world_env): {agent_cell}")

            index = 0

            for position in self.target_positions:
                target_cell = self.convert_position_to_grid_cell(position)

                if agent.id == 22 and index == 53:
                    print(f"agent cell: {agent_cell}, target cell: {target_cell}, target: {index} target mask: {self.target_mask[index]}, next_target_id: {agent.next_target_id}")

                if agent_cell == target_cell and self.target_mask[index] == 0 and agent.next_target_id == index:
                    self.target_mask[index] = 1
                    agent.grid_mask = np.zeros((self.grid_dim, self.grid_dim, 2))
                    agent.previous_position = None
                    agent.next_target_id = self.num_targets
                    agent.update_grid_mask()
                    agent.update_action_mask()

                index += 1


    def action_step(self, actions):
        done = False

        for agent_id, action in actions.items():
            agent = self.list_of_agents[agent_id]
            if actions[agent.id] == 0:
                agent.previous_position = copy.deepcopy(agent.position)
                agent.position[1] += 1 / self.grid_dim
            elif actions[agent.id] == 1:
                agent.previous_position = copy.deepcopy(agent.position)
                agent.position[1] -= 1 / self.grid_dim
            elif actions[agent.id] == 2:
                agent.previous_position = copy.deepcopy(agent.position)
                agent.position[0] -= 1 / self.grid_dim
            elif actions[agent.id] == 3:
                agent.previous_position = copy.deepcopy(agent.position)
                agent.position[0] += 1 / self.grid_dim

            agent.update_grid_mask()
            agent.update_action_mask()

            if sum(agent.action_mask) == 4:
                agent.grid_mask = np.zeros((self.grid_dim, self.grid_dim, 2))
                agent.update_action_mask()
                agent.reset_flag = True
                # reset = True

        self.update_target_mask()

        if sum(self.target_mask) == self.num_targets:
            done = True

        target_mask = self.target_mask
        rew = 1

        return None, None, rew, done, target_mask


    def cluster_step(self):
        all_connected, _ = self.are_agents_connected()
        print(f"all_connected: {all_connected}, cluster_counter: {self.cluster_counter}")
        if not all_connected:
            print(f"Starting cluster formation phase")
            self.cluster_formation_phase()
            # self.cluster_maintenance_phase()
            # self.render()
            self.cluster_formation_phase()
            # self.cluster_maintenance_phase()
            # self.render()
            # self.cluster_maintenance_phase()
            self.cluster_formation_phase()
            # self.render()
            print(f"Finished cluster formation phase")

        if all_connected and self.cluster_counter % 3 == 0:
            print(f"Starting cluster maintenance phase")
            self.cluster_maintenance_phase()
            # self.render()
            print(f"Finishing cluster maintenance phase")

        self.cluster_counter += 1


    def cluster_maintenance_phase(self):
        self.lvl1_cluster_assimilate_step()
        self.cluster_assimilate_step()
        self.cluster_rebalance_step()

        self.lvl1_cluster_split_step()
        self.cluster_split_step()
        self.cluster_rebalance_step()

        self.lvl1_transfer_agents_step()
        # print(f"Completed lvl1 transfer.  Starting transfer.")
        # self.render()
        self.transfer_agents_step()
        self.cluster_rebalance_step()

        self.cluster_join_step()
        self.cluster_rebalance_step()

        self.lvl1_cluster_assimilate_step()
        self.cluster_assimilate_step()
        self.cluster_rebalance_step()

        self.lvl1_cluster_split_step()
        # self.render()
        self.cluster_split_step()
        # self.render()
        self.cluster_rebalance_step()

        self.lvl1_transfer_agents_step()
        self.transfer_agents_step()
        self.cluster_rebalance_step()


    def cluster_formation_phase(self):
        all_connected, _ = self.are_agents_connected()
        print(f"all_connected: {all_connected}, cluster_counter: {self.cluster_counter}")
        if not all_connected:
            self.lvl1_cluster_formation_step()
            self.cluster_join_step()
            self.cluster_formation_step()
            self.cluster_join_step()

        self.lvl1_cluster_assimilate_step()
        self.cluster_assimilate_step()
        self.cluster_rebalance_step()

        self.lvl1_cluster_split_step()
        # self.render()
        self.cluster_split_step()
        # self.render()
        self.cluster_rebalance_step()

        self.lvl1_transfer_agents_step(cluster_formation=True)
        self.transfer_agents_step(cluster_formation=True)
        self.cluster_rebalance_step()


    def lvl1_cluster_formation_step(self):
        print(f"lvl1 Cluster formation step")
        for agent in self.list_of_agents:
            level = agent.clusterHeadToken.level_ if agent.clusterHeadToken else 0
            threshold = self.clusterParams['l1_distance'] * ((level + 1) ** 2)
            agent.get_nearest_neighbors(self.list_of_agents, threshold)
            print(f"agent: {agent.id}, level: {level}, neighbors: {agent.list_of_neighbors}")

        for agent in self.list_of_agents:
            level = agent.clusterHeadToken.level_ if agent.clusterHeadToken else 0

            # skip over any agents with level > 0
            if level != 0:
                continue
            # skip over level 0 agents that are already in a level 1 cluster
            elif agent.lvl1Cluster is not None:
                continue

            # if we've made it this far then the agent is eligible to form a lvl1 cluster with an eligible neighbor
            # we will create a token and look for eligible neighbors
            token = ClusterToken(level = level + 1, clusterParams=self.clusterParams)

            print(f"attempting to form level {level + 1} cluster for agent {agent.id}")
            # list of eligible neighbors to form a cluster with
            lvl1_candidate_members = []

            # cluster formation should be restricted to agents in the neighborhood
            for neighbor_agent in self.list_of_agents:
                if neighbor_agent.id == agent.id:
                    continue

                # neighbor_agent = self.list_of_agents[neighbor_id]
                neighbor_level = neighbor_agent.clusterHeadToken.level_ if neighbor_agent.clusterHeadToken else 0
                if neighbor_level != 0:
                    print(
                        f"Skipping neighbor {neighbor_agent.id} for agent {agent.id} at level {level + 1} because agent is not level 0")
                    continue
                elif neighbor_agent.lvl1Cluster is not None:
                    print(
                        f"Skipping neighbor {neighbor_agent.id} for agent {agent.id} at level {level + 1} because neighbor is already in a lvl1 cluster")
                    continue

                lvl1_candidate_members.append(neighbor_agent)

            if len(lvl1_candidate_members) == 0:
                print(
                    f"Unable to form a level 1 cluster with agent {agent.id} because it has no eligible neighbors")
                continue

            # sort the members by distance
            distance_of_candidates = [(candidate, euclidean(agent.position, candidate.position)) for candidate in lvl1_candidate_members]
            distance_of_candidates = sorted(distance_of_candidates, key=lambda x: x[1])

            # max number of cluster members
            max_num_members = min(len(lvl1_candidate_members), self.clusterParams['max_num_agents'] - 2)

            # take the closest candidates up to the max number of members
            lvl1_candidate_members = [candidate for candidate, _ in distance_of_candidates[:max_num_members]]

            # sumLoc = [0, 0]
            # num_new_cluster_members = 0
            for lvl1_member in lvl1_candidate_members:
                # sumLoc = [x + y for x, y in zip(sumLoc, lvl1_member.position)]
                # num_new_cluster_members += 1
                token.members_.append(lvl1_member)
                lvl1_member.lvl1Cluster = token

            print(
                f"Giving lvl1 cluster head token to agent {agent.id} for new cluster with members {[member.id for member in token.members_]}")
            token.tokenHolder_ = agent
            agent.clusterHeadToken = token
            agent.lvl1Cluster = token

        print(f"Concluding lvl1 cluster formation step")
        for agent in self.list_of_agents:
            if agent.clusterHeadToken:
                print(
                    f"Agent {agent.id} has cluster head token with level {agent.clusterHeadToken.level_} and members {[member.id for member in agent.clusterHeadToken.members_]}")


    def cluster_formation_step(self):
        print(f"Cluster formation step")
        for agent in self.list_of_agents:
            level = agent.clusterHeadToken.level_ if agent.clusterHeadToken else 0
            threshold = self.clusterParams['l1_distance'] * ((level + 1) ** 2)
            agent.get_nearest_neighbors(self.list_of_agents, threshold)
            print(f"agent: {agent.id}, level: {level}, neighbors: {agent.list_of_neighbors}")

        #iterate through all the agents and attempt to form clusters
        for agent in self.list_of_agents:
            level = agent.clusterHeadToken.level_ if agent.clusterHeadToken else 0

            # skip over level 0 agents
            if level == 0:
                print(f"Skipping agent {agent.id} at level {level} because it is a level 0 agent")
                continue
            # skip over higher-level agents that already have a parent
            elif agent.clusterHeadToken.higherCluster_ is not None:
                print(f"Skipping agent {agent.id} at level {level} because it already has a parent cluster")
                continue
            # skip over higher-level agents that are parents of small clusters (these need to be merged)
            elif len(agent.clusterHeadToken.members_) < 0.5 * self.clusterParams['max_num_agents']:
                # agent already has a cluster head token and it is half empty.  Try to merge instead.
                print(f"Skipping agent {agent.id} at level {level} because it has a half empty cluster")
                continue

            # if we've made it this far then the agent is eligible to form a new cluster with an eligible neighbor
            # we will create a token and look for eligible neighbors
            token = ClusterToken(level=level + 1, clusterParams=self.clusterParams)

            print(f"attempting to form level {level + 1} cluster for agent {agent.id}")
            # list of eligible neighbors to form a cluster with
            higher_cluster_members = [agent]

            for neighbor_agent in self.list_of_agents:
                # neighbor_agent = self.list_of_agents[agent_id]
                if neighbor_agent.id == agent.id:
                    continue

                neighbor_level = neighbor_agent.clusterHeadToken.level_ if neighbor_agent.clusterHeadToken else 0
                if level == 0:
                    print(f"Skipping neighbor {neighbor_agent.id} for agent {agent.id} at level {level + 1} because agent is level 0")
                    continue
                elif neighbor_level != level:
                    print(f"Skipping neighbor {neighbor_agent.id} for agent {agent.id} at level {level + 1} because neighbor level {neighbor_level} does not match agent level {level}")
                    continue
                elif neighbor_agent.clusterHeadToken.higherCluster_ is not None:
                    print(f"Skipping neighbor {neighbor_agent.id} for agent {agent.id} at level {level + 1} because neighbor has a parent")
                    continue
                # elif len(neighbor_agent.clusterHeadToken.members_) < 0.5 * self.clusterParams['max_num_agents']:
                #     print(f"Skipping neighbor {neighbor_agent.id} for agent {agent.id} at level {level + 1} because neighbor's cluster is half empty")
                #     continue

                higher_cluster_members.append(neighbor_agent)


            # order higher_cluster_members by distance from agent
            distance_of_higher_cluster_members = [(member, euclidean(agent.position, member.position)) for member in higher_cluster_members]
            distance_of_higher_cluster_members = sorted(distance_of_higher_cluster_members, key=lambda x: x[1])

            max_num_members = min(len(higher_cluster_members), self.clusterParams['max_num_agents'] - 2)
            higher_cluster_members = [member for member, _ in distance_of_higher_cluster_members[:max_num_members]]


            if len(higher_cluster_members) > 1:
                print(f"Forming a new level {level + 1} cluster with {[member.id for member in higher_cluster_members]} members")
            else:
                print(f"Skipping agent {agent.id} at level {level + 1} because it has no eligible neighbors")
                continue

            # get idle members of all neighbors in higher_cluster_members
            candidate_cluster_heads = agent.clusterHeadToken.get_idle()
            for member in higher_cluster_members:
                #if len(member.clusterHeadToken.members_) < 0.5 * self.clusterParams['max_num_agents']:
                #     continue
                # else:
                candidate_cluster_heads += member.clusterHeadToken.get_idle()

            sumLoc = [0, 0]
            num_new_cluster_members = 0
            # compute centroid of the new cluster and add higher_members to the token
            for higher_member in higher_cluster_members:
                sumLoc = [x + y for x, y in zip(sumLoc, higher_member.position)]
                num_new_cluster_members += 1
                token.members_.append(higher_member)
                higher_member.clusterHeadToken.higherCluster_ = token

            sumLoc = [x + y for x, y in zip(sumLoc, agent.position)]
            num_new_cluster_members += 1

            combined_centroid = [x / num_new_cluster_members for x in sumLoc]

            centroidDistance = [(candidate, euclidean(combined_centroid, candidate.position)) for candidate in candidate_cluster_heads]
            sortedCandidateList = sorted(centroidDistance, key=lambda x: x[1])
            bestAgent = sortedCandidateList[0][0]

            print(f"Giving cluster head token to agent {bestAgent.id} for new cluster with members {[member.id for member in token.members_]}")
            token.tokenHolder_ = bestAgent
            bestAgent.clusterHeadToken = token

        print(f"Concluding cluster formation step")
        for agent in self.list_of_agents:
            if agent.clusterHeadToken:
                print(f"Agent {agent.id} has cluster head token with level {agent.clusterHeadToken.level_} and members {[member.id for member in agent.clusterHeadToken.members_]}")


    def lvl1_cluster_assimilate_step(self):
        print(f"lvl1 Cluster assimilate step")
        for agent in self.list_of_agents:
            level = agent.clusterHeadToken.level_ if agent.clusterHeadToken else 0
            threshold = self.clusterParams['l1_distance'] * ((level + 1) ** 2)
            agent.get_nearest_neighbors(self.list_of_agents, threshold)
            print(f"agent: {agent.id}, level: {level}, neighbors: {agent.list_of_neighbors}")

        # iterate through all non-empty level 1 clusters and get the average difference between centroid and members
        print(f"Calculating average distance between centroid and members of level 1 clusters")
        agent_average_distance = 0
        num_lvl1_clusters = 0
        for agent in self.list_of_agents:
            level = agent.clusterHeadToken.level_ if agent.clusterHeadToken is not None else 0
            if level != 1:
                continue
            elif len(agent.clusterHeadToken.members_) == 0:
                continue

            agent_centroid = agent.clusterHeadToken.update_centroid()
            agent_average_distance_per_cluster = 0
            for member in agent.clusterHeadToken.members_:
                agent_average_distance_per_cluster += euclidean(agent_centroid, member.position)

            agent_average_distance_per_cluster /= len(agent.clusterHeadToken.members_)

            num_lvl1_clusters += 1
            agent_average_distance += agent_average_distance_per_cluster

        agent_average_distance /= num_lvl1_clusters

        print(f"Average distance between centroid and members of level 1 clusters: {agent_average_distance}")

        # iterate through all level 2 clusters and decide whether to assimilate level 1 clusters
        print(f"Looking through level 1 clusters to find level 1 clusters")

        for agent in self.list_of_agents:
            level = agent.clusterHeadToken.level_ if agent.clusterHeadToken is not None else 0
            size_of_my_cluster = len(agent.clusterHeadToken.members_) if agent.clusterHeadToken else 0

            if level != 1:
                print(f"Skipping agent {agent.id} at level {level} because it is not a level 1 agent")
                continue
            # if the cluster is full that is taken care of elsewhere (cluster split)

            agents_parent_id = agent.clusterHeadToken.higherCluster_.tokenHolder_.id if agent.clusterHeadToken.higherCluster_ else None

            print(f"Attempting to assimilate cluster of agent {agent.id} at level {level}")
            assimilate_candidates = []
            for neighbor_id in agent.list_of_neighbors:
                if agent.id == neighbor_id:
                    continue
                neighbor_agent = self.list_of_agents[neighbor_id]
                neighbor_level = neighbor_agent.clusterHeadToken.level_ if neighbor_agent.clusterHeadToken else 0
                neighbor_agents_parent_cluster = neighbor_agent.clusterHeadToken.higherCluster_ if neighbor_agent.clusterHeadToken else None
                neighbor_agents_parent_id = neighbor_agents_parent_cluster.tokenHolder_.id if neighbor_agents_parent_cluster else None
                size_of_neighbor_cluster = len(
                    neighbor_agent.clusterHeadToken.members_) if neighbor_agent.clusterHeadToken else 0

                # reject neighbors that are not at the same level, are not cluster-heads, or have a different parent id
                # since we only accept neighbors with the same parent id this ensures that every cluster-heads remains in the level 1 cluster of its subtree
                if neighbor_level != level:
                    print(
                        f"Skipping neighbor {neighbor_agent.id} for agent {agent.id} at level {level} because neighbor level {neighbor_level} does not match agent level {level}")
                    continue
                elif size_of_my_cluster + size_of_neighbor_cluster + 1 >= self.clusterParams['max_num_agents']:
                    print(
                        f"Skipping neighbor {neighbor_agent.id} for agent {agent.id} at level {level} because neighbor's cluster is too big to assimilate into")
                    continue
                elif agents_parent_id != neighbor_agents_parent_id:
                    print(
                        f"Skipping neighbor {neighbor_agent.id} for agent {agent.id} at level {level} because neighbor's parent id {neighbor_agents_parent_id} does not match agent's parent id {agents_parent_id}")
                    continue

                # compute average distance between centroid of neighbor and neighbor members
                neighbor_centroid = neighbor_agent.clusterHeadToken.update_centroid()
                agent_centroid = agent.clusterHeadToken.update_centroid()

                print(f"Agent {agent.id} has centroid {agent_centroid}.")
                print(
                    f"Neighbor {neighbor_agent.id} has centroid {neighbor_centroid}.")
                print(f"Distance between centroids: {euclidean(agent_centroid, neighbor_centroid)}")
                # print(
                #     f"Average distance between centroid deviations: {0.5 * (agent_average_distance + neighbor_average_distance)}")

                # if the average average distance in the cluster is less than 2 times the distance between centroids then we do not assimilate (clusters aren't close enough)
                # singleton level 1 clusters are always assimilated
                if agents_parent_id is not None:
                    if agent_average_distance < 2 * euclidean(agent_centroid, neighbor_centroid):
                        print(
                            f"Skipping assimilation of neighbor {neighbor_agent.id} into agent {agent.id} at level {level} because clusters are not close enough")
                        continue

                assimilate_candidates.append(neighbor_agent)

            if len(assimilate_candidates) > 0:
                centroidDistance = [(candidate, euclidean(candidate.clusterHeadToken.update_centroid(), agent.clusterHeadToken.update_centroid())) for candidate in assimilate_candidates if candidate.clusterHeadToken]
                sortedCandidateList = sorted(centroidDistance, key=lambda x: x[1])
                bestCandidate = sortedCandidateList[0][0]
            else:
                print(f"No assimilate candidates found for agent {agent.id} at level {level}")
                continue

            if len(bestCandidate.clusterHeadToken.members_) * size_of_my_cluster != 0:
                for member in agent.clusterHeadToken.members_:
                    print(
                        f"Assimilating member {member.id} from agent {agent.id} to best candidate {bestCandidate.id}")
                    bestCandidate.clusterHeadToken.members_.append(member)
                    member.lvl1Cluster = bestCandidate.clusterHeadToken

                # remove agent from parent cluster (if it exists) because agent is going to assimilate into best_candidate's cluster
                if agents_parent_id is not None:
                    agents_parent = agent.clusterHeadToken.higherCluster_.tokenHolder_
                    agents_parent.clusterHeadToken.members_ = [m for m in agents_parent.clusterHeadToken.members_ if m.id != agent.id]

                # add agent to the best candidate's cluster
                print(f"Assimilating agent {agent.id} to candidate {bestCandidate.id}")
                bestCandidate.clusterHeadToken.members_.append(agent)
                agent.lvl1Cluster = bestCandidate.clusterHeadToken
                agent.clusterHeadToken = None
            elif len(bestCandidate.clusterHeadToken.members_) == 0:
                # add bestCandidate to agent's cluster
                agent.clusterHeadToken.members_.append(bestCandidate)

                # remove bestCandidate from parent cluster
                if agents_parent_id is not None:
                    agents_parent = agent.clusterHeadToken.higherCluster_.tokenHolder_
                    agents_parent.clusterHeadToken.members_ = [m for m in agents_parent.clusterHeadToken.members_ if m.id != bestCandidate.id]

                # agent.clusterHeadToken = [m for m in agent.clusterHeadToken.members_ if m.id != bestCandidate.id]

                # add bestCandidate to agent's cluster
                print(f"Assimilating agent {bestCandidate.id} to agent {agent.id}")
                agent.clusterHeadToken.members_.append(bestCandidate)
                bestCandidate.lvl1Cluster = agent.clusterHeadToken
                bestCandidate.clusterHeadToken = None
            elif len(agent.clusterHeadToken.members_) == 0:
                # add agent to bestCandidate's cluster
                bestCandidate.clusterHeadToken.members_.append(agent)
                agent.lvl1Cluster = bestCandidate.clusterHeadToken
                agent.clusterHeadToken = None

                if agents_parent_id is not None:
                    agents_parent = agent.clusterHeadToken.higherCluster_.tokenHolder_
                    agents_parent.clusterHeadToken.members_ = [m for m in agents_parent.clusterHeadToken.members_ if m.id != bestCandidate.id]

                # remove agent from agent's parent cluster because agent is going to assimilate into neighbor_agent's cluster
                agents_parent.clusterHeadToken.members_ = [m for m in agents_parent.clusterHeadToken.members_ if m.id != agent.id]

                print(f"Assimilating agent {agent.id} to candidate {bestCandidate.id}")

        print(f"Concluding lvl1 cluster assimilation step")
        for agent in self.list_of_agents:
            if agent.clusterHeadToken:
                print(
                    f"Agent {agent.id} has cluster head token with level {agent.clusterHeadToken.level_} and members {[member.id for member in agent.clusterHeadToken.members_]}")


    def cluster_assimilate_step(self):
        print(f"Cluster assimilate step")
        for agent in self.list_of_agents:
            level = agent.clusterHeadToken.level_ if agent.clusterHeadToken else 0
            threshold = self.clusterParams['l1_distance'] * ((level + 1) ** 2)
            agent.get_nearest_neighbors(self.list_of_agents, threshold)
            print(f"agent: {agent.id}, level: {level}, neighbors: {agent.list_of_neighbors}")

        # we iterate through all the agents and stop at agents that are eligible for assimilation
        for agent in self.list_of_agents:
            level = agent.clusterHeadToken.level_ if agent.clusterHeadToken else 0
            agents_parent_cluster = agent.clusterHeadToken.higherCluster_ if agent.clusterHeadToken else None
            agents_parent_id = agents_parent_cluster.tokenHolder_.id if agents_parent_cluster else None

            # cluster_centroid = agent.clusterHeadToken.get_centroid()

            # skip over agents that are level 0 or have a half full cluster (level 0 agents are not cluster-heads)
            if level <= 1:
                print(f"Skipping agent {agent.id} because it is a level {level} agent (less than 2)")
                continue
            elif len(agent.clusterHeadToken.members_) > 0.5 * self.clusterParams['max_num_agents']:
                print(f"Skipping agent {agent.id} at level {level} because it has a half full cluster")
                continue

            print(f"Attempting to assimilate cluster of agent {agent.id} at level {level}")
            assimilate_candidates = []
            for agent_id in agent.list_of_neighbors:
                neighbor_agent = self.list_of_agents[agent_id]
                neighbor_level = neighbor_agent.clusterHeadToken.level_ if neighbor_agent.clusterHeadToken else 0
                neighbor_agents_parent_cluster = neighbor_agent.clusterHeadToken.higherCluster_ if neighbor_agent.clusterHeadToken else None
                neighbor_agents_parent_id = neighbor_agents_parent_cluster.tokenHolder_.id if neighbor_agents_parent_cluster else None
                size_of_neighbor_cluster = len(neighbor_agent.clusterHeadToken.members_) if neighbor_agent.clusterHeadToken else 0
                size_of_my_cluster = len(agent.clusterHeadToken.members_) if agent.clusterHeadToken else 0

                # reject neighbors that are not at the same level, are not cluster-heads, or have a different parent id
                # since we only accept neighbors with the same parent id this ensures that every cluster-heads remains in the level 1 cluster of its subtree
                if neighbor_level == 0:
                    print(f"Skipping neighbor {neighbor_agent.id} for agent {agent.id} at level {level + 1} because neighbor is level 0")
                    continue
                elif neighbor_level != level:
                    print(f"Skipping neighbor {neighbor_agent.id} for agent {agent.id} at level {level + 1} because neighbor level {neighbor_level} does not match agent level {level}")
                    continue
                elif size_of_my_cluster + size_of_neighbor_cluster + 1 >= self.clusterParams['max_num_agents']:
                    print(f"Skipping neighbor {neighbor_agent.id} for agent {agent.id} at level {level + 1} because neighbor's cluster is too big to assimilate into")
                    continue
                elif agents_parent_id != neighbor_agents_parent_id:
                    print(f"Skipping neighbor {neighbor_agent.id} for agent {agent.id} at level {level + 1} because neighbor's parent id {neighbor_agents_parent_id} does not match agent's parent id {agents_parent_id}")
                    continue

                # compute average distance between centroid of neighbor and neighbor members
                neighbor_centroid = neighbor_agent.clusterHeadToken.update_centroid()
                neighbor_average_distance = 0
                for member in neighbor_agent.clusterHeadToken.members_:
                    neighbor_average_distance += euclidean(neighbor_centroid, member.position)
                neighbor_average_distance /= len(neighbor_agent.clusterHeadToken.members_) if len(neighbor_agent.clusterHeadToken.members_) > 0 else 0

                agent_centroid = agent.clusterHeadToken.update_centroid()
                agent_average_distance = 0
                for member in agent.clusterHeadToken.members_:
                    agent_average_distance += euclidean(agent_centroid, member.position)
                agent_average_distance /= len(agent.clusterHeadToken.members_) if len(agent.clusterHeadToken.members_) > 0 else 0

                print(f"Agent {agent_id} has centroid {agent_centroid} and average distance {agent_average_distance}.")
                print(f"Neighbor {neighbor_agent.id} has centroid {neighbor_centroid} and average distance {neighbor_average_distance}.")
                print(f"Distance between centroids: {euclidean(agent_centroid, neighbor_centroid)}")
                print(f"Average distance between centroid deviations: {0.5 * (agent_average_distance + neighbor_average_distance)}")

                # if the average average distance in the cluster is less than 2 times the distance between centroids then we do not assimilate (clusters aren't close enough)
                # singleton level 1 clusters are always assimilated
                if agents_parent_id is not None:
                    if 0.5 * (agent_average_distance + neighbor_average_distance) < 2 * euclidean(agent_centroid, neighbor_centroid):
                        print(f"Skipping assimilation of neighbor {neighbor_agent.id} into agent {agent.id} at level {level + 1} because clusters are not close enough")
                        continue

                assimilate_candidates.append(neighbor_agent)

            # get the centroid of each cluster associated with an assimilate candidate
            if len(assimilate_candidates) > 0:
                centroidDistance = [(candidate, euclidean(candidate.clusterHeadToken.update_centroid(), agent.clusterHeadToken.update_centroid())) for candidate in assimilate_candidates if candidate.clusterHeadToken]
                sortedCandidateList = sorted(centroidDistance, key=lambda x: x[1])
                bestCandidate = sortedCandidateList[0][0]
            else:
                print(f"No assimilate candidates found for agent {agent.id} at level {level}")
                continue

            # for each member in the agent's cluster, add them to the best candidate's cluster
            for member in agent.clusterHeadToken.members_:
                print(f"Assimilating member {member.id} from agent {agent.id} to candidate {bestCandidate.id}")
                bestCandidate.clusterHeadToken.members_.append(member)
                # if level == 1:
                #     member.lvl1Cluster = bestCandidate.clusterHeadToken
                # else:
                member.clusterHeadToken.higherCluster_ = bestCandidate.clusterHeadToken

            # remove agent from higher level cluster (if it exists) because agent is either going to assimilate or destroy token
            if agent.clusterHeadToken.higherCluster_:
                agent.clusterHeadToken.higherCluster_.members_ = [member for member in agent.clusterHeadToken.higherCluster_.members_ if member.id != agent.id]

            # add the agent itself to the best candidate's cluster if a level 1 cluster otherwise destroy token
            # if level == 1:
            #     print(f"Assimilating agent {agent.id} to candidate {bestCandidate.id}")
            #     bestCandidate.clusterHeadToken.members_.append(agent)
            #     agent.lvl1Cluster = bestCandidate.clusterHeadToken
            #     agent.clusterHeadToken = None
            # else:
            print(f"Destroying agents {agent.id} cluster head token and reverting back to lvl1 cluster")
            agent.clusterHeadToken = None

        print(f"Concluding cluster assimilation step")
        for agent in self.list_of_agents:
            if agent.clusterHeadToken:
                print(
                    f"Agent {agent.id} has cluster head token with level {agent.clusterHeadToken.level_} and members {[member.id for member in agent.clusterHeadToken.members_]}")


    def cluster_rebalance_step(self):
        print(f"Cluster rebalance step")
        for agent in self.list_of_agents:
            level = agent.clusterHeadToken.level_ if agent.clusterHeadToken else 0
            threshold = self.clusterParams['l1_distance'] * ((level + 1) ** 2)
            agent.get_nearest_neighbors(self.list_of_agents, threshold)
            print(f"agent: {agent.id}, level: {level}, neighbors: {agent.list_of_neighbors}")

        # iterate through all the agents and stop at agents that are eligible for rebalancing
        for agent in self.list_of_agents:
            level = agent.clusterHeadToken.level_ if agent.clusterHeadToken else 0

            if level == 0:
                print(f"Skipping agent {agent.id} at level {level} because it is a level 0 agent")
                continue
            elif len(agent.clusterHeadToken.members_) == 0:
                print(f"Skipping agent {agent.id} at level {level} because it has no members")
                continue

            print(f"Computing average distance of agent {agent.id} at level {level}")

            centroid = agent.clusterHeadToken.update_centroid()
            average_distance = 0

            for member in agent.clusterHeadToken.members_:
                average_distance += euclidean(centroid, member.position)

            average_distance /= len(agent.clusterHeadToken.members_)

            candidate_clusterheads = agent.clusterHeadToken.get_idle()
            candidate_clusterheads.append(agent)  # include the agent itself as a candidate
            centroidDistance = [(candidate, euclidean(centroid, candidate.position)) for candidate in candidate_clusterheads]
            sortedCandidateList = sorted(centroidDistance, key=lambda x: x[1])


            if len(sortedCandidateList) == 0:
                print(f"No idle agents found to rebalance cluster of agent {agent.id} at level {level}")
                continue

            bestCandidate = sortedCandidateList[0][0]

            normalized_proposed_gap = euclidean(centroid, bestCandidate.position) / average_distance
            normalized_current_gap = euclidean(centroid, agent.position) / average_distance

            if bestCandidate.id == agent.id:
                print(f"Agent {agent.id} is already the best candidate for rebalancing")
                continue

            if normalized_proposed_gap > 0.5 * normalized_current_gap:
                print(f"Proposed new cluster-head {bestCandidate.id} for agent {agent.id} at level {level} is not significantly better than current agent")
                continue

            print(f"Rebalancing cluster of agent {agent.id} at level {level} to candidate {bestCandidate.id}")
            token = agent.clusterHeadToken
            token.tokenHolder_ = bestCandidate
            bestCandidate.clusterHeadToken = token
            agent.clusterHeadToken = None

            # add bestCandidate to members of higherCluster (if it exists)
            if token.higherCluster_:
                token.higherCluster_.members_ = [member for member in token.higherCluster_.members_ if member.id != agent.id]
                token.higherCluster_.members_.append(bestCandidate)

            # add agent to members of lvl1 cluster
            if level == 1:
                token.members_ = [member for member in token.members_ if member.id != bestCandidate.id]
                token.members_.append(agent)


        print(f"Concluding cluster rebalance step")
        for agent in self.list_of_agents:
            if agent.clusterHeadToken:
                print(
                    f"Agent {agent.id} has cluster head token with level {agent.clusterHeadToken.level_} and members {[member.id for member in agent.clusterHeadToken.members_]}")


    # this only occurs during the cluster formation phase
    # we need to lift restrictions on the distance from which an agent can join a cluster
    # at most one agent/cluster needs to join
    def cluster_join_step(self):
        print(f"Cluster join step")
        for agent in self.list_of_agents:
            level = agent.clusterHeadToken.level_ if agent.clusterHeadToken else 0
            threshold = self.clusterParams['l1_distance'] * ((level + 1) ** 2)
            agent.get_nearest_neighbors(self.list_of_agents, threshold)
            print(f"agent: {agent.id}, level: {level}, neighbors: {agent.list_of_neighbors}")

        for agent in self.list_of_agents:
            # min_distance_so_far = float('inf')
            # chosen_cluster_head = None
            level = agent.clusterHeadToken.level_ if agent.clusterHeadToken else 0
            if level == 0:
                has_parent = agent.lvl1Cluster is not None
            else:
                has_parent = agent.clusterHeadToken.higherCluster_ is not None

            # skip over agents that have parents
            if has_parent:
                continue

            join_candidates = []
            # we need to check all agents (mandatory join)
            for neighbor_agent in self.list_of_agents:
                neighbor_level = neighbor_agent.clusterHeadToken.level_ if neighbor_agent.clusterHeadToken else 0

                if neighbor_level != level + 1:
                    print(f"Skipping neighbor {neighbor_agent.id} for agent {agent.id} because neighbor level {neighbor_level} is not one higher than agent level {level}")
                    continue
                elif len(neighbor_agent.clusterHeadToken.members_) >= self.clusterParams['max_num_agents'] - 1:
                    print(f"Skipping neighbor {neighbor_agent.id} for agent {agent.id} because neighbor's cluster doesn't have room")
                    continue


                join_candidates.append(neighbor_agent)

            candidate_distances = [(candidate, euclidean(agent.position, candidate.position)) for candidate in join_candidates]
            candidate_distances = sorted(candidate_distances, key=lambda x: x[1])
            # should be guaranteed so no check required
            if len(join_candidates) > 0:
                chosen_cluster_head = candidate_distances[0][0]
                print(f"Chosen cluster head for agent {agent.id} is {chosen_cluster_head.id} at level {level + 1}")
            else:
                print(f"No cluster head found for agent {agent.id} at level {level + 1}")
                continue

            # add agent to the chosen cluster head's members
            chosen_cluster_head.clusterHeadToken.members_.append(agent)
            if level == 0:
                print(f"Agent {agent.id} joining lvl1 cluster head {chosen_cluster_head.id} at level {level + 1}")
                agent.lvl1Cluster = chosen_cluster_head.clusterHeadToken
            else:
                print(f"Agent {agent.id} joining cluster head {chosen_cluster_head.id} at level {level + 1}")
                agent.clusterHeadToken.higherCluster_ = chosen_cluster_head.clusterHeadToken

        print(f"After cluster join step")
        for agent in self.list_of_agents:
            if agent.clusterHeadToken:
                print(f"Agent {agent.id} has cluster head token with level {agent.clusterHeadToken.level_} and members {[member.id for member in agent.clusterHeadToken.members_]}")


    def lvl1_cluster_split_step(self):
        print(f"lvl1 Cluster split step")
        for agent in self.list_of_agents:
            level = agent.clusterHeadToken.level_ if agent.clusterHeadToken else 0
            threshold = self.clusterParams['l1_distance'] * ((level + 1) ** 2)
            agent.get_nearest_neighbors(self.list_of_agents, threshold)
            print(f"agent: {agent.id}, level: {level}, neighbors: {agent.list_of_neighbors}")

        for agent in self.list_of_agents:
            print(f"Processing agent {agent.id} for lvl1 cluster split")
            level = agent.clusterHeadToken.level_ if agent.clusterHeadToken else 0

            # check if cluster is full
            if level != 1:
                print(f"Skipping agent {agent.id} because it is not a level 1 agent")
                continue
            elif not agent.clusterHeadToken.is_full():
                print(f"Skipping agent {agent.id} because its cluster is not full")
                continue

            has_parent = agent.clusterHeadToken.higherCluster_ is not None
            original_centroid = agent.clusterHeadToken.update_centroid()

            # we can't perform a split if the parent cluster is full
            if has_parent and agent.clusterHeadToken.higherCluster_.is_full():
                print(f"Skipping agent {agent.id} because its parent cluster is full")
                continue

            # get all idle members of the cluster
            idle_members = [member for member in agent.clusterHeadToken.members_ if
                                      member.clusterHeadToken is None]

            # if there are not idle members, we can't split the cluster
            # we need at least 2 idle members - one for the new cluster head, and one as a member.
            if len(idle_members) <= 1:
                print(f"Not enough idle members to form a second cluster. Skipping split.")
                continue

            if has_parent and len(idle_members) <= 2:
                print(f"Not enough idle members to join original and second cluster under a new cluster-head. Skipping split.")
                continue

            # pick the agents that are furthest away from the cluster head
            second_cluster_candidatesDistance = [(member, euclidean(agent.position, member.position)) for member in idle_members]
            second_cluster_candidatesDistance = sorted(second_cluster_candidatesDistance, key=lambda x: x[1], reverse=True)

            # print(f"Second cluster candidates for agent {agent.id} at level {level}: {[member.id for member, _ in second_cluster_candidatesDistance]}")
            # print(f"Length of second cluster candidates: {len(second_cluster_candidatesDistance)}")

            if len(idle_members) > 0.5 * self.clusterParams['max_num_agents']:
                second_cluster_size = self.clusterParams['max_num_agents'] // 2
            else:
                second_cluster_size = len(idle_members)

            # print(f"Second cluster size for agent {agent.id} at level {level}: {second_cluster_size}")
            second_cluster = [member for member, _ in second_cluster_candidatesDistance[:second_cluster_size]]
            # print(f"Second cluster for agent {agent.id} at level {level}: {[member.id for member in second_cluster]}")

            # compute centroid of second cluster
            sumLoc = [0, 0]

            # compute centroid of the new cluster and add higher_members to the token
            for member in second_cluster:
                sumLoc = [x + y for x, y in zip(sumLoc, member.position)]


            second_cluster_centroid = [x / second_cluster_size for x in sumLoc]

            second_cluster_distance_from_centroid = [(member, euclidean(second_cluster_centroid, member.position)) for member in second_cluster]
            second_cluster_distance_from_centroid = sorted(second_cluster_distance_from_centroid, key=lambda x: x[1])

            second_cluster_head = second_cluster_distance_from_centroid[0][0]
            second_cluster = [member for member, _ in second_cluster_distance_from_centroid if
                              member.id != second_cluster_head.id]
            print(f"Selected agent {second_cluster_head.id} as the head of the second cluster with members {[member.id for member in second_cluster]} for split cluster of agent {agent.id} at level {level}")

            # remove second cluster head from the original cluster
            # should not do this because second_cluster_head is idle and this takes it out of the level 1 cluster
            # but you are creating a level 1 cluster
            agent.clusterHeadToken.members_ = [member for member in agent.clusterHeadToken.members_ if
                                               member.id != second_cluster_head.id]

            # remove second cluster members from the original cluster
            for member in second_cluster:
                agent.clusterHeadToken.members_ = [m for m in agent.clusterHeadToken.members_ if m.id != member.id]

            # create a new token for the second cluster
            token = ClusterToken(level=level, clusterParams=self.clusterParams)

            # assign the new cluster head token to the second cluster head
            token.tokenHolder_ = second_cluster_head
            second_cluster_head.clusterHeadToken = token
            second_cluster_head.lvl1Cluster = token

            # assign the lvl1 cluster token to each member of the second cluster and add each member to the token
            for member in second_cluster:
                token.members_.append(member)
                member.clusterHeadToken = None
                member.lvl1Cluster = token

            # add the new cluster to the parent cluster
            if agent.clusterHeadToken.higherCluster_ is not None:
                agent.clusterHeadToken.higherCluster_.members_.append(second_cluster_head)
                token.higherCluster_ = agent.clusterHeadToken.higherCluster_
            else:
                # if there is no parent cluster, we need to create one
                idle_members = [member for member in idle_members if member.id != second_cluster_head.id]
                higher_cluster_distance_from_centroid = [(member, euclidean(original_centroid, member.position)) for member in idle_members]
                higher_cluster_distance_from_centroid = sorted(higher_cluster_distance_from_centroid, key=lambda x: x[1])
                higher_cluster_head = higher_cluster_distance_from_centroid[0][0]

                # remove higher cluster head from the original cluster
                # this we should not do because higher_cluster_head is idle and this takes it out of the level 1 cluster
                # old_token = higher_cluster_head.lvl1Cluster
                # old_token.members_ = [member for member in old_token.members_ if member.id != higher_cluster_head.id]

                # create a new token for the higher cluster
                higher_token = ClusterToken(level = level + 1, clusterParams=self.clusterParams)
                higher_token.higherCluster_ = None
                higher_token.members_ = [agent, second_cluster_head]
                higher_token.tokenHolder_ = higher_cluster_head

                # give token to the higher cluster head
                higher_cluster_head.clusterHeadToken = higher_token

                # update the members of the higher cluster
                for member in higher_token.members_:
                    member.clusterHeadToken.higherCluster_ = higher_token

        print(f"After split lvl1 cluster step")
        for agent in self.list_of_agents:
            if agent.clusterHeadToken:
                print(
                    f"Agent {agent.id} has cluster head token with level {agent.clusterHeadToken.level_} and members {[member.id for member in agent.clusterHeadToken.members_]}")


    def cluster_split_step(self):
        # we want to split the cluster so that the average (over the partitions) of the average (within the partition)
        # distance between the centroid and the members is minimized
        def compute_average_over_partition(cluster_one, cluster_two):
            def compute_average_difference(cluster):
                sumLoc = [0, 0]
                for member in cluster:
                    sumLoc = [x + y for x, y in zip(sumLoc, member.position)]
                centroid = [x / len(cluster) for x in sumLoc]

                distance = 0
                for member in cluster_one:
                    distance += euclidean(centroid, member.position)

                return distance / len(cluster)

            centroid_average_difference = compute_average_difference(cluster_one)
            centroid_average_difference = compute_average_difference(cluster_two)
            return 0.5 * (centroid_average_difference + centroid_average_difference)

        print(f"Split cluster step")
        for agent in self.list_of_agents:
            level = agent.clusterHeadToken.level_ if agent.clusterHeadToken else 0
            threshold = self.clusterParams['l1_distance'] * ((level + 1) ** 2)
            agent.get_nearest_neighbors(self.list_of_agents, threshold)
            print(f"agent: {agent.id}, level: {level}, neighbors: {agent.list_of_neighbors}")

        for agent in self.list_of_agents:
            print(f"Processing agent {agent.id} for cluster split")
            level = agent.clusterHeadToken.level_ if agent.clusterHeadToken else 0

            # check if cluster is full
            if level <= 1:
                print(f"Skipping agent {agent.id} because it is a level 2 or higher agent")
                continue
            elif not agent.clusterHeadToken.is_full():
                print(f"Skipping agent {agent.id} because its cluster is not full")
                continue

            has_parent = agent.clusterHeadToken.higherCluster_ is not None
            original_centroid = agent.clusterHeadToken.update_centroid()

            # create the most suitable tentative second cluster
            second_cluster_distance_from_centroid = [(member, euclidean(original_centroid, member.position)) for member in agent.clusterHeadToken.members_]
            second_cluster_distance_from_centroid = sorted(second_cluster_distance_from_centroid, key=lambda x: x[1], reverse=True)
            ordered_members = [member for member, _ in second_cluster_distance_from_centroid]

            second_cluster_size = len(ordered_members) // 2

            first_candidate_second_cluster = ordered_members[:second_cluster_size]
            first_candidate_first_cluster = [member for member in agent.clusterHeadToken.members_ if member not in first_candidate_second_cluster]

            second_candidate_second_cluster = ordered_members[::2]
            second_candidate_first_cluster = [member for member in agent.clusterHeadToken.members_ if member not in second_candidate_second_cluster]

            if compute_average_over_partition(first_candidate_first_cluster, first_candidate_second_cluster) < compute_average_over_partition(second_candidate_first_cluster, second_candidate_second_cluster):
                print(f"Using the worst members from the original cluster for the second cluster")
                second_cluster = first_candidate_second_cluster
            else:
                print(f"Using equal members from the original cluster for the second cluster")
                second_cluster = second_candidate_second_cluster

            print(f"Proposed second cluster for agent {agent.id} at level {level}: {[member.id for member in second_cluster]}")

            # check that each member in the tentative second cluster has at least two idle members
            skinny_check = False
            for member in second_cluster:
                if len(member.clusterHeadToken.get_idle()) < 2:
                    print(f"Member {member.id} in tentative second cluster of agent {agent.id} does not have at least two idle members. Skipping split.")
                    skinny_check = True
                    break

            if skinny_check:
                print(f"Skipping agent {agent.id} because its tentative second cluster is too skinny")
                continue

            # compute centroid of second cluster
            sumLoc = [0, 0]

            # compute centroid of the new cluster and add higher_members to the token
            for member in second_cluster:
                sumLoc = [x + y for x, y in zip(sumLoc, member.position)]

            second_cluster_centroid = [x / second_cluster_size for x in sumLoc]

            # get all idle members of the cluster
            idle_agents = []
            for second_member in second_cluster:
                for member in second_member.clusterHeadToken.get_idle():
                    idle_agents.append(member)

            second_cluster_distance_from_second_centroid = [
                (member, euclidean(second_cluster_centroid, member.position)) for member in idle_agents]
            second_cluster_distance_from_second_centroid = sorted(second_cluster_distance_from_second_centroid, key=lambda x: x[1])

            second_cluster_head = second_cluster_distance_from_second_centroid[0][0]
            # second_cluster = [member for member, _ in second_cluster_distance_from_second_centroid if
            #                   member.id != second_cluster_head.id]
            print(
                f"Selected agent {second_cluster_head.id} as the head of the second cluster with members {[m.id for m in second_cluster]} for split cluster of agent {agent.id} at level {level}")

            # remove second cluster head from the original cluster
            agent.clusterHeadToken.members_ = [member for member in agent.clusterHeadToken.members_ if
                                               member.id != second_cluster_head.id]

            # remove second cluster members from the original cluster
            for member in second_cluster:
                agent.clusterHeadToken.members_ = [m for m in agent.clusterHeadToken.members_ if m.id != member.id]

            # create a new token for the second cluster
            token = ClusterToken(level=level, clusterParams=self.clusterParams)

            # assign the new cluster head token to the second cluster head
            token.tokenHolder_ = second_cluster_head
            second_cluster_head.clusterHeadToken = token

            # assign the cluster token to each member of the second cluster
            for member in second_cluster:
                token.members_.append(member)
                member.clusterHeadToken.higherCluster_ = token

            # add the new cluster to the parent cluster
            if has_parent:
                agent.clusterHeadToken.higherCluster_.members_.append(second_cluster_head)
                token.higherCluster_ = agent.clusterHeadToken.higherCluster_
            else:
                # if there is no parent cluster, we need to create one
                idle_members = second_cluster_head.clusterHeadToken.get_idle()
                higher_cluster_distance_from_centroid = [(member, euclidean(original_centroid, member.position)) for member in idle_members]
                higher_cluster_distance_from_centroid = sorted(higher_cluster_distance_from_centroid, key=lambda x: x[1])
                higher_cluster_head = higher_cluster_distance_from_centroid[0][0]

                # create a new token for the higher cluster
                higher_token = ClusterToken(level = level + 1, clusterParams=self.clusterParams)
                higher_token.higherCluster_ = None
                higher_token.members_ = [agent, second_cluster_head]
                higher_token.tokenHolder_ = higher_cluster_head

                # give token to the higher cluster head
                higher_cluster_head.clusterHeadToken = higher_token

                # update the members of the higher cluster
                for member in higher_token.members_:
                    member.clusterHeadToken.higherCluster_ = higher_token

            self.render()

        print(f"After split cluster step")
        for agent in self.list_of_agents:
            if agent.clusterHeadToken:
                print(
                    f"Agent {agent.id} has cluster head token with level {agent.clusterHeadToken.level_} and members {[member.id for member in agent.clusterHeadToken.members_]}")


    def lvl1_transfer_agents_step(self, cluster_formation = False):
        print(f"lvl1 Transfer agents step")
        # case 1: agent is a level 1 cluster head and member is a higher-level cluster.
        # it must be the case that member is a level 2 or higher cluster head.
        # if the subtree condition is satisfied then agent is a child of member
        # this implies all the siblings of agent are also children of member
        # so we can transfer member to a sibling of agent without violating the subtree condition.
        def find_root():
            # find the root of the cluster tree
            level = 0
            root_agent = None
            for agent in self.list_of_agents:
                if agent.clusterHeadToken is None:
                    continue
                if agent.clusterHeadToken.level_ > level:
                    level = agent.clusterHeadToken.level_
                    root_agent = agent

            return root_agent

        def find_suitable_lvl1_cousin(member, agent):
            parent_of_member = member.lvl1Cluster.tokenHolder_
            # member always has level 0
            level_of_parent_of_member = 1
            # find_suitable_cousin is only invoked if agent has a parent
            # we don't need to check that here (it's already been checked in the previous invocation)
            parent_of_agent = agent.clusterHeadToken.higherCluster_.tokenHolder_

            # get all the level 1 cousins of member
            # in the cluster formation phase we look for the best level 1 cluster to transfer
            # in the cluster maintenance phase we prioritize clusters that are cousins
            if cluster_formation:
                root = find_root()
                cousins = root.clusterHeadToken.get_lvlX_members(level_of_parent_of_member)
            else:
                cousins = parent_of_agent.clusterHeadToken.get_lvlX_members(level_of_parent_of_member)
            # else:

            # remove the parent of member from the list of cousins and any cousins that are already full (if needed)
            cousins = [cousin for cousin in cousins if
                       cousin.id != parent_of_member.id and not cousin.clusterHeadToken.is_full()]

            # no cousins were found so now we need to decide whether to escalate
            if len(cousins) == 0:
                # we check escalation levels
                level_of_agent = agent.clusterHeadToken.level_
                current_escalation_level = level_of_agent - level_of_parent_of_member

                # we check if agent has a grandparent
                # now we check if agent has a parent before escalating
                grand_parent_exists = parent_of_agent.clusterHeadToken.higherCluster_ is not None

                # we have exceeded the max number of escalation levels
                if current_escalation_level > self.clusterParams['escalation_levels']:
                    print(
                        f"Exceeded max escalation levels for agent {agent.id} and member {member.id}. No suitable cousin found.")
                    return None
                elif not grand_parent_exists:
                    print(
                        f"No grandparent exists for agent {agent.id} and member {member.id}. No suitable cousin found.")
                    return None

                return find_suitable_lvl1_cousin(member, parent_of_agent)
            else:
                return cousins

        for agent in self.list_of_agents:
            level = agent.clusterHeadToken.level_ if agent.clusterHeadToken else 0
            threshold = self.clusterParams['l1_distance'] * ((level + 1) ** 2)
            agent.get_nearest_neighbors(self.list_of_agents, threshold)
            print(f"agent: {agent.id}, level: {level}, neighbors: {agent.list_of_neighbors}")

        # iterate through all the agents and stop at agents that are eligible for transfer
        for agent in self.list_of_agents:
            level = agent.clusterHeadToken.level_ if agent.clusterHeadToken else 0

            # skip over agents that are level 0 because cluster-heads manage transfers
            if level != 1:
                print(f"Skipping agent {agent.id} because it is not a level 1 agent")
                continue
            elif agent.clusterHeadToken.higherCluster_ is None:
                print(f"Skipping transfer because agent {agent.id} has no parent cluster")
                continue

            # find all the outliers and pick the worst ones first
            members_distance = [(member, euclidean(agent.position, member.position)) for member in
                                agent.clusterHeadToken.members_]
            members_distance = sorted(members_distance, key=lambda x: x[1], reverse=True)

            outliers = [member for member, distance in members_distance if
                        self.grid_dim * distance > self.clusterParams['l1_distance'] * (level ** 2)]

            # at a level 1 cluster, there are two tyes of outliers: idle members and higher-level clusterheads
            # idle members are fine (no problem)
            # higher-level clusterheads are a problem because the parent of the cluster-head is the level 1 parent not the higher-level parent
            for member in outliers:
                # get distance between agent and member
                # distance = self.grid_dim * chebyshev_distance(agent.position, member.position)
                # if distance < self.clusterParams['l1_distance'] * (level ** 2):
                #     print(f"Skipping member {member.id} of agent {agent.id} because distance {distance}it is within the radius {self.clusterParams['l1_distance'] * (level ** 2)}")
                #     continue

                print(f"Attempting to transfer member {member.id} through agent {agent.id} at level {level}")
                cousins = find_suitable_lvl1_cousin(member, agent)

                if cousins is None:
                    print(f"No suitable cousins found for member {member.id} of agent {agent.id}. Skipping transfer.")
                    continue

                cousin_distances = [(cousin, euclidean(cousin.position, member.position)) for cousin in cousins]
                cousin_distances = sorted(cousin_distances, key=lambda x: x[1])

                best_cousin = cousin_distances[0][0]

                print(
                    f"Distance from best cousin {best_cousin.id} to member {member.id} is {euclidean(best_cousin.position, member.position)}")
                print(
                    f"Distance from agent {agent.id} to member {member.id} is {euclidean(agent.position, member.position)}")

                if euclidean(best_cousin.position, member.position) > euclidean(agent.position, member.position):
                    print(
                        f"Best cousin {best_cousin.id} for member {member.id} of agent {agent.id} is further away than agent. Skipping transfer.")
                    continue

                agent.clusterHeadToken.members_ = [m for m in agent.clusterHeadToken.members_ if
                                                              m.id != member.id]

                # we need to add member to the best cousin's cluster members_
                best_cousin.clusterHeadToken.members_.append(member)
                member.lvl1Cluster = best_cousin.clusterHeadToken

                print(f"Transferring member {member.id} from agent {agent.id} to cousin {best_cousin.id}")

                # we need to worry about transferring a member from a cluster that only has one member
                # I am going to allow empty lvl1 clusters but no empty higher-level clusters
                if len(agent.clusterHeadToken.members_) == 0:
                    # do nothing and wait to get assimilated or absorb a new agent
                    print(f"Leaving agent {agent.id} with empty lvl1 cluster token")

        print(f"After transfer agents step")
        for agent in self.list_of_agents:
            if agent.clusterHeadToken:
                print(
                    f"Agent {agent.id} has cluster head token with level {agent.clusterHeadToken.level_} and members {[member.id for member in agent.clusterHeadToken.members_]}")


    def transfer_agents_step(self, cluster_formation = False):
        print(f"Transfer agents step")
        # the subtree condition: the lvl1 cluster of a higher-level agent must always be in the subtree of the agent.

        # case 2: agent is a level k cluster head and member is a level k-1 cluster.
        # do we need to worry about member violating the subtree condition after a transfer?
        # no because if member satisfies the subtree condition, then its level 1 cluster will still be in its
        # subtree after transferring.  the subtree moves along with member.
        # so that means we only need to worry about agent
        # if the lvl1 cluster of agent is not in the subtree of member, then we can transfer member without
        # worrying about the subtree condition.
        # what happens if the lvl1 cluster of agent IS in the subtree of member?
        # here things get a little tricky
        # transferring member to a sibling of agent removes the lvl1 cluster of agent from its subtree.
        # the subtree of a sibling of agent is not in the subtree of agent.
        # however we can transfer member to a child of agent without violating the subtree condition.
        # the subtree of a child of agent is still in the subtree of agent.

        # case 3: agent is a level k cluster and member is a level k-m cluster where m > 1.
        # don't need to worry about member violating the subtree condition after transfer
        # if the lvl1 cluster of agent is not in the subtree of member then we can transfer member without
        # worrying if agent will violate with subtree condition
        # if the lvl1 cluster of agent IS in the subtree of member, then transferring member to a sibling subtree of agent
        # will violate the subtree condition because the lvl1 cluster of agent will no longer be in the subtree of agent
        # we can transfer member to a child of agent without violating the subtree condition

        # case 4: agent is a level k cluster and member is a level k+m cluster.
        # the lvl1 cluster of member is in its subtree (a priori assumption)
        # this implies the lvl1 cluster of member is in the subtree of agent (we haven't escalated up to level k+m yet)
        # therefore we can transfer member to any sibling of agent without violating the subtree condition at member,
        # since agent is also in the subtree of member.
        # suppose the lvl1 cluster of agent is in the subtree of agent.  Clearly member is not the lvl1 cluster head of agent
        # because member is a level k+m cluster.
        # therefore moving member to a sibling of agent will not violate the subtree condition at agent

        # so first we check if member is a lower level than agent (if not we don't need to worry about the subtree condition)
        # if member is a lower level than agent we check if the lvl1 cluster of agent is in the subtree of member
        # if the answer is no we don't need to worry about the subtree condition
        # if the answer is yes, then we will not be able to transfer member to a sibling of agent
        # we can transfer member to another child of agent though.

        # this is only invoked if the subtree condition is satisfied
        def find_root():
            # find the root of the cluster tree
            level = 0
            root_agent = None
            for agent in self.list_of_agents:
                if agent.clusterHeadToken is None:
                    continue
                if agent.clusterHeadToken.level_ > level:
                    level = agent.clusterHeadToken.level_
                    root_agent = agent

            return root_agent

        def find_suitable_cousin(member, agent):
            # suppose member is level l (member's parent has level l+1)
            # we need to get all the level l+1 members in the subtree of agent's parent.
            # member's parent might not be agent (agent might be a higher ancestor)

            # member might be idle (no clusterHeadToken)
            # there is no way that member is idle.  It must have a clusterHeadToken
            parent_of_member = member.clusterHeadToken.higherCluster_.tokenHolder_

            # parent_of_member = member.clusterHeadToken.higherCluster_.tokenHolder_

            level_of_parent_of_member = parent_of_member.clusterHeadToken.level_
            # find_suitable_cousin is only invoked if agent has a parent
            # we don't need to check that here (it's already been checked in the previous invocation)
            parent_of_agent = agent.clusterHeadToken.higherCluster_.tokenHolder_

            # get all the level x cousins of member
            # cousins = parent_of_agent.clusterHeadToken.get_lvlX_members(level_of_parent_of_member)

            if cluster_formation:
                root = find_root()
                cousins = root.clusterHeadToken.get_lvlX_members(level_of_parent_of_member)
            else:
                cousins = parent_of_agent.clusterHeadToken.get_lvlX_members(level_of_parent_of_member)

            # remove the parent of member from the list of cousins and any cousins that are already full (if needed)
            cousins = [cousin for cousin in cousins if
                       cousin.id != parent_of_member.id and not cousin.clusterHeadToken.is_full()]

            # no cousins were found so now we need to decide whether to escalate
            if len(cousins) == 0:
                # we check escalation levels
                level_of_agent = agent.clusterHeadToken.level_
                current_escalation_level = level_of_agent - level_of_parent_of_member

                # we check if the subtree condition is satisfied
                # level_of_member is not necessarily level_of_parent_of_member - 1 (in a level 1 cluster some members might have higher levels)
                # also a member could belong to two clusters, a lvl1 cluster and a higher-level cluster
                # level of member is always level_of_parent_of_member because level 1 clusters are handles separately
                level_of_member = level_of_parent_of_member - 1

                lvl1_parent_for_agent = agent.lvl1Cluster.tokenHolder_
                if level_of_member < level_of_agent and level_of_member > 0:
                    subtree_condition = lvl1_parent_for_agent in member.clusterHeadToken.get_all_members()
                else:
                    subtree_condition = False

                # we check if agent has a grandparent
                # now we check if agent has a parent before escalating
                grand_parent_exists = parent_of_agent.clusterHeadToken.higherCluster_ is not None

                # we have exceeded the max number of escalation levels
                if current_escalation_level > self.clusterParams['escalation_levels']:
                    print(f"Exceeded max escalation levels for agent {agent.id} and member {member.id}. No suitable cousin found.")
                    return None
                # the subtree condition will not be satisfied if we escalate further
                elif not subtree_condition:
                    print(f"Subtree condition not satisfied for agent {agent.id} and member {member.id}. No suitable cousin found.")
                    return None
                # we have reached the top of the tree, can't escalate further
                elif not grand_parent_exists:
                    print(f"No grandparent exists for agent {agent.id} and member {member.id}. No suitable cousin found.")
                    return None

                return find_suitable_cousin(member, parent_of_agent)

            else:

                return cousins

        for agent in self.list_of_agents:
            level = agent.clusterHeadToken.level_ if agent.clusterHeadToken else 0
            threshold = self.clusterParams['l1_distance'] * ((level + 1) ** 2)
            agent.get_nearest_neighbors(self.list_of_agents, threshold)
            print(f"agent: {agent.id}, level: {level}, neighbors: {agent.list_of_neighbors}")

        # iterate through all the agents and stop at agents that are eligible for transfer
        for agent in self.list_of_agents:
            level = agent.clusterHeadToken.level_ if agent.clusterHeadToken else 0

            # skip over agents that are level 0 because cluster-heads manage transfers
            if level <= 1:
                print(f"Skipping agent {agent.id} because it is a level {level} agent")
                continue
            elif agent.clusterHeadToken.higherCluster_ is None:
                print(f"Skipping transfer because agent {agent.id} has no parent cluster")
                continue

            list_of_neighbors = [self.list_of_agents[agent_id] for agent_id in agent.list_of_neighbors]
            candidates = [member for member in list_of_neighbors if member.clusterHeadToken is not None and member.clusterHeadToken.level_ == level]

            if len(candidates) == 0:
                print(f"No candidates found for agent {agent.id} at level {level}. Skipping transfer.")
                continue

            outliers = []
            for member in agent.clusterHeadToken.members_:
                member_distance = []
                for candidate in candidates:
                    member_distance.append((member, euclidean(member.position, candidate.position)))

                member_distance = sorted(member_distance, key=lambda x: x[1])
                if member_distance[0][1] < euclidean(member.position, agent.position):
                    outliers.append(member)

            if len(outliers) == 0:
                print(f"No outliers found for agent {agent.id} at level {level}. Skipping transfer.")
                continue

            outlier_distance = [(member, euclidean(agent.position, agent.position)) for member in outliers]
            outlier_distance = sorted(outlier_distance, key=lambda x: x[1], reverse=True)
            outliers = [member for member, distance in outlier_distance]

            print(f"Outliers for agent {agent.id} at level {level}: {[member.id for member in outliers]}")

            # at a level 1 cluster, there are two tyes of outliers: idle members and higher-level clusterheads
            # idle members are fine (no problem)
            # higher-level clusterheads are a problem because the parent of the cluster-head is the level 1 parent not the higher-level parent
            for member in outliers:
                # get distance between agent and member
                # distance = self.grid_dim * chebyshev_distance(agent.position, member.position)
                # if distance < self.clusterParams['l1_distance'] * (level ** 2):
                #     print(f"Skipping member {member.id} of agent {agent.id} because distance {distance}it is within the radius {self.clusterParams['l1_distance'] * (level ** 2)}")
                #     continue

                print(f"Attempting to transfer member {member.id} through agent {agent.id} at level {level}")
                cousins = find_suitable_cousin(member, agent)

                if cousins is None:
                    print(f"No suitable cousins found for member {member.id} of agent {agent.id}. Skipping transfer.")
                    continue

                cousin_distances = [(cousin, euclidean(cousin.position, member.position)) for cousin in cousins]
                cousin_distances = sorted(cousin_distances, key=lambda x: x[1])

                best_cousin = cousin_distances[0][0]

                print(f"Distance from best cousin {best_cousin.id} to member {member.id} is {euclidean(best_cousin.position, member.position)}")
                print(f"Distance from agent {agent.id} to member {member.id} is {euclidean(agent.position, member.position)}")

                if euclidean(best_cousin.position, member.position) > euclidean(agent.position, member.position):
                    print(f"Best cousin {best_cousin.id} for member {member.id} of agent {agent.id} is further away than agent. Skipping transfer.")
                    continue

                # we need to move member to the best cousin's cluster
                # how do we do this?
                # we need to remove member from the parent of member's cluster
                agent.clusterHeadToken.members_ = [m for m in agent.clusterHeadToken.members_ if m.id != member.id]

                # we need to add member to the best cousin's cluster members_
                best_cousin.clusterHeadToken.members_.append(member)

                # we need to link members token to best cousin's cluster token (if member has a higher-level token)
                member.clusterHeadToken.higherCluster_ = best_cousin.clusterHeadToken

                print(f"Transferring member {member.id} from agent {agent.id} to cousin {best_cousin.id}")

                # we need to worry about transferring a member from a cluster that only has one member
                # I am going to allow empty lvl1 clusters but no empty higher-level clusters
                if len(agent.clusterHeadToken.members_) == 0:
                    # remove parent_of_member from it's parent cluster (which is guaranteed to exist)
                    print(f"Destroying higher-level token for {agent.id} because it has no members left")
                    parent_of_agent = agent.clusterHeadToken.higherCluster_.tokenHolder_
                    parent_of_agent.clusterHeadToken.members_ = [m for m in parent_of_agent.clusterHeadToken.members_ if m.id != agent.id]
                    # we need to destroy the parent of member's cluster token
                    agent.clusterHeadToken = None

        print(f"After transfer agents step")
        for agent in self.list_of_agents:
            if agent.clusterHeadToken:
                print(
                    f"Agent {agent.id} has cluster head token with level {agent.clusterHeadToken.level_} and members {[member.id for member in agent.clusterHeadToken.members_]}")


    def mop_up_step(self):
        # returns a boolean(whether or not all agents are united under a single hierarchical tree) and the level of the tree
        for agent in self.list_of_agents:
            level = agent.clusterHeadToken.level_ if agent.clusterHeadToken else 0
            threshold = self.clusterParams['l1_distance'] * ((level + 1) ** 2)
            agent.get_nearest_neighbors(self.list_of_agents, threshold)
            print(f"agent: {agent.id}, level: {level}, neighbors: {agent.list_of_neighbors}")

        # first link up the level 0 agents with level 1 clusters
        for agent in self.list_of_agents:
            if agent.lvl1Cluster is not None:
                print(f"Agent {agent.id} is already in a level 1 cluster. Skipping mop-up.")
                continue

            print(f"Finding best level 1 cluster head for agent {agent.id}")
            # get all the level 1 cluster heads with clusters that are not full
            cluster_heads = [a for a in self.list_of_agents if a.clusterHeadToken is not None]
            lvl1_cluster_heads = [ch for ch in cluster_heads if ch.clusterHeadToken.level_ == 1 and not ch.clusterHeadToken.is_full()]

            lvl1_cluster_head_distances = [(ch, euclidean(agent.position, ch.position)) for ch in lvl1_cluster_heads]
            lvl1_cluster_head_distances = sorted(lvl1_cluster_head_distances, key=lambda x: x[1])

            if len(lvl1_cluster_heads) == 0:
                print(f"No available level 1 cluster heads for agent {agent.id}.")
                break

            best_lvl1_candidate = lvl1_cluster_head_distances[0][0]

            print(f"Best level 1 cluster head for agent {agent.id} is {best_lvl1_candidate.id}")
            # add agent to the best lvl1 cluster head's members
            best_lvl1_candidate.clusterHeadToken.members_.append(agent)
            agent.lvl1Cluster = best_lvl1_candidate.clusterHeadToken

        print(f"After linking level 0 agents with level 1 clusters")
        # check if all agents are connected and get the highest level of the tree
        all_connected, highest_level = self.are_agents_connected()

        if all_connected:
            print(f"All agents are connected under a single hierarchical tree with highest level {highest_level}.")
            return
        else:
            print(f"Not all agents are connected. Highest level of the tree is {highest_level}.")

        for level in range(1, highest_level):
            # get all agents with cluster head tokens at this level that don't have a parent cluster
            orphan_agents = []
            for agent in self.list_of_agents:
                if agent.clusterHeadToken is not None and agent.clusterHeadToken.level_ == level and agent.clusterHeadToken.higherCluster_ is None:
                    orphan_agents.append(agent)

            # match orphan agents with the best cluster head at the next level
            available_cluster_heads = []
            for agent in self.list_of_agents:
                if agent.clusterHeadToken is not None and agent.clusterHeadToken.level_ == level + 1 and not agent.clusterHeadToken.is_full():
                    available_cluster_heads.append(agent)

            if len(available_cluster_heads) == 0:
                print(f"No available cluster heads at level {level + 1}. Cannot perform mop-up.")
                continue

            for orphan_agent in orphan_agents:
                agent_distance = [(ch, euclidean(orphan_agent.position, ch.position)) for ch in available_cluster_heads]
                sorted_agent_distance = sorted(agent_distance, key=lambda x: x[1])
                # if len(sorted_agent_distance) == 0:
                #     print(f"No available cluster heads for orphan agent {orphan_agent.id} at level {level}.")
                #     continue
                # else:
                best_candidate = sorted_agent_distance[0][0]

                print(f"Best cluster head for orphan agent {orphan_agent.id} at level {level} is {best_candidate.id}")
                best_candidate.clusterHeadToken.members_.append(orphan_agent)
                orphan_agent.clusterHeadToken.higherCluster_ = best_candidate.clusterHeadToken

                if best_candidate.clusterHeadToken.is_full():
                    print(f"Cluster head {best_candidate.id} is now full. Removing from available cluster heads.")
                    available_cluster_heads.remove(best_candidate)

                if len(available_cluster_heads) == 0:
                    print(f"No more available cluster heads at level {level + 1}.")
                    break

        print(f"After mop-up step")
        for agent in self.list_of_agents:
            if agent.clusterHeadToken:
                print(
                    f"Agent {agent.id} has cluster head token with level {agent.clusterHeadToken.level_} and members {[member.id for member in agent.clusterHeadToken.members_]}")


    def render(self):
        fig, ax = plt.subplots(figsize=(5,5), dpi=300)

        display_targets(ax, self.target_positions, self.target_mask, self.grid_dim)

        for x_cell, y_cell in itertools.product(range(self.grid_dim), range(self.grid_dim)):
            if self.grid_state[x_cell, y_cell] == 1:
                display_obstacles(ax, (x_cell / self.grid_dim, y_cell / self.grid_dim), self.grid_dim)

        display_grid(ax, self.grid_dim)

        display_cluster_lines(ax, self.list_of_agents, self.grid_dim)

        # need to iterate through all of the agents
        display_agents(ax, self.list_of_agents, self.grid_dim)

        display_cluster_heads(ax, self.list_of_agents, self.grid_dim)

        plt.show()

        # for agent in self.list_of_agents:
        #     print(f"agent: {agent.id}, neighbors: {agent.list_of_neighbors}")

        print(f"Rendering")
        for agent in self.list_of_agents:
            if agent.clusterHeadToken:
                print(f"Agent {agent.id} has cluster head token with level {agent.clusterHeadToken.level_} and members {[member.id for member in agent.clusterHeadToken.members_]}")

        # input("Press any key to continue...")


    def cluster_render(self):
        sorted_agent_list = sorted(self.list_of_agents, key=lambda agent: agent.get_level())
        for agent in sorted_agent_list:
            token = agent.clusterHeadToken
            if token:
                for member in agent.clusterHeadToken.members_:
                    self.plotter.add_lines(agent, member, token.level_)

                if token.tokenHolder_.id == agent.id:
                    self.plotter.plot_agent(agent, token.level_)
            else:
                self.plotter.plot_agent(agent, 0)  # None)

            # agent.nearest_neighbors(agents,threshold=clusterConstraints['l1_distance'])
        # for target in self.targets.target_list:
        #     if target.id not in self.cleared_targets:
        #         self.plotter.plot_target(target)

        # if len(self.cleared_targets) != 0:
        #     for target in self.cleared_targets:
        #         self.plotter.plot_cleared_target(self.targets.full_target_list[target])

        # self.plotter.display()

        # if self.path:
        #     if len(str(self.i)) == 1:
        #         ip = f'00{self.i}'
        #     elif len(str(self.i)) == 2:
        #         ip = f'0{self.i}'
        #     else:
        #         ip = f'{self.i}'
        #     self.plotter.save(f'point_plot{ip}.png', path=self.path)


    def reset(self):
        self.grid_state = np.zeros((self.grid_dim, self.grid_dim))
        self.obstacle_positions = np.empty((0, 2))
        self.place_obstacles()

        self.agent_positions, self.target_positions = self.init_agents_and_targets_positions()
        self.reset_agents()

        self.target_mask = [0 for _ in range(self.num_targets)]
