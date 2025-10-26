from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import math
from ortools.graph.python import min_cost_flow
import numpy as np

from agent_and_target_clustering.additional_functions import euclidean
from astar import a_star

from grid_world_env import gridworld_env
# from grid_world_env import gridworld_env


class ClusteringTSP:
    def __init__(self, env):
        self.env = env

    def get_pairwise_distance_and_path_dict(self, agent):
        grid = self.env.grid_state
        pairwise_distances = {}
        pairwise_paths = {}
        pairwise_action_sequences = {}

        cluster_members = [member for member in agent.clusterHeadToken.members_]
        cluster_members.append(agent)  # Include the agent itself in the cluster

        self.env.update_agents_positions()
        agent_positions = [self.env.agent_positions[member.id] for member in cluster_members]

        # Filter out visited targets using the target_mask
        unvisited_target_positions = [self.env.target_positions[i] for i in agent.assigned_target_ids]

        # Convert agent and unvisited target positions to grid cells
        agent_positions = [self.env.convert_position_to_grid_cell(pos) for pos in agent_positions]
        target_positions = [self.env.convert_position_to_grid_cell(pos) for pos in unvisited_target_positions]

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
                    pairwise_action_sequences[(j, i)] = self.env.invert_actions(action_sequence)
                else:
                    distance = float('inf')
                    pairwise_paths[(i, j)] = None
                    pairwise_paths[(j, i)] = None
                    pairwise_action_sequences[(i, j)] = None
                    pairwise_action_sequences[(j, i)] = None

                pairwise_distances[(i, j)] = distance
                pairwise_distances[(j, i)] = distance

        return pairwise_distances, pairwise_paths, pairwise_action_sequences, positions

    def create_distance_matrix(self, agent):
        """Create a distance matrix for the locations."""
        # visited targets are not included in the distance matrix
        pairwise_distances, _, _, positions = self.get_pairwise_distance_and_path_dict(agent)
        return pairwise_distances, positions


    def convert_dict_to_matrix(self, distance_dict):
        """Convert a distance dictionary to a 2D distance matrix."""
        # Determine the size of the matrix
        max_node = max(max(key) for key in distance_dict.keys())
        size = max_node + 1  # Nodes are 0-indexed

        # Initialize a 2D matrix with zeros
        distance_matrix = [[0] * size for _ in range(size)]

        # Populate the matrix with values from the dictionary
        for (i, j), distance in distance_dict.items():
            distance_matrix[i][j] = distance

        return distance_matrix


    def add_dummy_node_to_matrix(self, distance_matrix):
        """Add a dummy node with zero distance to all other nodes in a distance matrix."""
        size = len(distance_matrix)
        dummy_node = size  # Use the next integer as the dummy node index

        # Add a new row and column for the dummy node
        for i in range(size):
            distance_matrix[i].append(0)  # Add dummy node column

        distance_matrix.append([0] * (size + 1))  # Add dummy node row

        # Set the distance from the dummy node to all other nodes to zero
        # Set distances from the dummy node to all other nodes to zero
        for i in range(size + 1):
            distance_matrix[dummy_node][i] = 0
            distance_matrix[i][dummy_node] = 0

        return distance_matrix


    def get_routes_as_dict(self, manager, routing, solution, num_agents):
        """Extract routes for each agent and return as a dictionary."""
        routes = {}
        for agent_id in range(num_agents):
            index = routing.Start(agent_id)
            route = []
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            routes[agent_id] = route[1:]
            routes[agent_id] = [index - num_agents for index in route[1:]]
        return routes

    def solve_open_ended_vrp(self, agent):
        """Solve a multi-agent TSP with open-ended paths using OR-Tools."""

        # distance_matrix = create_distance_matrix(locations)
        distance_dict, positions = self.create_distance_matrix(agent)
        # print(f"distance_dict: {distance_dict}")
        # Convert the distance dictionary to a 2D matrix
        distance_matrix = self.convert_dict_to_matrix(distance_dict)
        # print(f"distance_matrix: {distance_matrix}")
        # Add a dummy node with zero distance to all other nodes
        distance_matrix = self.add_dummy_node_to_matrix(distance_matrix)

        # print(f"positions: {positions}")
        # print(f"distance_matrix: {distance_matrix}")

        num_locations = len(distance_matrix)
        num_agents = len(agent.clusterHeadToken.members_) + 1

        # Agents start at the first num_agents locations
        start_indices = list(range(num_agents))
        end_indices = [num_locations - 1] * num_agents  # Dummy node as end location

        # For now, set same end locations; we'll override this later
        # end_indices = list(range(num_agents))

        # Create routing manager and model
        manager = pywrapcp.RoutingIndexManager(num_locations, num_agents, start_indices, end_indices)
        routing = pywrapcp.RoutingModel(manager)

        # Distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add a distance dimension to minimize route length
        dimension_name = "Distance"
        routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            100000,  # allow very long routes
            True,  # start cumul to zero
            dimension_name
        )
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)

        # Trick: Allow routes to end anywhere by adding "disjunctions"
        # for node in range(num_agents, num_locations):
        #     index = manager.NodeToIndex(node)
        #     routing.AddDisjunction([index], 0)  # All targets must be visited

        # Search strategy
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.time_limit.seconds = 10

        # Solve
        solution = routing.SolveWithParameters(search_parameters)

        return distance_matrix, num_agents, manager, routing, solution


    def get_cluster_capacity(self, env, list_of_lvl1_cluster_heads, logging):
        num_targets = env.num_targets
        num_agents = env.num_agents
        num_targets_per_agent = 2 * num_targets / num_agents

        cluster_capacities = []
        for agent in list_of_lvl1_cluster_heads:
            cluster_size = len(agent.clusterHeadToken.members_) + 1  # +1 to include the cluster head itself
            cluster_capacity = math.ceil(cluster_size * num_targets_per_agent)
            cluster_capacities.append(cluster_capacity)


        return cluster_capacities


    def assign_targets_to_lvl1_clusters(self, env, logging):
        list_of_agents = env.list_of_agents
        list_of_lvl1_cluster_heads = [agent for agent in list_of_agents if agent.is_lvl1_cluster_head()]

        unvisited_targets = [target_id for target_id, visited in enumerate(env.target_mask) if not visited]

        n_c = len(list_of_lvl1_cluster_heads)
        n_t = len(unvisited_targets)

        cluster_capacities = self.get_cluster_capacity(env, list_of_lvl1_cluster_heads, logging)

        # Node indexing
        # 0 ................ n_t-1 : target nodes
        # n_t .............. n_t+n_c-1 : cluster head nodes
        # n_t+n_c : source node, t = n_t + n_c + 1 : sink node
        S = n_t + n_c
        T = S + 1
        mcf = min_cost_flow.SimpleMinCostFlow()

        # Add target nodes with supply of 1
        def add_arc(u, v, cap, cost):
            mcf.add_arc_with_capacity_and_unit_cost(u, v, int(cap), int(round(cost)))

        # source -> targets (cap=1, cost=0)
        for i in range(n_t):
            add_arc(S, i, 1, 0)

        # targets -> clusters (cap=1, cost=euclidean distance)
        for i in range(n_t):
            for k in range(n_c):
                cluster_head = list_of_lvl1_cluster_heads[k]
                cluster_centroid = cluster_head.clusterHeadToken.update_centroid()
                target_id = unvisited_targets[i]
                cost = euclidean(env.target_positions[target_id], cluster_centroid)
                scaled_cost = env.grid_dim * cost # has to be an integer so a cost between 0 and 1 is no good
                # target_position = env.target_positions[i]
                # formatted_target_position = [np.round(target_position[0], 2), np.round(target_position[1], 2)]
                # cluster_position = cluster_head.position
                # formatted_cluster_position = [np.round(cluster_position[0], 2), np.round(cluster_position[1], 2)]
                # formatted_cluster_centroid = [np.round(cluster_centroid[0], 2), np.round(cluster_centroid[1], 2)]

                # logging.info(f'Cost from target {i}:{formatted_target_position} to cluster {cluster_head.id}:{formatted_cluster_position} with centroid {formatted_cluster_centroid} is {scaled_cost}')
                add_arc(i, n_t + k, 1, scaled_cost)

        # clusters -> sink (cap=cluster_capacity, cost=0)
        for k in range(n_c):
            logging.info(f'Cluster head {list_of_lvl1_cluster_heads[k].id} has capacity {cluster_capacities[k]}')
            add_arc(n_t + k, T, cluster_capacities[k], 0)

        # Supplies: +1 per target, -n_t at sink
        supplies = [0]*(T+1)
        supplies[S] = n_t
        supplies[T] = -n_t

        for node, sup in enumerate(supplies):
            mcf.set_node_supply(node, sup)
            # mcf.SetNodeSupply(node, sup)

        status = mcf.solve()
        # status = mcf.Solve()
        if status == mcf.INFEASIBLE:
            logging.warning('No feasible solution.  Check supplies/capacities.')
            return None
        elif status == mcf.UNBALANCED:
            logging.warning('Supplies and demands dont add up.')
            return None
        elif status == mcf.OPTIMAL:
            logging.info('Minimum cost flow solved optimally.')
        if status != mcf.OPTIMAL:
            logging.warning('There was an issue with the min cost flow input.')
            return None

        # assign = [-1]*n_t
        for a in range(mcf.num_arcs()):
            u, v = mcf.tail(a), mcf.head(a)
            if 0 <= u < n_t and n_t <= v < n_t + n_c:
                if mcf.flow(a) == 1:
                    cluster_head_index = v - n_t
                    cluster_head = list_of_lvl1_cluster_heads[cluster_head_index]
                    target_id = unvisited_targets[u]
                    cluster_head.assigned_target_ids.append(target_id)
                    # assign[u] = v - n_t
                    # logging.info(f'Target {u} assigned to cluster head {cluster_head.id})')

        return list_of_lvl1_cluster_heads






