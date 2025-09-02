import time
from additional_functions import euclidean
from clustering_route_planner import ClusteringTSP
from params import allParams
from CBSsolver import CBS
from grid_world_env import gridworld_env
import numpy as np
from buffers import MultiAgentPathPlanningBuffer
import multiprocessing
import pickle
import os
import math

import logging
from logging.handlers import RotatingFileHandler


# Assuming `env`, `route_solver`, `CBS`, and `path_planning_buf` are already initialized

# what are the observations?
# the positions of the agents, the positions of the targets, the positions and length of the obstacles

# initalize route planner


def find_first_target_reached_index(paths, targets):
    """
    Find the first index (in a path vector of agent positions) where an agent's position coincides with its target position.

    Args:
        paths (list of list of tuples): List of paths for all agents.
        targets (list of tuples): List of target positions for all agents.

    Returns:
        list: A list of indices where each agent reaches its target, or -1 if not reached.
    """
    indices = []
    for path, target in zip(paths, targets):
        index = next((i for i, position in enumerate(path) if position == target), -1)
        indices.append(index)
    return indices


def create_prefix_action_sequences(action_sequences, smallest_index):
    """
    Create a new action_sequence for each agent that is the prefix up to the smallest_index.

    Args:
        action_sequences (list of list of int): Original action sequences for all agents.
        smallest_index (int): The index up to which the prefix is taken.

    Returns:
        list of list of int: New action sequences with the prefix up to smallest_index.
    """
    if smallest_index == 0:
        raise ValueError("smallest_index must be greater than 0 to create a prefix action sequence.")
    else:
        new_action_sequences = [
            actions[:smallest_index] if smallest_index <= len(actions) else actions
            for actions in action_sequences
        ]

    return new_action_sequences

def find_current_assigned_cluster(target_id, list_of_lvl1_cluster_heads):
    for agent in list_of_lvl1_cluster_heads:
        if target_id in agent.assigned_target_ids:
            return agent.id

    return None

def distance_to_closest_target(agent, list_of_target_ids, list_of_target_positions):
    best_distance = float("inf")
    for target_id in list_of_target_ids:
        target_position = list_of_target_positions[target_id]
        distance = euclidean(agent.position, target_position)
        if distance < best_distance:
            best_distance = distance
            # closest_target_id = target_id

    return best_distance

def distance_to_closest_agent(target_position, target_cluster, list_of_agent_positions):
    best_distance = float("inf")
    list_of_agent_ids = [member.id for member in target_cluster.clusterHeadToken.members_]
    list_of_agent_ids.append(target_cluster.id)
    for agent_id in list_of_agent_ids:
        agent_position = list_of_agent_positions[agent_id]
        distance = euclidean(agent_position, target_position)
        if distance < best_distance:
            best_distance = distance
            # closest_agent_id = agent_id

    return best_distance


def assign_targets_to_lvl1_clusters_OR_tools(env, route_solver, logging):
    list_of_lvl1_cluster_heads = route_solver.assign_targets_to_lvl1_clusters(env, logging)

    # if the target was previously assigned to another cluster, see if it should be reassigned
    # hysteresis to prevent oscillations
    for agent in list_of_lvl1_cluster_heads:
        for member in agent.clusterHeadToken.members_:
            member_target_id = member.next_target_id

            if member_target_id is None:
                continue
            if member_target_id == env.num_targets:
                continue
            if member_target_id in agent.assigned_target_ids:
                continue  # Skip if the target is already assigned to this cluster

            logging.info("Target reassignment check for agent %d with target %d", member.id, member_target_id)
            target_position = env.target_positions[member_target_id]
            target_cluster_id = find_current_assigned_cluster(member_target_id, list_of_lvl1_cluster_heads)
            target_cluster = env.list_of_agents[target_cluster_id]

            distance_to_nearest_in_cluster_target = distance_to_closest_target(member, agent.assigned_target_ids,
                                                                               env.target_positions)
            distance_to_nearest_out_cluster_agent = distance_to_closest_agent(target_position, target_cluster,
                                                                              env.agent_positions)

            position_of_target = env.target_positions[member_target_id]
            print(f"current assigned target to member {member.id}: {member_target_id}")
            logging.info(f"current assigned target to member {member.id}: {member_target_id}")
            print(f"member position: {member.position}, target position: {position_of_target}")
            logging.info(f"member position: {member.position}, target position: {position_of_target}")

            distance_to_own_target = euclidean(member.position, position_of_target)
            print(f"distance_to_own_target: {distance_to_own_target}")
            logging.info(f"distance_to_own_target: {distance_to_own_target}")

            # keep the target if it is closer to the member than to any target in the cluster
            print(f"distance_to_nearest_in_cluster_target: {distance_to_nearest_in_cluster_target}")
            logging.info(f"distance_to_nearest_in_cluster_target: {distance_to_nearest_in_cluster_target}")
            if distance_to_own_target < distance_to_nearest_in_cluster_target and distance_to_own_target < distance_to_nearest_out_cluster_agent:
                print(
                    f"Hysteresis: Reassigning target {member_target_id} from agent {target_cluster_id} to agent {agent.id}")
                logging.info(
                    f"Hysteresis: Reassigning target {member_target_id} from agent {target_cluster_id} to agent {agent.id}")
                target_cluster = env.list_of_agents[target_cluster_id]
                new_set_of_assigned_targets = [id for id in target_cluster.assigned_target_ids if
                                               id != member_target_id]
                target_cluster.assigned_target_ids = new_set_of_assigned_targets
                agent.assigned_target_ids.append(member_target_id)
                # input("Press Enter to continue...")
            else:
                logging.info("Target %d is dropped for agent %d", member_target_id, member.id)
                member.next_target_id = env.num_targets  # null target

    # print(f"Agent {best_agent.id} assigned targets: {best_agent.assigned_target_ids}")
    for agent in list_of_lvl1_cluster_heads:
        target_position_dict = {}
        for target_id in agent.assigned_target_ids:
            target_position = env.target_positions[target_id]
            target_position = target_position.tolist()
            formatted_target_position = [np.round(target_position[0], 2), np.round(target_position[1], 2)]
            target_position_dict[target_id] = formatted_target_position

        formatted_agent_position = [np.round(agent.position[0], 2), np.round(agent.position[1], 2)]
        print(f"Agent {agent.id}:{formatted_agent_position} assigned targets: {target_position_dict}")
        logging.info(f"Agent {agent.id}:{formatted_agent_position} assigned targets: {target_position_dict}")
        # input("Press Enter to continue...")

    return list_of_lvl1_cluster_heads


def assign_targets_to_lvl1_clusters(env, logging):
    list_of_agents = env.list_of_agents
    # for agent in list_of_agents:
    #     print(f"Agent ID: {agent.id}")
    #     print(f"condition status: {agent.is_lvl1_cluster_head()}")
    list_of_lvl1_cluster_heads = [agent for agent in list_of_agents if agent.is_lvl1_cluster_head()]
    # for agent in list_of_lvl1_cluster_heads:
    #     print(f"Agent ID: {agent.id}, cluster members: {[member.id for member in agent.clusterHeadToken.members_]}")

    num_lvl1_clusters = len(list_of_lvl1_cluster_heads)

    num_targets = env.num_targets
    num_agents = env.num_agents
    # alpha = allParams["targetParams"]["alpha"]
    # beta = allParams["targetParams"]["beta"]
    variable_lambda = 1 / num_lvl1_clusters

    unvisited_targets = [target_id for target_id, visited in enumerate(env.target_mask) if not visited]

    for target_id in unvisited_targets:
        best_cost = float('inf')
        best_agent = None

        for agent in list_of_lvl1_cluster_heads:
            cluster_centroid = agent.clusterHeadToken.update_centroid()

            # Calculate the distance from the agent's cluster centroid to the target position
            target_position = env.target_positions[target_id]
            target_centroid_distance = euclidean(cluster_centroid, target_position)

            capacity_used = len(agent.assigned_target_ids)
            capacity_limit = num_targets * (len(agent.clusterHeadToken.members_) + 1) / num_agents

            penalty = max(0, capacity_used / capacity_limit - 1)

            # cost = target_centroid_distance + variable_lambda * penalty
            cost = target_centroid_distance

            if cost < best_cost:
                best_cost = cost
                best_agent = agent

        # print(f"Assigning target {target_id} to agent {best_agent.id} with cost {best_cost}")
        best_agent.assigned_target_ids.append(target_id)

    # if the target was previously assigned to another cluster, see if it should be reassigned
    # hysteresis to prevent oscillations
    for agent in list_of_lvl1_cluster_heads:
        for member in agent.clusterHeadToken.members_:
            member_target_id = member.next_target_id

            if member_target_id is None:
                continue
            if member_target_id == env.num_targets:
                continue
            if member_target_id in agent.assigned_target_ids:
                continue  # Skip if the target is already assigned to this cluster

            target_position = env.target_positions[member_target_id]
            target_cluster_id = find_current_assigned_cluster(member_target_id, list_of_lvl1_cluster_heads)
            target_cluster = env.list_of_agents[target_cluster_id]


            distance_to_nearest_in_cluster_target = distance_to_closest_target(member, agent.assigned_target_ids, env.target_positions)
            distance_to_nearest_out_cluster_agent = distance_to_closest_agent(target_position, target_cluster, env.agent_positions)

            position_of_target = env.target_positions[member_target_id]
            print(f"next_target_id of member {member.id}: {member_target_id}")
            logging.info(f"next_target_id of member {member.id}: {member_target_id}")
            print(f"member position: {member.position}, target position: {position_of_target}")
            logging.info(f"member position: {member.position}, target position: {position_of_target}")

            distance_to_own_target = euclidean(member.position, position_of_target)
            print(f"distance_to_own_target: {distance_to_own_target}")
            logging.info(f"distance_to_own_target: {distance_to_own_target}")

            # keep the target if it is closer to the member than to any target in the cluster
            print(f"distance_to_nearest_in_cluster_target: {distance_to_nearest_in_cluster_target}")
            logging.info(f"distance_to_nearest_in_cluster_target: {distance_to_nearest_in_cluster_target}")
            if distance_to_own_target < distance_to_nearest_in_cluster_target and distance_to_own_target < distance_to_nearest_out_cluster_agent:
                print(f"Hysteresis: Reassigning target {member_target_id} from agent {target_cluster_id} to agent {agent.id}")
                logging.info(f"Hysteresis: Reassigning target {member_target_id} from agent {target_cluster_id} to agent {agent.id}")
                target_cluster = env.list_of_agents[target_cluster_id]
                new_set_of_assigned_targets = [id for id in target_cluster.assigned_target_ids if id != member_target_id]
                target_cluster.assigned_target_ids = new_set_of_assigned_targets
                agent.assigned_target_ids.append(member_target_id)
                # input("Press Enter to continue...")
            else:
                member.next_target_id = env.num_targets  # null target

    # print(f"Agent {best_agent.id} assigned targets: {best_agent.assigned_target_ids}")
    for agent in list_of_lvl1_cluster_heads:
        print(f"Agent {agent.id} assigned targets: {agent.assigned_target_ids}")
        logging.info(f"Agent {agent.id} assigned targets: {agent.assigned_target_ids}")
        # input("Press Enter to continue...")

    return list_of_lvl1_cluster_heads


def generate_trajectories(instance_id = None, seed = None, iteration = 1, num_epochs = 1):
    # os.makedirs('saved_environments', exist_ok=True)
    log_file = 'clustering_solver.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=2),
            logging.StreamHandler()
        ]
    )

    os.makedirs('saved_buffer', exist_ok=True)

    env = gridworld_env(seed=seed)
    route_solver = ClusteringTSP(env)
    # steps_per_epoch = 5

    buffer = MultiAgentPathPlanningBuffer(env.num_agents, env.num_targets, env.num_obstacles)

    for e in range(num_epochs):
        ego_agent_position_buf = np.zeros((0, 2), dtype=np.float32)  # Buffer for ego agent positions
        other_agents_positions_buf = np.zeros((0, env.num_agents - 1, 2),
                                              dtype=np.float32)  # Buffer for other agents' positions
        target_positions_buf = np.zeros((0, env.num_targets, 2), dtype=np.float32)  # Buffer for target positions
        obstacle_positions_buf = np.zeros((0, env.num_obstacles, 8), dtype=np.float32)

        action_mask_buf = np.zeros((0, 5), dtype=np.bool_)
        target_mask_buf = np.zeros((0, env.num_targets + 1), dtype=np.bool_)
        action_buf = np.zeros(0, dtype=np.float32)
        route_buf = np.zeros(0, dtype=np.int64)

        # for CBS solver, need grid, agent cell positions and target cell positions
        done = False

        env.cluster_step()
        env.render()

        while not done:  # Continue until the experiment ends
            # need to assign targets to each level 1 cluster
            # lvl1_cluster_heads = assign_targets_to_lvl1_clusters(env, logging=logging)
            lvl1_cluster_heads = assign_targets_to_lvl1_clusters_OR_tools(env, route_solver, logging=logging)

            # need to get manager, routing, solution for every level 1 cluster
            all_routes = {}
            # for each level 1 cluster head, solve the open-ended VRP
            print(f"Solving open-ended VRP")
            logging.info("Solving open-ended VRP")
            for agent in lvl1_cluster_heads:
                distance_matrix, num_agents, manager, routing, solution = route_solver.solve_open_ended_vrp(agent)
                routes = route_solver.get_routes_as_dict(manager, routing, solution, num_agents)

                # these need to be assigned targets

                assigned_targets = [target_id for target_id in agent.assigned_target_ids]

                # need to relabel the route agent_ids
                cluster_member_ids = [member.id for member in agent.clusterHeadToken.members_]
                cluster_member_ids.append(agent.id)  # Add the agent itself in the cluster

                # the routes consist of actual target IDs assigned to the actual agent IDs in the  cluster
                relabelled_routes = {
                    cluster_member_ids[agent_id]: [assigned_targets[index] for index in route]
                    for agent_id, route in routes.items()
                }
                print(f"Agent {agent.id} relabelled routes: {relabelled_routes}")
                logging.info(f"Agent {agent.id} relabelled routes: {relabelled_routes}")
                print(f"Assigned targets for agent {agent.id}: {assigned_targets}")
                logging.info(f"Assigned targets for agent {agent.id}: {assigned_targets}")

                for agent_id, route in relabelled_routes.items():
                    if route:
                        all_routes[agent_id] = route

            # input("Routes displayed.  Press Enter to continue...")

            # Clear assigned targets for the next iteration
            for agent in lvl1_cluster_heads:
                agent.assigned_target_ids = []

            # Initialize first_targets with the null target for all agents
            first_targets = [env.num_targets] * env.num_agents  # Null target is represented by env.num_targets
            agents = [agent for agent, route in all_routes.items() if route]

            # Update first_targets for agents with non-empty routes
            # need to change things if the agent is currently on the target
            for agent_id, route in zip(agents, [route for route in all_routes.values() if route]):
                agent_position_cell = env.convert_position_to_grid_cell(env.agent_positions[agent_id])
                first_target_cell = env.convert_position_to_grid_cell(env.target_positions[route[0]])
                if agent_position_cell == first_target_cell:
                    print(f"Agent {agent_id} is already at the first target {route[0]}. Marking as visited.")
                    logging.info(f"Agent {agent_id} is already at the first target {route[0]}. Marking as visited.")
                    env.target_mask[route[0]] = True  # Mark the target as visited
                    if len(route) > 1:
                        first_targets[agent_id] = route[1]
                    else:
                        first_targets[agent_id] = env.num_targets
                    print(f"Agent {agent_id} first target updated to {first_targets[agent_id]}")
                    logging.info(f"Agent {agent_id} first target updated to {first_targets[agent_id]}")
                else:
                    first_targets[agent_id] = route[0]

            filtered_agent_cell_positions = [
                env.convert_position_to_grid_cell(position) for position in env.agent_positions
            ]

            filtered_target_cell_positions = [
                env.convert_position_to_grid_cell(env.target_positions[target]) if target != env.num_targets else None
                for target in first_targets
            ]

            # Step 3: Generate collision-free paths using CBSsolver
            print(f"Finding collision-free paths")
            logging.info("Finding collision-free paths")
            cbs_solver = CBS(env.grid_state, filtered_agent_cell_positions, filtered_target_cell_positions)
            paths, action_sequence = cbs_solver.plan_paths()

            if not paths:
                print("No collision-free paths found. Resetting environment.")
                logging.info("No collision-free paths found. Resetting environment.")
                env.reset()
                # env_file_path = f'saved_environments/env_instance_{instance_id}_iteration_{iteration}_epoch_{e}.pkl'
                # with open(env_file_path, 'wb') as f:
                #     pickle.dump(env, f)
                # print(f"Env for instance:iteration:epoch {instance_id}:{iteration}:{e} saved successfully.")
                continue
            else:
                # env.render()

                # indices on the path vector where the target is reached for each agent
                indices = find_first_target_reached_index(paths, filtered_target_cell_positions)

                # print(f"Indices of first target reached for each agent: {indices}")
                # logging.info(f"Indices of first target reached for each agent: {indices}")
                valid_indices = [index for index in indices if index != -1]

                # print(f"Valid indices of first target reached for each agent: {valid_indices}")
                # logging.info(f"Valid indices of first target reached for each agent: {valid_indices}")
                smallest_index = min(valid_indices) if valid_indices else None

                # print(f"Smallest index of first target reached: {smallest_index}")
                # logging.info(f"Smallest index of first target reached: {smallest_index}")
                # action sequences for each agent up to the first target reached
                new_action_sequence = create_prefix_action_sequences(action_sequence, smallest_index)

                # print(f"Old Action Sequence")
                # logging.info("Old Action Sequence")
                # for index, actions in enumerate(action_sequence):
                    # print(f"Agent {index} original action sequence: {actions}")
                    # logging.info(f"Agent {index} original action sequence: {actions}")

                # print(f"New Action Sequence")
                # logging.info("New Action Sequence")
                # for index, actions in enumerate(new_action_sequence):
                #     print(f"Agent {index} action sequence: {actions}")
                #     logging.info(f"Agent {index} action sequence: {actions}")

                # input("Action sequences displayed. Press Enter to continue...")

                # print(f"Iterating through the path segments")
                # logging.info("Iterating through the path segments")
                for step in range(smallest_index):  # Iterate over the shortest path
                    next_actions = {}

                    for index, actions in enumerate(
                            new_action_sequence):  # Go through every agent and their action sequence

                        # all of this is for populating the buffer used in supervised learning
                        agent = env.list_of_agents[index]  # Get the agent object from the environment
                        ego_agent_position = agent.position
                        ego_agent_position = ego_agent_position.reshape(1, 2)  # Get the position of the agent
                        ego_agent_position_buf = np.concatenate((ego_agent_position_buf, ego_agent_position), axis=0)

                        agents_positions, target_positions, obstacle_positions = env.get_path_model_obs_input()

                        other_agents_positions = np.delete(agents_positions, index, axis=1)
                        other_agents_positions_buf = np.concatenate(
                            (other_agents_positions_buf, other_agents_positions), axis=0)

                        target_positions_buf = np.concatenate((target_positions_buf, target_positions), axis=0)
                        obstacle_positions_buf = np.concatenate((obstacle_positions_buf, obstacle_positions), axis=0)

                        action = actions[step]  # get the action of a particular sequence of an agent
                        next_actions[index] = action  # set the action as the value to the agent key

                        action = np.array([action])
                        action_buf = np.concatenate((action_buf, action), axis=0)

                        action_mask = np.array(agent.action_mask)
                        action_mask = action_mask.reshape(1, 5)
                        action_mask_buf = np.concatenate((action_mask_buf, action_mask), axis=0)

                        target_mask = np.array(env.target_mask)
                        target_mask = np.append(target_mask, 0)  # append the null target to the target mask
                        # target_mask = target_mask
                        # print(f"target_mask: {target_mask}")
                        target_mask = target_mask.reshape(1, env.num_targets + 1)
                        target_mask_buf = np.concatenate((target_mask_buf, target_mask), axis=0)

                        target = first_targets[index]  # get the target of the agent
                        # used to determine if the agent has reached the target it was assigned (not accidentally reached another)
                        # I want to use this for assigning targets to clusters
                        agent.next_target_id = target
                        target = np.array([target])

                        route_buf = np.concatenate((route_buf, target), axis=0)

                    print(f"Executing actions for agents: {next_actions}")
                    logging.info(f"Executing actions for agents: {next_actions}")
                    # input("Press Enter to execute actions...")

                    _, _, _, temp_done, _ = env.action_step(next_actions)
                    env.cluster_step()
                    done = temp_done
                    env.render()

                print(f"finished iterating through the path segments")
                logging.info("finished iterating through the path segments")
                # input("Press Enter to continue...")

        buffer.store(ego_agent_position_buf, other_agents_positions_buf, target_positions_buf, obstacle_positions_buf,
                     action_mask_buf, target_mask_buf, action_buf, route_buf)
        # buffer.store(obs_buf, action_mask_buf, target_mask_buf, action_buf, route_buf)

        env_file_path = f'saved_buffer/buffer_instance_{instance_id}_iteration_{iteration}_epoch_{e}.pkl'
        with open(env_file_path, 'wb') as f:
            pickle.dump(buffer, f)
        print(f"Buffer for instance:iteration:epoch {instance_id}:{iteration}:{e} saved successfully.")
        logging.info(f"Buffer for instance:iteration:epoch {instance_id}:{iteration}:{e} saved successfully.")

        env.reset()

    return buffer


if __name__ == "__main__":
    seed = 100
    buffer = generate_trajectories(seed=seed)
    print(f"Final Ego Agent Position Buffer Shape: {buffer.ego_agent_buf.shape}")
    print(f"Final Other Agents Position Buffer Shape: {buffer.other_agents_buf.shape}")
    print(f"Final Target Position Buffer Shape: {buffer.targets_buf.shape}")
    print(f"Final Obstacle Position Buffer Shape: {buffer.obstacles_buf.shape}")

    print(f"Final Action Mask Shape: {buffer.action_mask_buf.shape}")
    print(f"Final Target Mask Shape: {buffer.target_mask_buf.shape}")
    print(f"Final Action Buffer Shape: {buffer.action_buf.shape}")
    print(f"Final Route Buffer Shape: {buffer.route_buf.shape}")

