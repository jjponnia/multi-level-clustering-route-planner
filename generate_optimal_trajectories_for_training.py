import time

from multi_agent_route_planner import MultiAgentTSP
from CBSsolver import CBS
from grid_world_env import gridworld_env
import numpy as np
from buffers import MultiAgentPathPlanningBuffer
import multiprocessing
import pickle
import os

# Assuming `env`, `route_solver`, `CBS`, and `path_planning_buf` are already initialized

# what are the observations?
# the positions of the agents, the positions of the targets, the positions and length of the obstacles

# initalize route planner


def find_first_target_reached_index(paths, targets):
    """
    Find the first index where an agent's position coincides with its target position.

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
    new_action_sequences = [
        actions[:smallest_index] if smallest_index <= len(actions) else actions
        for actions in action_sequences
    ]
    return new_action_sequences


def generate_trajectories(instance_id = None, seed = None, iteration = 1, num_epochs = 1):
    # os.makedirs('saved_environments', exist_ok=True)
    os.makedirs('saved_buffer', exist_ok=True)

    env = gridworld_env(seed=seed)
    route_solver = MultiAgentTSP(env)
    # steps_per_epoch = 5

    buffer = MultiAgentPathPlanningBuffer(env.num_agents, env.num_targets, env.num_obstacles)

    for e in range(num_epochs):
        # print(f"epoch: {e}")

        # env_file_path = f'saved_environments/env_instance_{instance_id}_iteration_{iteration}_epoch_{e}.pkl'

        # with open(env_file_path, 'wb') as f:
        #     pickle.dump(env, f)
        # print(f"Env for instance:iteration:epoch {instance_id}:{iteration}:{e} saved successfully.")

        # obs_buf = np.zeros((0, buffer.obs_dim), dtype=np.float32)
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

        while not done:  # Continue until the experiment ends

            distance_matrix, num_agents, manager, routing, solution = route_solver.solve_open_ended_vrp()
            routes = route_solver.get_routes_as_dict(manager, routing, solution, num_agents)

            unvisited_targets = [target_id for target_id, visited in enumerate(env.target_mask) if not visited]

            relabelled_routes = {
                agent_id: [unvisited_targets[index] for index in route]
                for agent_id, route in routes.items()
            }

            # Initialize first_targets with the null target for all agents
            first_targets = [env.num_targets] * env.num_agents  # Null target is represented by env.num_targets
            agents = [agent for agent, route in relabelled_routes.items() if route]

            # Update first_targets for agents with non-empty routes
            for agent, target in zip(agents, [route[0] for route in relabelled_routes.values() if route]):
                first_targets[agent] = target

            filtered_agent_cell_positions = [
                env.convert_position_to_grid_cell(position) for position in env.agent_positions
            ]
            filtered_target_cell_positions = [
                env.convert_position_to_grid_cell(env.target_positions[target]) if target != env.num_targets else None
                for target in first_targets
            ]

            # Step 3: Generate collision-free paths using CBSsolver
            cbs_solver = CBS(env.grid_state, filtered_agent_cell_positions, filtered_target_cell_positions)
            paths, action_sequence = cbs_solver.plan_paths()

            if not paths:
                print("No collision-free paths found. Resetting environment.")
                env.reset()
                # env_file_path = f'saved_environments/env_instance_{instance_id}_iteration_{iteration}_epoch_{e}.pkl'
                # with open(env_file_path, 'wb') as f:
                #     pickle.dump(env, f)
                # print(f"Env for instance:iteration:epoch {instance_id}:{iteration}:{e} saved successfully.")
                continue
            else:
                # env.render()

                indices = find_first_target_reached_index(paths, filtered_target_cell_positions)

                valid_indices = [index for index in indices if index != -1]
                smallest_index = min(valid_indices) if valid_indices else None

                new_action_sequence = create_prefix_action_sequences(action_sequence, smallest_index)

                for step in range(smallest_index):  # Iterate over the shortest path
                    next_actions = {}
                    num_agents = env.num_agents
                    for index, actions in enumerate(
                            new_action_sequence):  # Go through every agent and their action sequence

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

                        # agent_one_hot = env.convert_to_one_hot_vector(
                        #     index)  # convert the agent index to a one-hot vector
                        # agent_one_hot = agent_one_hot.reshape(1, num_agents)

                        # new_obs = env.get_path_model_obs_input()
                        # new_obs = np.concatenate((agent_one_hot, new_obs), axis=1)
                        # obs_buf = np.concatenate((obs_buf, new_obs), axis=0)

                        # agent = env.list_of_agents[index]
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

                        target = np.array([target])
                        route_buf = np.concatenate((route_buf, target), axis=0)

                    _, _, _, temp_done, _ = env.action_step(next_actions)
                    done = temp_done
                    # env.render()

        buffer.store(ego_agent_position_buf, other_agents_positions_buf, target_positions_buf, obstacle_positions_buf,
                     action_mask_buf, target_mask_buf, action_buf, route_buf)
        # buffer.store(obs_buf, action_mask_buf, target_mask_buf, action_buf, route_buf)

        env_file_path = f'saved_buffer/buffer_instance_{instance_id}_iteration_{iteration}_epoch_{e}.pkl'
        with open(env_file_path, 'wb') as f:
            pickle.dump(buffer, f)
        print(f"Buffer for instance:iteration:epoch {instance_id}:{iteration}:{e} saved successfully.")

        env.reset()

    return buffer


def worker(instance_id, seed, iteration, num_epochs, status_dict):
    """
    Worker function to perform a task.
    Args:
        instance_id (int): The ID of the instance.
        seed (int): The unique seed for the process.
    """
    try:
        timestamp = time.strftime("%H:%M:%S")
        status_dict[instance_id] = f"Started | Iteration: {iteration} | Timestamp: {timestamp}"

        _ = generate_trajectories(instance_id=instance_id, seed=seed, iteration=iteration, num_epochs=num_epochs)

        timestamp = time.strftime("%H:%M:%S")
        status_dict[instance_id] = f"Completed | Iteration: {iteration} | Timestamp: {timestamp}"

    except Exception as e:
        timestamp = time.strftime("%H:%M:%S")
        status_dict[instance_id] = f"Error: {e} | Iteration: {iteration} | Timestamp: {timestamp}"


def spawn_and_monitor_processes(num_processes, num_iterations, num_epochs, timeout):
    """
    Spawns and monitors processes in a loop.
    Args:
        num_processes (int): Number of processes to spawn per iteration.
        num_iterations (int): Number of iterations to run.
        timeout (int): Timeout in seconds to monitor each process.
    """

    for iteration in range(num_iterations):
        print(f"Starting iteration {iteration}/{num_iterations - 1}")

        with multiprocessing.Manager() as manager:
            status_dict = manager.dict()

            processes = []
            seeds = [(iteration * num_processes + i + 1) * 100 for i in range(num_processes)]

            # Spawn processes
            for instance_id, seed in enumerate(seeds):
                process = multiprocessing.Process(target=worker, args=(instance_id, seed, iteration, num_epochs, status_dict))
                processes.append((process, instance_id, seed))
                process.start()

            start_time = time.time()
            while time.time() - start_time < timeout:
                time.sleep(60) # Check every minute
                update = f"Process statuses with {timeout - (time.time() - start_time)} seconds remaining: {dict(status_dict)}"
                print(update)

                if all("Completed" in status for status in status_dict.values()):
                    print("All processes completed successfully.")
                    break

            # Monitor processes
            for process, instance_id, seed in processes:
                # process.join(timeout)
                if process.is_alive():
                    print(f"Worker {instance_id} with seed {seed} is non-responsive. Terminating.")
                    process.terminate()
                    process.join()

            print("Final statuses:", dict(status_dict))

        print(f"Iteration {iteration} completed.\n")


def concatenate_buffers(num_iterations, num_epochs, num_instances, num_agents, num_targets, num_obstacles):
    main_buffer = MultiAgentPathPlanningBuffer(num_agents, num_targets, num_obstacles)  # Initialize an empty buffer

    # Initialize empty buffers for concatenation
    # obs_buf = np.zeros((0, main_buffer.obs_dim), dtype=np.float32)

    ego_agent_buf = np.zeros((0, 2), dtype=np.float32)  # Buffer for ego agent positions
    other_agents_buf = np.zeros((0, num_agents - 1, 2), dtype=np.float32)  # Buffer for other agents' positions
    targets_buf = np.zeros((0, num_targets, 2), dtype=np.float32)  # Buffer for target positions
    obstacles_buf = np.zeros((0, num_obstacles, 8), dtype=np.float32)  # Buffer for obstacle positions

    action_mask_buf = np.zeros((0, 5), dtype=np.bool_)
    target_mask_buf = np.zeros((0, num_targets + 1), dtype=np.bool_)
    action_buf = np.zeros(0, dtype=np.float32)
    route_buf = np.zeros(0, dtype=np.int64)

    for iteration in range(num_iterations):
        for instance_id in range(num_instances):
            for epoch in range(num_epochs):
                print(f"Loading buffer for instance {instance_id}, iteration {iteration}, epoch {epoch}")
                # Load the buffer from the file
                buffer_path = f'saved_buffer/buffer_instance_{instance_id}_iteration_{iteration}_epoch_{epoch}.pkl'

                if os.path.exists(buffer_path):
                    with open(buffer_path, 'rb') as f:
                        buffer = pickle.load(f)

                    # Extract and concatenate the attributes
                    ego_agent_buf = np.concatenate((ego_agent_buf, buffer.ego_agent_buf), axis=0)
                    other_agents_buf = np.concatenate((other_agents_buf, buffer.other_agents_buf), axis=0)
                    targets_buf = np.concatenate((targets_buf, buffer.targets_buf), axis=0)
                    obstacles_buf = np.concatenate((obstacles_buf, buffer.obstacles_buf), axis=0)

                    action_mask_buf = np.concatenate((action_mask_buf, buffer.action_mask_buf), axis=0)
                    target_mask_buf = np.concatenate((target_mask_buf, buffer.target_mask_buf), axis=0)
                    action_buf = np.concatenate((action_buf, buffer.action_buf), axis=0)
                    route_buf = np.concatenate((route_buf, buffer.route_buf), axis=0)
                else:
                    print(f"Buffer file {buffer_path} does not exist. Skipping.")

    main_buffer.ego_agent_buf = ego_agent_buf
    main_buffer.other_agents_buf = other_agents_buf
    main_buffer.targets_buf = targets_buf
    main_buffer.obstacles_buf = obstacles_buf

    main_buffer.action_mask_buf = action_mask_buf
    main_buffer.target_mask_buf = target_mask_buf
    main_buffer.action_buf = action_buf
    main_buffer.route_buf = route_buf

    return main_buffer


if __name__ == "__main__":
    # generate_trajectories()
    # num_processes = 30
    # num_iterations = 45
    # num_epochs = 15

    num_processes = 30
    num_iterations = 20
    num_epochs = 15

    timeout = 1200  # seconds

    spawn_and_monitor_processes(num_processes, num_iterations, num_epochs, timeout)

    env = gridworld_env()
    num_agents = env.num_agents
    num_targets = env.num_targets
    num_obstacles = env.num_obstacles

    buffer = concatenate_buffers(num_iterations, num_epochs, num_processes, num_agents, num_targets, num_obstacles)

    # print(f"Final Obs Buffer Shape: {buffer.obs_buf.shape}")

    print(f"Final Ego Agent Position Buffer Shape: {buffer.ego_agent_buf.shape}")
    print(f"Final Other Agents Position Buffer Shape: {buffer.other_agents_buf.shape}")
    print(f"Final Target Position Buffer Shape: {buffer.targets_buf.shape}")
    print(f"Final Obstacle Position Buffer Shape: {buffer.obstacles_buf.shape}")

    print(f"Final Action Mask Shape: {buffer.action_mask_buf.shape}")
    print(f"Final Target Mask Shape: {buffer.target_mask_buf.shape}")
    print(f"Final Action Buffer Shape: {buffer.action_buf.shape}")
    print(f"Final Route Buffer Shape: {buffer.route_buf.shape}")

    with open('final_buffer.pkl', 'wb') as f:
        pickle.dump(buffer, f)

    print("Final buffer saved successfully to 'final_buffer.pkl'.")


# buffer = generate_trajectories()