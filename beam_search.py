from copy import deepcopy

from grid_world_env import gridworld_env
from buffers import RoutePlanningBuffer
import numpy as np
import torch
import heapq
import copy
from algos_scratch_pad.utils.logx import EpochLogger
from algos_scratch_pad.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from algos_scratch_pad.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from torch.optim import Adam
# from route_planning_model import RoutePlanningModel
from path_planning_model import PathPlanningModel

def beam_search_targets(env, path_model, beam_width=100):
    """
    Beam search to find the optimal sequence of unvisited targets.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # things to include in the beam: log prob, target sequence, temp_env, route_length
    beam = [(0, [], copy.deepcopy(env), 0)]  # (negative log prob, target sequence, env)
    final_sequences = []
    # # print(f"Target Positions: {env.target_positions}")

    while beam:

        new_beam = []

        for neg_log_prob, target_sequence, temp_env, route_length in beam:

            temp_env.update_grid_mask()
            temp_env.update_action_mask()
            temp_env.update_target_mask()

            # # print(f"Current Beam: {neg_log_prob}, {target_sequence}, {temp_env.agent_position}, {route_length}")
            # input("Press Enter to continue...")  # Pause to inspect the current beam
            # print(f"Current Beam: {beam}")
            # print(f"sum: {sum(temp_env.target_mask)}, target_mask: {temp_env.target_mask}, num_targets: {temp_env.num_targets}")

            if sum(temp_env.target_mask)==temp_env.num_targets:
                final_sequences.append((neg_log_prob, target_sequence, temp_env, route_length))
                # print(f"hello")
                continue

            target_mask = temp_env.target_mask
            action_mask = temp_env.action_mask
            obs_input = temp_env.get_path_model_obs_input()

            # print(f"obs_input (beam_search_targets): {obs_input.shape}, target_mask: {target_mask}, action_mask: {action_mask}")

            obs_input = torch.as_tensor(obs_input, dtype=torch.float32).to(device)
            target_mask = torch.as_tensor(target_mask, dtype=torch.bool).to(device)
            action_mask = torch.as_tensor(action_mask, dtype=torch.bool).to(device)

            target_dist, _ = path_model(obs_input, target_mask, action_mask)
            target_probs = target_dist.probs.detach().cpu().numpy().flatten()

            top_k = heapq.nlargest(beam_width, range(len(target_probs)), key=target_probs.__getitem__)

            for idx in top_k:
                prob = target_probs[idx]
                # # print(f"Target Index: {idx}, Probability: {prob}, Target_Mask: {target_mask[idx]})")
                # input("Press Enter to continue...")  # Pause to inspect target probabilities
                if prob > 0 and not target_mask[idx]:
                    # # print(f"selected target: {idx}, prob: {prob}")
                    new_temp_env = copy.deepcopy(temp_env)
                    new_temp_env.target_mask[idx] = 1

                    agent_position = new_temp_env.agent_position
                    agent_position_cell = new_temp_env.convert_position_to_grid_cell(agent_position)

                    target_position = new_temp_env.target_positions[idx]
                    target_position_cell = new_temp_env.convert_position_to_grid_cell(target_position)

                    new_segment_length = abs(agent_position_cell[0] - target_position_cell[0]) + abs(agent_position_cell[1] - target_position_cell[1])
                    new_route_length = route_length + new_segment_length

                    new_temp_env.agent_position = target_position

                    new_beam.append((neg_log_prob - np.log(prob), target_sequence + [idx], new_temp_env, new_route_length))

                    # # print(f"New Beam Entry: {neg_log_prob - np.log(prob)}, {target_sequence + [idx]}, {new_temp_env.agent_position}, {new_route_length}")

        beam = heapq.nsmallest(beam_width, new_beam, key=lambda x: x[0])

        # # for neg_log_prob, target_sequence, temp_env, route_length in beam:
            # # print(f"Beam (targets): {neg_log_prob}, {target_sequence}, {temp_env.agent_position}, {route_length}")

        # final_sequences.extend(beam)


    # Return the best sequence of targets
    # # for neg_log_prob, target_sequence, temp_env, route_length in final_sequences:
        # # print(f"Final Sequence (targets): {neg_log_prob}, {target_sequence}, {temp_env.agent_position}, {route_length}")

    best_sequence = min(final_sequences, key=lambda x: x[3])

    # # print(f"Best Sequence (targets): {best_sequence[0]}, {best_sequence[1]}, {best_sequence[2].agent_position}, {best_sequence[3]}")
    # input("Press Enter to continue...")  # Pause to inspect the best sequence

    return best_sequence[1]  # Return the target sequence


def beam_search_trajectories(env, path_model, target, beam_width=100):
    """
    Beam search to find the optimal trajectory to reach a specific target.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    final_trajectories = []
    target_position = env.target_positions[target]
    target_position_cell = env.convert_position_to_grid_cell(target_position)

    agent_position = env.agent_position
    agent_position_cell = env.convert_position_to_grid_cell(agent_position)
    distance_to_go = abs(agent_position_cell[0] - target_position_cell[0]) + abs(agent_position_cell[1] - target_position_cell[1])

    beam = [(0, 0, False, [], copy.deepcopy(env), [], distance_to_go)]  # (neg log prob, distance, target_reached, path, env, actions)

    target_reached = False

    while beam:

        new_beam = []
        # print(f"target_reached_flag: {target_reached_flag}")
        for neg_log_prob, distance, done, path, temp_env, actions, distance_to_go in beam:

            temp_env.update_grid_mask()
            temp_env.update_action_mask()
            temp_env.update_target_mask()

            # # print(f"Current Beam: {neg_log_prob}, {distance}, {done}, {path}, {temp_env.agent_position}, {actions}, {distance_to_go}")
            # input("Press Enter to continue...")  # Pause to inspect the current beam

            if target_reached or done or len(actions) > temp_env.grid_dim:
                # neg_log_prob, distance, path, temp_env, actions, distance_to_go
                final_trajectories.append((neg_log_prob, distance, path, actions, distance_to_go))
                target_reached = True
                # input("Press Enter to continue...")  # Pause to inspect the final trajectory
                # target_reached = True
                continue

            target_mask = temp_env.target_mask
            action_mask = temp_env.action_mask
            obs_input = temp_env.get_path_model_obs_input()

            obs_input = torch.as_tensor(obs_input, dtype=torch.float32).to(device)
            target_mask = torch.as_tensor(target_mask, dtype=torch.bool).to(device)
            action_mask = torch.as_tensor(action_mask, dtype=torch.bool).to(device)

            _, action_dist = path_model(obs_input, target_mask, action_mask)
            action_probs = action_dist.probs.detach().cpu().numpy().flatten()

            top_k = heapq.nlargest(beam_width, range(len(action_probs)), key=action_probs.__getitem__)

            for idx in top_k:
                prob = action_probs[idx]

                target_reached_flag = False

                # # print(f"Action Index: {idx}, Probability: {prob})")
                # input("Press Enter to continue...")  # Pause to inspect target probabilities

                # need to check with action_mask because we are going through (possibly) all actions
                if prob > 0 and not action_mask[idx]:
                    # # print(f"selected action: {idx}, prob: {prob}")
                    new_temp_env = copy.deepcopy(temp_env)

                    _, _, rew, _, _ = new_temp_env.action_step(idx)

                    # agent_position_cell = new_temp_env.convert_position_to_grid_cell(new_temp_env.agent_position)

                    agent_position = new_temp_env.agent_position
                    agent_position_cell = new_temp_env.convert_position_to_grid_cell(agent_position)

                    distance_to_go = abs(agent_position_cell[0] - target_position_cell[0]) + abs(agent_position_cell[1] - target_position_cell[1])

                    if agent_position_cell == target_position_cell:
                        # # print(f"Target Reached: {agent_position_cell}")
                        target_reached_flag = True
                        # target_reached = True
                        # input("Press Enter to continue...")  # Pause to inspect target reached
                        # done = True

                    offset = 0
                    if new_temp_env.reset_flag:
                        offset = new_temp_env.grid_dim
                        new_temp_env.reset_flag = False

                    new_beam.append((neg_log_prob - np.log(prob), distance + rew + offset, target_reached_flag, path + [new_temp_env.agent_position], new_temp_env, actions + [idx], distance_to_go))

                    # # print(f"New Beam Entry: {neg_log_prob - np.log(prob)}, {distance + rew + offset}, {target_reached_flag}, {path + [new_temp_env.agent_position]}, {new_temp_env.agent_position}, {actions + [idx]}, {distance_to_go}")

                    # target_reached_flag = False

        beam = heapq.nsmallest(beam_width, new_beam, key=lambda x: x[0])

        # # for neg_log_prob, distance, done, path, temp_env, actions, distance_to_go in beam:
            # # print(f"Beam (trajectories): {neg_log_prob}, {distance}, {done}, {path}, {temp_env.agent_position}, {actions}, {distance_to_go}")

    # Return the best trajectory
    best_trajectory = min(final_trajectories, key=lambda x: len(x[3]) + x[4])

    # print(f"Best Trajectory (actions): {best_trajectory[0]}, {best_trajectory[1]}, {best_trajectory[2]}, {best_sequence[3]}")
    # neg_log_prob, distance, path, temp_env, actions, distance_to_go
    # # print(f"best trajectory: {best_trajectory}")
    # for neg_log_prob, distance, path, actions, distance_to_go in best_trajectory:
    #     print(f"Best Trajectory: {neg_log_prob}, {distance}, {path}, {actions} {distance_to_go}")

    # input("Press Enter to continue...")  # Pause to inspect the best sequence

    return best_trajectory[2], best_trajectory[3]  # Return path and actions


def combined_beam_search(env, path_model, beam_width=100):
    """
    Perform a combined beam search over targets and trajectories.
    """
    # Step 1: Beam search over targets
    target_sequence = beam_search_targets(env, path_model, beam_width)

    # Step 2: Use the first target in the sequence
    first_target = target_sequence[0]

    # Step 3: Beam search over trajectories to reach the first target
    path, actions = beam_search_trajectories(env, path_model, first_target, beam_width)

    return target_sequence, path, actions