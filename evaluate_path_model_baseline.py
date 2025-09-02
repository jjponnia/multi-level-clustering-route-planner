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

def get_estimate_of_remaining_tour_distance(env, path_model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # get obs_input, target_mask, action_mask

    temp_env = copy.deepcopy(env)

    obs_input = temp_env.get_path_model_obs_input()
    obs_input = torch.as_tensor(obs_input, dtype=torch.float32).to(device)

    temp_env.update_target_mask()
    target_mask = temp_env.target_mask

    temp_env.update_action_mask()
    action_mask = temp_env.action_mask

    route = []

    first_target = True
    initial_distance = 0
    distance = 0

    while sum(target_mask) < env.num_targets:
        # print(f"target_mask (beam_search): {target_mask}")
        # print(f"action_mask (beam_search): {action_mask}")
        # print(f"obs_input (beam_search): {obs_input.shape}")

        obs_input = torch.as_tensor(obs_input, dtype=torch.float32).to(device)
        target_mask = torch.as_tensor(target_mask, dtype=torch.bool).to(device)
        action_mask = torch.as_tensor(action_mask, dtype=torch.bool).to(device)

        # print("get estimate of remaining tour distance")
        target_dist, action_dist = path_model(obs_input, target_mask, action_mask)

        target_probs = target_dist.probs
        best_target = torch.argmax(target_probs).item()
        route.append(best_target)

        # if start_route_segment is None:
        #     start_route_segment = best_target
        next_target_position = copy.deepcopy(temp_env.target_positions[best_target])
        next_target_position_cell = temp_env.convert_position_to_grid_cell(next_target_position)
        print(f"next_target_position (remaining tour distance): {next_target_position_cell}")
        # input("Press Enter to continue...")

        agent_position_cell = temp_env.convert_position_to_grid_cell(temp_env.agent_position)
        print(f"agent_position (remaining tour distance): {agent_position_cell}")
        # input("Press Enter to continue...")

        if first_target:
            print("FIRST TARGET")
            first_target = False
            initial_distance = abs(agent_position_cell[0] - next_target_position_cell[0]) + abs(agent_position_cell[1] - next_target_position_cell[1])
            print(f"initial_distance (remaining tour distance): {initial_distance}")
            distance += initial_distance
            # print(f"distance (remaining tour distance): {distance}")
        else:
            next_segment = abs(agent_position_cell[0] - next_target_position_cell[0]) + abs(agent_position_cell[1] - next_target_position_cell[1])
            print(f"next_segment (remaining tour distance): {next_segment}")
            distance += next_segment
            print(f"distance (remaining tour distance): {distance}")

        obs_input = temp_env.get_path_model_obs_input()

        # temp_agent_position = temp_agent_position / env.grid_dim

        # temp_env = copy.deepcopy(env)
        # next_target_position = np.array(next_target_position)
        # temp_env.agent_position = next_target_position / temp_env.grid_dim
        temp_env.agent_position = next_target_position

        print(f"temp_env.agent_position (remaining tour distance): {temp_env.agent_position}")
        # print(f"agent_grid_cell: {temp_env.convert_position_to_grid_cell(temp_env.agent_position)}")
        # input("Press Enter to continue...")
        # temp_env.target_mask = target_mask
        temp_env.update_grid_mask()
        temp_env.update_action_mask()
        temp_env.update_target_mask()

        action_mask = temp_env.action_mask
        target_mask = temp_env.target_mask

        print(f"target_mask (remaining tour distance): {target_mask}")
        print(f"best_target (remaining tour distance): {best_target}")
        # input("Press Enter to continue...")

        obs_input[0][0] = temp_env.agent_position[0]
        obs_input[0][1] = temp_env.agent_position[1]

        # target_mask[best_target] = True

    # print(f"route (remaining tour distance): {route}")

    return initial_distance, distance


def beam_search(env, path_model, beam_width=100):

    # agent_positions, target_positions = env.get_obs_inputs()
    # obs_input, mask = env.get_path_model_obs_input()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the beam with the initial state
    done = False
    # new_beam.append((neg_log_prob - np.log(prob), new_distance_to_target, done, position_path + [agent_position], new_temp_env, action_sequence + [idx], reset_flag, new_estimated_remaining_tour_distance
    # negative log prob, distance_to_target, done, position_path, env, action_sequence, reset_flag, estimated_remaining_distance
    beam = [(0, 0, done, [], copy.deepcopy(env), [], False, 0)]  # (negative log prob, distance to target, done, position path, env)
    # beam = [(0, 0, done, [])]  # (negative log prob, rew, path, mask)

    final_trajectories = []

    # env.render()

    target_reached = False

    while beam and not target_reached:
        new_beam = []

        # neg_log_prob, distance_to_target, done, position_path, temp_env, action_sequence, reset_flag, estimated_remaining_distance
        for neg_log_prob, distance_to_target, done, position_path, temp_env, action_sequence, reset_flag, estimated_remaining_tour_distance in beam:
            print(
                f"current trajectory: {action_sequence}, distance_to_target: {distance_to_target}, done: {done}, reset_flag: {reset_flag}, estimated_remaining_tour_distance: {estimated_remaining_tour_distance}")
            # for col in reversed(temp_env.grid_mask[:, :, 1].T):
            #     print(col)

            if done or len(action_sequence) > 3*env.grid_dim:
                final_trajectories.append((neg_log_prob, distance_to_target, done, position_path, [], action_sequence, reset_flag, estimated_remaining_tour_distance))
                target_reached = True
                continue

            # if len(position_path) > 0:
            #     agent_position = position_path[-1]
            #     env.agent_position = deepcopy(agent_position)

            # obs_input, mask = temp_env.get_path_model_obs_input()
            obs_input = temp_env.get_path_model_obs_input()

            temp_env.update_target_mask()
            target_mask = temp_env.target_mask

            temp_env.update_action_mask()
            action_mask = temp_env.action_mask

            obs_input = torch.as_tensor(obs_input, dtype=torch.float32).to(device)
            target_mask = torch.as_tensor(target_mask, dtype=torch.bool).to(device)
            action_mask = torch.as_tensor(action_mask, dtype=torch.bool).to(device)

            # need to iteratively compute the remaining sequence of targets and the next action to take
            # I need to give the environment
            target_dist, action_dist = path_model(obs_input, target_mask, action_mask)
            # probs = route_model(torch.as_tensor(current_agent_position, dtype=torch.float32), torch.as_tensor(target_positions, dtype=torch.float32), torch.as_tensor(mask, dtype=torch.bool)).probs
            action_probs = action_dist.probs.detach().cpu().numpy().flatten()
            # print(f"action_probs: {action_probs}")
            # action_probs = action_probs.detach().cpu().numpy().flatten()

            # target_probs = target_dist.probs
            # target_probs = target_probs.detach().cpu().numpy().flatten()

            top_k = heapq.nlargest(beam_width, range(len(action_probs)), key=action_probs.__getitem__)

            for idx in top_k:

                prob = action_probs[idx]
                # current_target_position
                # print(f"proposed action: {idx}")
                print(f"proposed action: {idx}")
                print(f"action_mask: {temp_env.action_mask[idx]}")
                print(f"prob: {prob}")
                if prob > 0 and not temp_env.action_mask[idx]:
                    print(f"accepted: {idx}")
                    new_temp_env = copy.deepcopy(temp_env)
                    obs, action_mask, rew, done, target_mask = new_temp_env.action_step(idx)
                    # new_temp_env.render()
                    # reset_flag = new_temp_env.reset_flag
                    # distance_to_target = temp_env.get_distance_to_target()*env.grid_dim
                    # if distance_to_target < 0.5 / env.grid_dim:
                    #     distance_to_target = 0

                    agent_position = new_temp_env.agent_position

                    new_distance_to_target, new_estimated_remaining_tour_distance = get_estimate_of_remaining_tour_distance(new_temp_env, path_model)

                    print(f"new_distance_to_target (proposed trajectory): {new_distance_to_target}")
                    print(f"new_estimated_remaining_tour_distance (proposed trajectory): {new_estimated_remaining_tour_distance}")

                    # distance_from_start = abs(position_path[0][0] - agent_position[0]) + abs(
                    #     position_path[0][1] - agent_position[1])
                    # new_temp_env = copy.deepcopy(temp_env)
                    # if prob < 0.01:
                    #     prob = 0.01

                    new_beam.append((neg_log_prob - np.log(prob), new_distance_to_target, done, position_path + [agent_position], new_temp_env, action_sequence + [idx], reset_flag, new_estimated_remaining_tour_distance))

                # else:
                    # print(f"rejected: {idx}")

        for trajectory in new_beam:
            print(
                f"proposed trajectory (new beam): {trajectory[5]}, distance_to_target: {trajectory[1]}, done: {trajectory[2]}, reset_flag: {trajectory[6]}, estimated_remaining_tour_distance: {trajectory[7]}")
            # print(f"proposed trajectory (new beam): {trajectory[5]}, reset_flag: {trajectory[6]}, distance_to_target: {trajectory[1]}")

        # new_beam = [trajectory for trajectory in new_beam if not trajectory[6]]
        # Sort new_beam to prioritize trajectories with done = True
        # new_beam.sort(key=lambda x: (x[2], x[0]))

        # print(f"new_beam: {new_beam}")
        beam = heapq.nsmallest(beam_width, new_beam, key=lambda x: x[0])
        # backup_beam = heapq.nlargest(beam_width, new_beam, key=lambda x: x[2])


    for trajectory in final_trajectories:
        # print(f"trajectory: {trajectory}")
        print(f"proposed trajectory (final trajectory): {trajectory[5]}, distance_to_target: {trajectory[1]}, done: {trajectory[2]}, reset_flag: {trajectory[6]}, estimated_remaining_tour_distance: {trajectory[7]}")
        # print(f"proposed trajectory (final trajectory): {trajectory[4]}, reset_flag: {trajectory[5]}, distance_to_target: {trajectory[1]}")
        # print(f"reset: {trajectory[5]}")
        # print(f"distance_to_target: {trajectory[1]}")
    # print(f"final_trajectories: {final_trajectories}")


    best_trajectory = min(final_trajectories, key=lambda x: x[1] + len(x[5]) + x[7])
    print(f"best trajectory: {best_trajectory}")

    # path_length = len(best_trajectory[3])
    distance_to_target = best_trajectory[1]
    estimated_remaining_tour_distance = best_trajectory[7]
    action_sequence = best_trajectory[5]

    print("BEST TRAJECTORY")
    # negative log prob, distance_to_target, done, position_path, env, action_sequence, reset_flag, estimated_remaining_distance
    # input("Press Enter to continue...")
    return distance_to_target, best_trajectory[3], action_sequence, estimated_remaining_tour_distance