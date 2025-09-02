from grid_world_env import gridworld_env
from path_planning_model import PathPlanningModel
from evaluate_path_model_baseline import beam_search
from beam_search import combined_beam_search
import torch


# imitation7
# grid_dim = 7
# obstacle_dim = 3
# Average tour length: 6.3574
# Average optimal tour length: 6.1632
# Performance ratio: 0.9694529210054424

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env = gridworld_env()
obs_dim = 2 * env.num_obstacles + 2 * env.num_agents + 2 * env.num_targets + 1

path_model = PathPlanningModel(obs_dim=obs_dim, n_targets=env.num_targets, n_actions=env.n_acts).to(device)
path_model.load_state_dict(torch.load('imitation_model8.pth'))

# Function to evaluate the model on a single instance
def evaluate_instance(temp_env, model, beam_width=24):
    target_sequence, position_path, proposed_action_sequence = combined_beam_search(temp_env, model, beam_width=beam_width)

    temp_optimal_action_sequence, _ = env.get_optimal_path()
    # # print(f"proposed_action_sequence: {proposed_action_sequence}")
    return proposed_action_sequence, temp_optimal_action_sequence

# Generate and evaluate 500 random TSP instances
num_instances = 100
total_length = 0
total_optimal_length = 0

for i in range(num_instances):
    print(f"Instance {i}")
    # print("NEW INSTANCE")
    env = gridworld_env()
    # env.render()
    good_path = False
    while not good_path:
        # print("Finding optimal path")
        result = env.get_optimal_path()
        # if e==278:
        #     print(f"result: {result}")
        if result is not None:
            action_sequence, path = result
            # print(f"action_sequence: {action_sequence}")
            good_path = True
        else:
            env.reset()

    done = False
    path_length = 0
    # path_length, optimal_length, distance_to_next_target, action_sequence, estimated_remaining_tour_distance
    # _, _, distance_to_next_target, action_sequence, estimated_remaining_tour_distance
    action_sequence, optimal_action_sequence = evaluate_instance(env, path_model)
    # print("EVALUATE INSTANCE COMPLETE")

    # print(f"optimal_length: {optimal_length}")
    # input("Press Enter to continue...")
    # # print(f"action_sequence: {action_sequence}")
    # for col in reversed(env.grid_mask[:, :, 1].T):
    #     print(col)

    # env.render()

    # escape_count = 0
    while not done and path_length < 4*env.grid_dim:
        action = action_sequence[0]
        # print(f"action_taken: {action}")
        obs, action_mask, _, done, target_mask = env.action_step(action)
        # escape_count += 1
        # for col in reversed(env.grid_mask[:, :, 1].T):
        #     print(col)

        # env.render()
        path_length += 1
        if not done:
            # temp_path_length, temp_optimal_length, temp_distance_to_next_target, action_sequence, temp_estimated_remaining_tour_distance
            action_sequence, _ = evaluate_instance(env, path_model)
            # # print(f"action_sequence: {action_sequence}")
            # print(f"grid_mask_count: {env.grid_mask_count}")


    # # print(f"path_length: {path_length}")
    optimal_length = len(optimal_action_sequence)
    # # print(f"optimal_length: {optimal_length}")
    # input("Press Enter to continue...")
    total_length += path_length
    total_optimal_length += optimal_length

# Calculate average performance
average_length = total_length / num_instances
average_optimal_length = total_optimal_length / num_instances

print(f"Average tour length: {average_length}")
print(f"Average optimal tour length: {average_optimal_length}")
print(f"Performance ratio: {average_optimal_length / average_length}")

