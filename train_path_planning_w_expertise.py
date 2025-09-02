import numpy as np


def get_next_target(new_target_mask, route):
    for target in route:
        if new_target_mask[0, target] == 0:  # Check if the target has not been visited
            return target
    return None  # Return None if all targets have been visited


def train_path_planning_w_expertise(env, path_planning_buf, logger, steps_per_epoch):

    # print(f"num_obstacles: {env.num_obstacles}")
    # print(f"num_agents: {env.num_agents}")
    # print(f"num_targets: {env.num_targets}")

    obs_size = 2 * env.num_obstacles + 2 * env.num_agents + 2 * env.num_targets + 1

    for e in range(steps_per_epoch):
        # print(f"epoch: {e}")
        obs = np.zeros((0, obs_size))
        mask = np.zeros((0, 4))
        target_mask = np.zeros((0, env.num_targets))
        route_buf = np.zeros(0)
        good_path = False
        action_sequence = None

        # env.render()

        while not good_path:
            result = env.get_optimal_path()
            if result is not None:
                action_sequence, route = result
                good_path = True
            else:
                env.reset()

        action_sequence = np.array(action_sequence)
        route = np.array(route)
        # route = route.reshape(1, env.num_targets)
        # next_target = route[0]

        new_obs= env.get_path_model_obs_input()
        new_obs = np.array(new_obs)
        new_obs = new_obs.reshape(1, obs_size)

        env.update_grid_mask()
        env.update_action_mask()
        new_mask = env.action_mask
        new_mask = np.array(new_mask)
        new_mask = new_mask.reshape(1, 4)

        env.update_target_mask()
        new_target_mask = env.target_mask
        new_target_mask = np.array(new_target_mask)
        new_target_mask = new_target_mask.reshape(1, env.num_targets)

        next_target = get_next_target(new_target_mask, route)

        obs = np.concatenate((obs, new_obs), axis=0)
        mask = np.concatenate((mask, new_mask), axis=0)
        target_mask = np.concatenate((target_mask, new_target_mask), axis=0)
        route_buf = np.concatenate((route_buf, np.array([next_target])), axis=0)

        # env.render()

        for action in action_sequence:
            new_obs, new_mask, _, done, new_target_mask = env.action_step(action)

            # print(f"new_obs: {new_obs}")
            new_obs = np.array(new_obs)
            new_obs = new_obs.reshape(1, obs_size)

            new_mask = np.array(new_mask)
            new_mask = new_mask.reshape(1, 4)

            new_target_mask = np.array(new_target_mask)
            new_target_mask = new_target_mask.reshape(1, env.num_targets)

            # env.render()

            next_target = get_next_target(new_target_mask, route)

            if not done:
                obs = np.concatenate((obs, new_obs), axis=0)
                mask = np.concatenate((mask, new_mask), axis=0)
                route_buf = np.concatenate((route_buf, np.array([next_target])), axis=0)
                target_mask = np.concatenate((target_mask, new_target_mask), axis=0)

                # print(f"obs: {new_obs}")
                # print(f"obs: {obs.shape}")
                # print(f"action_mask: {new_mask}")
                # print(f"mask: {mask.shape}")
                # print(f"target_mask: {new_target_mask}")
                # print(f"target_mask: {target_mask.shape}")
                # print(f"route_buf: {route_buf}")
                # print(f"route_buf: {route_buf.shape}")

        action_sequence = action_sequence.reshape(-1, 1)

        path_planning_buf.store(obs, mask, action_sequence, route_buf, target_mask)

        optimal_path_length = len(action_sequence)
        logger.store(OptPathLength=optimal_path_length)

        env.reset()

        # Perform PPO update!
        # im_update()