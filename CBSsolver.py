import heapq
from collections import defaultdict, deque
from grid_world_env import gridworld_env
from clustering_route_planner import ClusteringTSP

import pickle

import numpy as np


class CBS:
    def __init__(self, grid, starts, goals):
        """
        Initialize the CBS solver.
        :param grid: 2D list representing the grid (0 = free, 1 = obstacle).
        :param starts: List of tuples representing agents' start positions.
        :param goals: List of tuples representing agents' goal positions.
        """
        self.grid = grid
        self.starts = starts
        self.goals = goals
        self.num_agents = len(starts)
        self.max_possible_steps = self.grid.shape[0] * self.grid.shape[1]

    def is_valid(self, position):
        """Check if a position is valid (within bounds and not an obstacle)."""
        x, y = position
        return 0 <= x < len(self.grid) and 0 <= y < len(self.grid[0]) and self.grid[x][y] == 0

    def get_neighbors(self, position):
        """Get valid neighbors of a position."""
        x, y = position
        neighbors = [(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]]
        return [n for n in neighbors if self.is_valid(n)]

    def display_open_list(self, open_list):
        print("Current open_list contents:")
        for f_cost, position, g_cost, path in open_list:
            print(f"f_cost: {f_cost}, position: {position}, g_cost: {g_cost}")

    def display_closed_set(self, closed_set):
        print("Current closed_set contents:")
        for position, g_cost in closed_set:
            print(f"position: {position}, g_cost: {g_cost}")

    def a_star(self, start, goal, constraints):
        """Perform A* search for a single agent with constraints."""
        open_list = []
        heapq.heappush(open_list, (0, start, 0, [], [], False))  # (f_cost, position, g_cost, path, action_sequence, goal_reached)
        closed_set = set()

        directions = [(0, 1), (0, -1), (-1, 0), (1, 0), (0, 0)]  # Up, Down, Left, Right, Stay
        # directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Up, Down, Left, Right
        action_names = [0, 1, 2, 3, 4]  # Up, Down, Left, Right, Stay

        max_constraint_time = max((time for _, time in constraints), default=0)

        while open_list:

            f_cost, current, g_cost, path, action_sequence, goal_reached = heapq.heappop(open_list)

            # action_sequence = []

            if current == goal:
                goal_reached = True
                # print(f"Goal reached: {current}")
                # input("Press Enter to continue...")
                # return path + [current], action_sequence

            if goal is None or goal_reached:
                if g_cost >= max_constraint_time:
                    return path + [current], action_sequence

            if (current, g_cost) in closed_set:
                continue
            else:
                closed_set.add((current, g_cost))

            for i, d in enumerate(directions):
                neighbor = (current[0] + d[0], current[1] + d[1])
                if self.is_valid(neighbor):

                    if (neighbor, g_cost + 1) in constraints:
                        continue
                    new_path = path + [current]
                    # print(f"action_sequence: {action_sequence}")
                    new_action_sequence = action_sequence + [action_names[i]]
                    # print(f"new_action_sequence: {new_action_sequence}")

                    if goal is not None and not goal_reached: # if there is a goal that hasn't been reached, minimize time-elapsed and distance from goal
                        h_cost = abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])  # Manhattan distance
                        heapq.heappush(open_list,
                                       (g_cost + 1 + h_cost, neighbor, g_cost + 1, new_path, new_action_sequence,
                                        goal_reached))
                    elif d != (0, 0): # if goal is None or goal_reached then prioritize minimal displacement
                        heapq.heappush(open_list,
                                       (f_cost + 1, neighbor, g_cost + 1, new_path, new_action_sequence,
                                        goal_reached))


        print(f"No valid path found for start {start} to goal {goal} with constraints: {constraints}")
        # input("Press Enter to continue...")
        return None, None  # No valid path found

    def detect_conflict(self, paths):
        """Detect the first conflict between agents' paths."""
        max_time = max(len(path) for path in paths)
        for t in range(max_time):
            positions = {}
            for agent_id, path in enumerate(paths):
                pos = path[t] if t < len(path) else path[-1]  # Stay at goal if path is shorter
                if pos in positions:
                    return t, agent_id, positions[pos]
                positions[pos] = agent_id
        return None

    def plan_paths(self):
        """Plan collision-free paths for all agents using CBS."""
        root = {"paths": [], "action_sequence": [], "constraints": defaultdict(set)}
        # I need to pad the paths so they are all the same length

        for i in range(self.num_agents):
            # if the goal location is none set the path to the current position and action sequence to null actions
            if self.goals[i] is None:
                path = [self.starts[i]]
                action_sequence = [4] # Null action (e.g., "Stay")
            else:
                path, action_sequence = self.a_star(self.starts[i], self.goals[i], root["constraints"][i])

            if not path:
                return None  # No solution

            root["paths"].append(path)
            root["action_sequence"].append(action_sequence)

        # make all the paths the same length by padding with null actions
        max_length = max(len(path) for path in root["paths"])
        # print(f"path for agent 23: {root['paths'][23]}")
        # flag = False
        for i in range(self.num_agents):
            while len(root["paths"][i]) < max_length:
                # if i==23:
                #     print(f"Padding path for agent {i} from length {len(root['paths'][i])} to {max_length}")
                #     flag = True
                root["paths"][i].append(root["paths"][i][-1])  # Repeat the last position
                root["action_sequence"][i].append(4)  # Null action (e.g., "Stay")

        # if flag:
        #     print(f"Padded path for agent 23: {root['paths'][23]}")
        #     input("Press Enter to continue...")


        open_list = []
        node_id = 0
        heapq.heappush(open_list, ((0, node_id), root))  # (cost, node)
        node_id += 1

        while open_list:
            (temp_cost, temp_node_id), node = heapq.heappop(open_list)
            if temp_cost > len(self.grid) * len(self.grid[0]):
                return None

            conflict = self.detect_conflict(node["paths"])
            # print(f"conflict: {conflict}")
            if not conflict:
                return node["paths"], node["action_sequence"]  # No conflicts, return paths

            t, a1, a2 = conflict
            print(f"Conflict detected at time {t} between agents {a1} and {a2}")
            print(f"Path of agent {a1}: {node['paths'][a1]}")
            print(f"Path of agent {a2}: {node['paths'][a2]}")


            for agent in [a1, a2]:
                # print(f"constrained agent: {agent}")
                new_constraints = node["constraints"].copy()

                # if t < len(node["paths"][agent]):
                #     new_constraints[agent].add((node["paths"][agent][t], t))  # Add constraint
                # else:
                #     new_constraints[agent].add((node["paths"][agent][-1], t))

                try:
                    new_constraints[agent].add((node["paths"][agent][t], t))
                except IndexError as e:
                    print(f"IndexError: {e}")
                    print(f"node contents: {node}")
                    print(f"agent: {agent}")
                    print(f"t: {t}")
                    raise  # Re-raise the exception after logging

                # new_constraints[agent].add((node["paths"][agent][t], t))
                new_node = {"paths": node["paths"][:], "action_sequence": node["action_sequence"][:], "constraints": new_constraints}
                # print(f"constrained agent: {agent}")
                # print(f"New node created with constraints: {new_constraints[agent]}")
                new_path, action_sequence = self.a_star(self.starts[agent], self.goals[agent], new_constraints[agent])
                if new_path:
                    new_node["paths"][agent] = new_path
                    new_node["action_sequence"][agent] = action_sequence

                    cost = max(len(p) for p in new_node["paths"])

                    for i in range(self.num_agents):
                        while len(new_node["paths"][i]) < cost:
                            new_node["paths"][i].append(root["paths"][i][-1])  # Repeat the last position
                            new_node["action_sequence"][i].append(4)  # Null action (e.g., "Stay")

                    heapq.heappush(open_list, ((cost, node_id), new_node))
                    node_id += 1
                else:
                    print(f"No valid path for agent {agent} with constraints: {new_constraints[agent]}")

        return None  # No solution found


def save_environment(env, filename):
    with open(filename, 'wb') as file:
        pickle.dump(env, file)
    print(f"Environment saved to {filename}")


def load_environment(filename):
    with open(filename, 'rb') as file:
        env = pickle.load(file)
    print(f"Environment loaded from {filename}")
    return env


if __name__ == "__main__":
    grid = [
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]
    starts = [(0, 0), (3, 0)]
    goals = [(3, 3), (0, 3)]

    # Load the environment
    # env = load_environment('environment.pkl')

    env = gridworld_env()
    env.render()

    # Save the environment to a file
    save_environment(env, 'environment2.pkl')

    # gives the positions as coordinates in a 1x1 grid
    agent_positions = env.agent_positions
    target_positions = env.target_positions

    agent_cell_positions = []
    target_cell_positions = []

    # gives the positions as coordinates in a nxn grid
    for position in agent_positions:
        agent_cell_position = env.convert_position_to_grid_cell(position)
        agent_cell_positions += [agent_cell_position]

    for position in target_positions:
        target_cell_position = env.convert_position_to_grid_cell(position)
        target_cell_positions += [target_cell_position]
        # agent_cell_positions = np.append(agent_cell_positions, agent_cell_position)

    # print(agent_cell_positions)

    # agent_positions = env.convert_position_to_grid_cell(agent_positions)
    # target_positions = env.convert_position_to_grid_cell(target_positions)
    env_grid = env.grid_state
    print(f"Grid state: {env_grid}")
    print(f"Agent positions: {agent_cell_positions}")
    print(f"Target positions: {target_cell_positions}")

    route_solver = ClusteringTSP(env)
    distance_matrix, num_agents, manager, routing, solution = route_solver.solve_open_ended_vrp()
    routes = route_solver.get_routes_as_dict(manager, routing, solution, num_agents)

    print(f"routes: {routes}")
    first_targets = [route[0] for route in routes.values() if route]
    agents = [agent for agent, route in routes.items() if route]

    print(f"agents: {agents}")
    filtered_agent_cell_positions = [agent_cell_positions[i] for i in agents]
    print(f"filtered_agent_cell_positions: {filtered_agent_cell_positions}")
    # filtered_agent_cell_positions = []
    # for position in filtered_agent_positions:
    #     filtered_agent_cell_position = env.convert_position_to_grid_cell(position)
    #     filtered_agent_cell_positions += [filtered_agent_cell_position]

    print(f"first_targets: {first_targets}")
    filtered_target_cell_positions = [target_cell_positions[i] for i in first_targets]
    # print(f"filtered_target_positions: {filtered_target_positions}")
    # filtered_target_cell_positions = []

    # for position in filtered_target_positions:
    #     filtered_target_cell_position = env.convert_position_to_grid_cell(position)
    #     filtered_target_cell_positions += [filtered_target_cell_position]

    print(f"filtered_target_cell_positions: {filtered_target_cell_positions}")
    print(f"filtered_agent_cell_positions: {filtered_agent_cell_positions}")

    cbs_solver = CBS(env_grid, filtered_agent_cell_positions, filtered_target_cell_positions)

    print(f"starts: {cbs_solver.starts}")
    print(f"goals: {cbs_solver.goals}")
    paths = cbs_solver.plan_paths()
    if paths:
        for agent_id, path in enumerate(paths):
            print(f"Agent {agent_id} path: {path}")
    else:
        print(f"Collision-free paths: {paths}")

    # cbs_solver = CBS(env_grid, agent_cell_positions, target_cell_positions)

    # paths = cbs_solver.plan_paths()

    # if paths:
    #     for agent_id, path in enumerate(paths):
    #         print(f"Agent {agent_id} path: {path}")
    # else:
    #     print(f"Collision-free paths: {paths}")