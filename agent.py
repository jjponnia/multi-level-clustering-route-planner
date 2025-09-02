import numpy as np
import itertools
from cluster_token import ClusterToken

from pyarrow import nulls


class Agent:
    def __init__(self, id):
        self.grid_dim = 150
        self.grid_state = np.zeros((self.grid_dim, self.grid_dim))
        self.position = np.array([0.5, 0.5])
        self.previous_position = None

        self.id = id

        self.n_acts = 5
        self.action_dict = self.init_action_dict()

        # action mask
        self.action_mask = [0 for _ in range(self.n_acts)]
        self.grid_mask = np.zeros((self.grid_dim, self.grid_dim, 2))

        self.next_target_id = None

        self.update_grid_mask()
        self.update_action_mask()
        self.reset_flag = False

        # how do I guarantee that every agent is a member of a level 1 cluster?
        # having a level 1 cluster token does NOT imply that the agent is a cluster head
        # if an agent has a level 1 cluster token, but doesn't have a cluster head token, it means that the agent is a member of a level 1 cluster but is not a cluster-head
        # the level of the agent in this case is 0, since it is not a cluster head
        self.lvl1Cluster = None
        # possession of a cluster head token implies that the agent is a cluster head of a cluster from level 1 and above
        # the level of an agent is always determined by the cluster head token it possesses
        # if an agent does not have a cluster head token, then the level of the agent is 0
        self.clusterHeadToken = None

        self.list_of_neighbors = []

        # this is a list of targets that the agent has been assigned to (if a level 1 cluster head)
        self.assigned_target_ids = []

    def is_lvl1_cluster_head(self):
        return self.clusterHeadToken is not None and self.clusterHeadToken.level_ == 1

    def get_nearest_neighbors(self, agents, threshold):
        self.list_of_neighbors = []


        for agent in agents:
            if self.id != agent.id:
                agent_position = agent.position
                my_position = self.position

                agent_position_cell = agent.convert_position_to_grid_cell(agent_position)
                my_position_cell = self.convert_position_to_grid_cell(my_position)

                distance = abs(agent_position_cell[0] - my_position_cell[0]) + abs(agent_position_cell[1] - my_position_cell[1])

                if distance <= threshold:
                    self.list_of_neighbors.append(agent.id)

        return self.list_of_neighbors


    def convert_position_to_grid_cell(self, position):
        x, y = position
        x_cell = int((x * self.grid_dim))
        y_cell = int((y * self.grid_dim))
        return x_cell, y_cell


    def init_action_dict(self):
        action_value = 1 / self.grid_dim
        return{
            0: np.array([0, action_value]),
            1: np.array([0, -action_value]),
            2: np.array([-action_value, 0]),
            3: np.array([action_value, 0]),
            4: np.array([0, 0])
        }


    def update_grid_mask(self):
        if self.previous_position is not None:
            # print(f"previous_position: {self.previous_position}")
            # print(f"current_position: {self.position}")
            previous_x_cell, previous_y_cell = self.convert_position_to_grid_cell(self.previous_position)
            current_x_cell, current_y_cell = self.convert_position_to_grid_cell(self.position)

            # print(f"previous_x_cell: {previous_x_cell}, previous_y_cell: {previous_y_cell}")
            # print(f"current_x_cell: {current_x_cell}, current_y_cell: {current_y_cell}")

            displacement = np.abs(current_x_cell - previous_x_cell) + np.abs(current_y_cell - previous_y_cell)

            # print(f"displacement: {displacement}")

            if displacement > 0:
                # print("hello: agent.update_grid_mask()")
                x_cell, y_cell = self.convert_position_to_grid_cell(self.previous_position)
                self.grid_mask[x_cell, y_cell, 1] = 1


        for x_cord, y_cord in itertools.product(range(self.grid_dim), range(self.grid_dim)):
            if self.grid_mask[x_cord, y_cord, 0] <= 10 and self.grid_mask[x_cord, y_cord, 1] == 1:
                self.grid_mask[x_cord, y_cord, 0] += 1
            else:
                self.grid_mask[x_cord, y_cord, 0] = 0
                self.grid_mask[x_cord, y_cord, 1] = 0


    def update_action_mask(self):
        self.action_mask = [0 for _ in range(self.n_acts)]

        for action in range(self.n_acts):
            proposed_position = self.position + self.action_dict[action]

            x, y = proposed_position
            x_cell, y_cell = self.convert_position_to_grid_cell(proposed_position)

            if 0 <= x <= 1 and 0 <= y <= 1 and self.grid_state[self.convert_position_to_grid_cell(proposed_position)] == 0:
                if self.grid_mask[x_cell, y_cell, 1] == 1 and action != 4:
                    self.action_mask[action] = 1
            else:
                # print(f"action (update_action_mask): {action}")
                self.action_mask[action] = 1
                # print(f"action_mask (update_action_mask): {self.action_mask}")

        if sum(self.action_mask) == 4:
            self.grid_mask = np.zeros((self.grid_dim, self.grid_dim, 2))
            self.reset_flag = True
            self.update_action_mask()


    def reset(self):
        self.grid_mask = np.zeros((self.grid_dim, self.grid_dim, 2))
        self.assigned_target_ids = []
        self.previous_position = None
        self.update_grid_mask()
        self.update_action_mask()
