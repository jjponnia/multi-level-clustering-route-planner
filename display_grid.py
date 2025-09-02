import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from sympy.printing.pretty.pretty_symbology import line_width


# Create a figure and axis


# Set the limits of the plot
def display_grid(ax, grid_dim):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Set the aspect of the plot to be equal
    ax.set_aspect('equal')

    base_width = 30
    line_width = base_width / grid_dim

    # Draw the grid lines
    for i in range(grid_dim):
        ax.axhline(i / grid_dim, color='lightgrey', linewidth=line_width)
        ax.axvline(i / grid_dim, color='lightgrey', linewidth=line_width)

    # Add labels to the grid cells
    # for i in range(3):
    #     for j in range(3):
    #         ax.text(i + 0.5, j + 0.5, f'({i},{j})', ha='center', va='center')

    # Set the ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_xticks(np.linspace(0, 1, 7))
    # ax.set_yticks(np.linspace(0, 1, 7))
    # ax.set_xticklabels(np.round(np.linspace(0, 1, 4), 2))
    # ax.set_yticklabels(np.round(np.linspace(0, 1, 4), 2))

def display_agents(ax, list_of_agents, grid_dim):
    # Draw the agent
    # x, y = agent_position
    # ax.text(x, y + 0.1, f'({x:.2f},{y:.2f})', ha='center', va='center')

    # ax.text(x, y + 0.05, next_action, ha='center', va='center')
    # print(f"grid_dim: {grid_dim}")
    index = 0
    marker_size = 120 / grid_dim
    text_size = 400 / grid_dim
    for agent in list_of_agents:
        ax.plot(agent.position[0], agent.position[1], 'bo', markersize=marker_size)
        ax.text(agent.position[0], agent.position[1] + 0.01, f'{index}', ha='center', va='bottom', fontsize=text_size)
        index += 1

        for i in range(grid_dim):
            for j in range(grid_dim):
                if agent.grid_mask[i, j, 1] == 1:
                    ax.add_patch(plt.Rectangle((i / grid_dim, j / grid_dim), 1 / grid_dim, 1 / grid_dim, fill=True,
                                               color='lightgrey'))


def display_targets(ax, target_positions, target_mask, grid_dim):
    # Draw the target

    index = 0
    marker_size = 120 / grid_dim
    text_size = 400 / grid_dim
    for target_position in target_positions:
        if target_mask[index]:
            ax.plot(target_position[0], target_position[1], 'g^', markersize=marker_size)
        else:
            ax.plot(target_position[0], target_position[1], 'r^', markersize=marker_size)

        ax.text(target_position[0], target_position[1] + 0.01, f'{index}', ha='center', va='bottom', fontsize=text_size)
        index += 1


def display_obstacles(ax, obstacle_position, grid_dim):
    x, y = obstacle_position
    ax.add_patch(plt.Rectangle((x, y), 1 / grid_dim, 1 / grid_dim, fill=True, color='gray'))

    x_top_right = x + 1 / grid_dim
    y_top_right = y + 1 / grid_dim

    ax.plot([x, x_top_right], [y, y_top_right], color='black', linewidth=0.5)
    ax.plot([x_top_right, x], [y, y_top_right], color='black', linewidth=0.5)

def display_cluster_heads(ax, agents, grid_dim):
    clusterHeads = [agent for agent in agents if agent.clusterHeadToken is not None]
    clusterHeads = sorted(clusterHeads, key=lambda x: x.clusterHeadToken.level_)
    line_width = 60 / grid_dim
    circle_radius = 0.75 * (1 / grid_dim)

    for clusterHead in clusterHeads:
        # Draw a circle around the cluster-head
        level = clusterHead.clusterHeadToken.level_
        color = (255, 0, 0)

        if level == 2:
            color = (0, 255, 0)
        elif level == 3:
            color = (0, 0, 255)
        elif level == 4:
            color = (0, 165, 255)
        elif level == 5:
            color = (128, 0, 128)

        circle = Circle(clusterHead.position, radius=circle_radius, edgecolor = np.array(color) / 255, fill=False, linewidth=line_width)
        ax.add_patch(circle)


def display_cluster_lines(ax, agents, grid_dim):
    clusterHeads = [agent for agent in agents if agent.clusterHeadToken is not None]
    clusterHeads = sorted(clusterHeads, key=lambda x: x.clusterHeadToken.level_)
    line_width = 30 / grid_dim
    circle_radius = 0.45 * (1 / grid_dim)
    offset = 0.2 * (1 / grid_dim)

    for clusterHead in clusterHeads:
        # Draw a circle around the cluster-head
        level = clusterHead.clusterHeadToken.level_
        color = (255, 0, 0)

        if level == 2:
            color = (0, 255, 0)
        elif level == 3:
            color = (0, 0, 255)
        elif level == 4:
            color = (0, 165, 255)
        elif level == 5:
            color = (128, 0, 128)

        # circle = Circle(clusterHead.position, radius=circle_radius, edgecolor = np.array(color) / 255, fill=False, linewidth=line_width)
        # ax.add_patch(circle)

        for member in clusterHead.clusterHeadToken.members_:
            if clusterHead.clusterHeadToken.level_ == 1:
                if member.clusterHeadToken is None:
                    ax.plot([clusterHead.position[0], member.position[0]], [clusterHead.position[1], member.position[1]], color = np.array(color) / 255, linewidth=line_width)
                else:
                    x1, y1 = (clusterHead.position[0], clusterHead.position[1])
                    x2, y2 = (member.position[0], member.position[1])
                    v = (x2 - x1, y2 - y1)

                    # Find the perpendicular vector
                    v_perpendicular = (-v[1], v[0])
                    # Normalize the perpendicular vector
                    magnitude = (np.sqrt(v_perpendicular[0]**2 + v_perpendicular[1]**2))
                    if magnitude != 0:
                        v_perpendicular = (0.5 * circle_radius * v_perpendicular[0] / magnitude, 0.5 * circle_radius * v_perpendicular[1] / magnitude)

                    ax.plot([clusterHead.position[0] + v_perpendicular[0], member.position[0] + v_perpendicular[0]],
                            [clusterHead.position[1] + v_perpendicular[1], member.position[1] + v_perpendicular[1]], color=np.array(color) / 255,
                            linewidth=line_width)
            else:
                ax.plot([clusterHead.position[0], member.position[0]], [clusterHead.position[1], member.position[1]],
                        color=np.array(color) / 255, linewidth=line_width)
            # cv2.line(self.image, (int(th_center[0]), int(th_center[1])), (int(m_center[0]), int(m_center[1])), color)

