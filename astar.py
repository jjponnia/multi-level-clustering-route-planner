import heapq


def heuristic(a, b):
    """Calculate Manhattan distance as the heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star(grid, start, goal):
    """Finds the shortest path in a grid using A* search."""
    start = tuple(start)
    goal = tuple(goal)
    # print(f"grid: {grid}")
    # print(f"start: {start}")
    # print(f"goal: {goal}")

    num_x_cells, num_y_cells = len(grid), len(grid[0])
    open_set = []  # Priority queue
    heapq.heappush(open_set, (0, start))  # (f-score, position)
    came_from = {}  # To reconstruct path
    g_score = {start: 0}  # Cost from start to each position
    f_score = {start: heuristic(start, goal)}
    actions = {}

    directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
    # directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Up, Down, Left, Right
    action_names = [0, 1, 2, 3]  # Up, Down, Left, Right

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            action_sequence = []

            while current in came_from:
                path.append(current)
                # print(f"curent: {current}")
                # print(f"came_from: {came_from[current]}")
                action_sequence.append(actions[current])
                current = came_from[current]
            path.append(start)
            # return path[::-1]  # Reverse to get path from start to goal
            return action_sequence[::-1], path[::-1]  # Reverse to get path from start to goal

        for i, d in enumerate(directions):
            neighbor = (current[0] + d[0], current[1] + d[1])
            x_cell = int(neighbor[0])
            y_cell = int(neighbor[1])

            if 0 <= x_cell < num_x_cells and 0 <= y_cell < num_y_cells and grid[x_cell][y_cell] == 0:
                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    actions[neighbor] = action_names[i]
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found

def cooperative_a_star(grid, agents):
    paths = []
    reservations = {}

    for agent in agents:
        path = a_star(grid, agent.start, agent.target)
        if not path:
            return None
        paths.append(path)
        for t, pos in enumerate(path):
            if pos not in reservations:
                reservations[pos] = set()
            reservations[pos].add(t)

    for t in range(max(len(path) for path in paths)):
        for i, path in enumerate(paths):
            if t < len(path):
                pos = path[t]
                if t in reservations[pos] and len(reservations[pos]) > 1:
                    return None

    return paths


