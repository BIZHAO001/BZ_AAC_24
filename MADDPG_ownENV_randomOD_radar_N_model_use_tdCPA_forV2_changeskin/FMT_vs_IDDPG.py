import time

import matplotlib
#_____________________static FMT for multiple aircaft________________
# import numpy as np
# import matplotlib.pyplot as plt
# import random
# from scipy.spatial import distance
#
#
# # Function to generate random nodes
# def generate_random_nodes(num_nodes, x_lim, y_lim, obstacle_list):
#     nodes = []
#     while len(nodes) < num_nodes:
#         x = random.uniform(x_lim[0], x_lim[1])
#         y = random.uniform(y_lim[0], y_lim[1])
#         if not is_in_obstacle([x, y], obstacle_list):
#             nodes.append([x, y])
#     return np.array(nodes)
#
#
# # Function to check if a point is inside an obstacle
# def is_in_obstacle(point, obstacle_list):
#     for (ox, oy, size) in obstacle_list:
#         if (point[0] - ox) ** 2 + (point[1] - oy) ** 2 <= size ** 2:
#             return True
#     return False
#
#
# # Function to find the nearest neighbor nodes to a node
# def nearest_neighbors(nodes, target_node, radius):
#     distances = distance.cdist(nodes, [target_node])
#     return np.where(distances <= radius)[0]
#
#
# # Function to check if there is a collision-free path between two points
# def is_collision_free(node1, node2, obstacle_list, step_size=0.1):
#     steps = int(np.linalg.norm(np.array(node2) - np.array(node1)) / step_size)
#     for i in range(steps):
#         interp = node1 + (i / steps) * (np.array(node2) - np.array(node1))
#         if is_in_obstacle(interp, obstacle_list):
#             return False
#     return True
#
#
# # FMT* algorithm implementation
# def fast_marching_tree(start, goal, num_nodes, x_lim, y_lim, obstacle_list, radius=10):
#     nodes = generate_random_nodes(num_nodes, x_lim, y_lim, obstacle_list)
#     nodes = np.vstack([start, nodes, goal])  # Include start and goal in the set of nodes
#
#     tree = {0: None}  # Tree structure (node_index: parent_node_index)
#     open_set = {0}  # Set of nodes to be explored
#     closed_set = set()  # Set of explored nodes
#
#     while open_set:
#         current_node = min(open_set, key=lambda node: np.linalg.norm(nodes[node] - goal))
#         open_set.remove(current_node)
#         closed_set.add(current_node)
#
#         if np.linalg.norm(nodes[current_node] - goal) <= radius:
#             # Ensure we add the goal node to the tree with its parent
#             tree[len(nodes) - 1] = current_node
#             break
#
#         neighbors = nearest_neighbors(nodes, nodes[current_node], radius)
#         for neighbor in neighbors:
#             if neighbor not in closed_set and is_collision_free(nodes[current_node], nodes[neighbor], obstacle_list):
#                 if neighbor not in tree:
#                     tree[neighbor] = current_node
#                     open_set.add(neighbor)
#
#     # Check if goal node was added successfully
#     if len(nodes) - 1 not in tree:
#         raise Exception("Goal node was not connected to the tree!")
#
#     # Extract path from tree
#     path = []
#     node = len(nodes) - 1  # Goal node
#     while node is not None:
#         path.append(nodes[node])
#         node = tree.get(node)  # Use .get() to avoid KeyError in case of an issue
#
#     return path[::-1]
#
#
# # Visualize the trajectory planning result for multiple aircraft
# def plot_paths(trajectories, obstacle_list, x_lim, y_lim):
#     plt.figure()
#     # Plot obstacles
#     for (ox, oy, size) in obstacle_list:
#         circle = plt.Circle((ox, oy), size, color='red')
#         plt.gca().add_patch(circle)
#
#     # Plot paths for all aircraft
#     for OD_name, path in trajectories.items():
#         path = np.array(path)
#         plt.plot(path[:, 0], path[:, 1], '-o', label=f'Trajectory {OD_name}')
#
#     plt.xlim(x_lim)
#     plt.ylim(y_lim)
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('Multiple Aircraft Trajectories with Static Obstacles')
#     plt.legend()
#     plt.show()
#
#
# # Example usage for multiple aircraft
# if __name__ == "__main__":
#     # Parameters
#     num_nodes = 5000  # sampled nodes from start to end
#     radius = 20  # connection threshold of nearby node
#     x_lim = [0, 200]
#     y_lim = [0, 200]
#     obstacle_list = [(30, 30, 20), (100, 100, 25), (160, 160, 30)]
#
#     # Define multiple OD locations in a dictionary
#     # The dictionary has the format: 'OD_name': (start_point, goal_point)
#     OD_dict = {
#         'OD1': (np.array([0, 0]), np.array([200, 200])),
#         'OD2': (np.array([0, 200]), np.array([200, 0])),
#         'OD3': (np.array([50, 0]), np.array([150, 200])),
#         'OD4': (np.array([200, 50]), np.array([0, 150]))  # Added fourth aircraft
#     }
#
#     # Store the trajectories for all aircraft
#     trajectories = {}
#     for OD_name, (start, goal) in OD_dict.items():
#         path = fast_marching_tree(start, goal, num_nodes, x_lim, y_lim, obstacle_list, radius)
#         trajectories[OD_name] = path
#
#     # Plot all trajectories
#     plot_paths(trajectories, obstacle_list, x_lim, y_lim)





#______________________________________________dynamic FMT____________________
# import numpy as np
# import matplotlib.pyplot as plt
# import random
# from scipy.spatial import distance
#
#
# # Function to check if a point is inside an obstacle
# def is_in_obstacle(point, obstacle_pos, size):
#     ox, oy = obstacle_pos
#     return (point[0] - ox) ** 2 + (point[1] - oy) ** 2 <= size ** 2
#
#
# # Function to find the nearest neighbor nodes to a node
# def nearest_neighbors(nodes, target_node, radius):
#     distances = distance.cdist(nodes, [target_node])
#     return np.where(distances <= radius)[0]
#
#
# # Function to generate random nodes
# def generate_random_nodes(num_nodes, x_lim, y_lim, obstacle_pos, obstacle_size):
#     nodes = []
#     while len(nodes) < num_nodes:
#         x = random.uniform(x_lim[0], x_lim[1])
#         y = random.uniform(x_lim[0], x_lim[1])
#         if not is_in_obstacle([x, y], obstacle_pos, obstacle_size):
#             nodes.append([x, y])
#     return np.array(nodes)
#
#
# # Function to check if there is a collision-free path between two points
# def is_collision_free(node1, node2, obstacle_pos, obstacle_size, step_size=0.05):
#     steps = int(np.linalg.norm(np.array(node2) - np.array(node1)) / step_size)
#     for i in range(steps):
#         interp = node1 + (i / steps) * (np.array(node2) - np.array(node1))
#         if is_in_obstacle(interp, obstacle_pos, obstacle_size):
#             return False
#     return True
#
#
# # Simulate obstacle moving 1 unit per timestep
# def dynamic_obstacle_movement(start_pos, end_pos, t, obstacle_speed=1):
#     total_distance = np.linalg.norm(end_pos - start_pos)
#     direction = (end_pos - start_pos) / total_distance
#     step_distance = obstacle_speed  # Move 1 unit per timestep
#     return start_pos + direction * step_distance * t
#
#
# # Replan the path at each timestep based on the current obstacle position
# def plan_path(current_pos, goal, num_nodes, x_lim, y_lim, obstacle_pos, obstacle_size, radius):
#     # Generate random nodes and add the current and goal positions
#     nodes = generate_random_nodes(num_nodes, x_lim, y_lim, obstacle_pos, obstacle_size)
#     nodes = np.vstack([current_pos, nodes, goal])  # Include current and goal in the set of nodes
#
#     # Tree structure to store the path from current position to goal
#     tree = {0: None}  # node_index: parent_node_index
#     open_set = {0}  # Set of nodes to be explored
#     closed_set = set()  # Set of explored nodes
#
#     while open_set:
#         current_node = min(open_set, key=lambda node: np.linalg.norm(nodes[node] - goal))
#         open_set.remove(current_node)
#         closed_set.add(current_node)
#
#         if np.linalg.norm(nodes[current_node] - goal) <= radius:
#             tree[len(nodes) - 1] = current_node
#             break
#
#         neighbors = nearest_neighbors(nodes, nodes[current_node], radius)
#         for neighbor in neighbors:
#             if neighbor not in closed_set and is_collision_free(nodes[current_node], nodes[neighbor], obstacle_pos,
#                                                                 obstacle_size, step_size=0.05):
#                 if neighbor not in tree:
#                     tree[neighbor] = current_node
#                     open_set.add(neighbor)
#
#     # Extract the path from the tree
#     path = []
#     node = len(nodes) - 1  # Goal node
#     while node is not None:
#         path.append(nodes[node])
#         node = tree.get(node)
#
#     return path[::-1]
#
#
# # Dynamic FMT* algorithm implementation
# def dynamic_flight_path(start, goal, num_nodes, x_lim, y_lim, obstacle_start, obstacle_end, t_max, obstacle_speed=1,
#                         radius=10):
#     flown_trajectory = [start]  # List to store the actual flown trajectory
#     obstacle_trajectory = []  # List to store the obstacle's movement
#     current_pos = start  # Current position of the aircraft
#     obstacle_size = 20  # Obstacle size
#     t = 0  # Timestep counter
#
#     # Initial path planning from start to goal
#     current_path = plan_path(current_pos, goal, num_nodes, x_lim, y_lim, obstacle_start, obstacle_size, radius)
#
#     while t < t_max and np.linalg.norm(current_pos - goal) > 0:
#         t += 1
#
#         # Move the obstacle by 1 unit per timestep
#         obstacle_pos = dynamic_obstacle_movement(obstacle_start, obstacle_end, t, obstacle_speed)
#         obstacle_trajectory.append(obstacle_pos)
#
#         # Aircraft moves to the next node in the current path
#         if len(current_path) > 1:
#             next_node = current_path[1]  # Move to the next node
#             current_pos = next_node
#             flown_trajectory.append(current_pos)
#
#         # Re-plan the path based on the updated obstacle position
#         if np.linalg.norm(current_pos - goal) > 0:
#             current_path = plan_path(current_pos, goal, num_nodes, x_lim, y_lim, obstacle_pos, obstacle_size, radius)
#
#         # Break the loop if the aircraft has reached the goal
#         if np.linalg.norm(current_pos - goal) <= 0:
#             break
#
#     # After reaching the goal, append the goal to the flown trajectory
#     flown_trajectory.append(goal)
#
#     return flown_trajectory, obstacle_trajectory
#
#
# # Visualization of the dynamic trajectory
# def plot_dynamic_trajectory(flown_trajectory, obstacle_trajectory, obstacle_size, x_lim, y_lim):
#     plt.figure()
#
#     # Plot flown trajectory
#     flown_trajectory = np.array(flown_trajectory)
#     plt.plot(flown_trajectory[:, 0], flown_trajectory[:, 1], '-o', label='Flown Trajectory')
#
#     # Plot obstacle's trajectory
#     obstacle_trajectory = np.array(obstacle_trajectory)
#     plt.plot(obstacle_trajectory[:, 0], obstacle_trajectory[:, 1], 'r--', label='Obstacle Trajectory')
#
#     # Plot obstacle at each timestep
#     for obs in obstacle_trajectory:
#         circle = plt.Circle(obs, obstacle_size, color='red', fill=True, alpha=0.3)
#         plt.gca().add_patch(circle)
#
#     plt.xlim(x_lim)
#     plt.ylim(y_lim)
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('Dynamic Obstacle and Aircraft Trajectory')
#     plt.legend()
#     plt.show()
#
#
# # Example usage
# if __name__ == "__main__":
#     # Parameters
#     start = np.array([0, 0])
#     goal = np.array([200, 200])
#     num_nodes = 2000  # Sampled nodes from start to end
#     radius = 10  # Connection threshold of nearby node
#     x_lim = [0, 200]
#     y_lim = [0, 200]
#
#     # Obstacle moves from one point to another within boundary
#     obstacle_start = np.array([100, 100])
#     obstacle_end = np.array([0, 0])
#
#     t_max = 80  # Manually adjust the number of time steps
#
#     # Run the dynamic FMT* algorithm
#     flown_trajectory, obstacle_trajectory = dynamic_flight_path(
#         start, goal, num_nodes, x_lim, y_lim, obstacle_start, obstacle_end, t_max, obstacle_speed=1, radius=10
#     )
#
#     # Plot the result
#     plot_dynamic_trajectory(flown_trajectory, obstacle_trajectory, 20, x_lim, y_lim)


#_________________________________________________FMT dynamic obstacle with mutiple aircraft____________
# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# import random
# from scipy.spatial import distance
#
#
# # Function to check if a point is inside an obstacle
# def is_in_obstacle(point, obstacle_pos, size):
#     ox, oy = obstacle_pos
#     return (point[0] - ox) ** 2 + (point[1] - oy) ** 2 <= size ** 2
#
#
# # Function to find the nearest neighbor nodes to a node
# def nearest_neighbors(nodes, target_node, radius):
#     distances = distance.cdist(nodes, [target_node])
#     return np.where(distances <= radius)[0]
#
#
# # Function to generate random nodes
# def generate_random_nodes(num_nodes, x_lim, y_lim, obstacle_pos, obstacle_size):
#     nodes = []
#     while len(nodes) < num_nodes:
#         x = random.uniform(x_lim[0], x_lim[1])
#         y = random.uniform(x_lim[0], x_lim[1])
#         if not is_in_obstacle([x, y], obstacle_pos, obstacle_size):
#             nodes.append([x, y])
#     return np.array(nodes)
#
#
# # Function to check if there is a collision-free path between two points
# def is_collision_free(node1, node2, obstacle_pos, obstacle_size, step_size=0.05):
#     steps = int(np.linalg.norm(np.array(node2) - np.array(node1)) / step_size)
#     for i in range(steps):
#         interp = node1 + (i / steps) * (np.array(node2) - np.array(node1))
#         if is_in_obstacle(interp, obstacle_pos, obstacle_size):
#             return False
#     return True
#
#
# # Simulate obstacle moving 1 unit per timestep
# def dynamic_obstacle_movement(start_pos, end_pos, t, obstacle_speed=1):
#     total_distance = np.linalg.norm(end_pos - start_pos)
#     direction = (end_pos - start_pos) / total_distance
#     step_distance = obstacle_speed
#     return start_pos + direction * step_distance * t
#
#
# # Replan the path at each timestep based on the current obstacle position
# def plan_path(current_pos, goal, num_nodes, x_lim, y_lim, obstacle_pos, obstacle_size, radius):
#     nodes = generate_random_nodes(num_nodes, x_lim, y_lim, obstacle_pos, obstacle_size)
#     nodes = np.vstack([current_pos, nodes, goal])  # Include current and goal in the set of nodes
#
#     tree = {0: None}  # node_index: parent_node_index
#     open_set = {0}  # Set of nodes to be explored
#     closed_set = set()  # Set of explored nodes
#
#     while open_set:
#         current_node = min(open_set, key=lambda node: np.linalg.norm(nodes[node] - goal))
#         open_set.remove(current_node)
#         closed_set.add(current_node)
#
#         if np.linalg.norm(nodes[current_node] - goal) <= radius:
#             tree[len(nodes) - 1] = current_node
#             break
#
#         neighbors = nearest_neighbors(nodes, nodes[current_node], radius)
#         for neighbor in neighbors:
#             if neighbor not in closed_set and is_collision_free(nodes[current_node], nodes[neighbor], obstacle_pos,
#                                                                 obstacle_size, step_size=0.05):
#                 if neighbor not in tree:
#                     tree[neighbor] = current_node
#                     open_set.add(neighbor)
#
#     # Extract the path from the tree
#     path = []
#     node = len(nodes) - 1  # Goal node
#     while node is not None:
#         path.append(nodes[node])
#         node = tree.get(node)
#
#     return path[::-1]
#
#
# # Dynamic FMT* algorithm implementation for multiple aircraft
# def dynamic_flight_path_multiple(aircraft_dict, num_nodes, x_lim, y_lim, obstacle_start, obstacle_end, t_max,
#                                  obstacle_speed=1, radius=10):
#     all_flown_trajectories = {}  # Dictionary to store flown trajectories for each aircraft
#     obstacle_trajectory = []  # List to store the obstacle's movement
#
#     # Initialize positions for each aircraft
#     current_positions = {name: OD[0] for name, OD in aircraft_dict.items()}
#     flown_trajectories = {name: [OD[0]] for name, OD in aircraft_dict.items()}
#
#     t = 0  # Timestep counter
#
#     while t < t_max:
#         t += 1
#
#         # Move the obstacle by 1 unit per timestep
#         obstacle_pos = dynamic_obstacle_movement(obstacle_start, obstacle_end, t, obstacle_speed)
#         obstacle_trajectory.append(obstacle_pos)
#
#         # Loop through each aircraft
#         for name, (start, goal) in aircraft_dict.items():
#             current_pos = current_positions[name]
#             if np.linalg.norm(current_pos - goal) > 0:  # Check if the aircraft has reached the goal
#                 # Replan the path based on the updated obstacle position
#                 current_path = plan_path(current_pos, goal, num_nodes, x_lim, y_lim, obstacle_pos, obstacle_size=20,
#                                          radius=radius)
#
#                 # Aircraft moves to the next node in the path
#                 if len(current_path) > 1:
#                     next_node = current_path[1]
#                     current_positions[name] = next_node
#                     flown_trajectories[name].append(next_node)
#
#         # Break the loop when all aircraft have reached their goals
#         if all(np.linalg.norm(current_positions[name] - goal) <= 0 for name, (start, goal) in aircraft_dict.items()):
#             break
#
#     return flown_trajectories, obstacle_trajectory
#
#
# # Visualization of the dynamic trajectories for multiple aircraft
# def plot_dynamic_trajectories(aircraft_trajectories, obstacle_trajectory, obstacle_size, x_lim, y_lim):
#     plt.figure()
#
#     # Plot flown trajectories for each aircraft
#     for name, trajectory in aircraft_trajectories.items():
#         trajectory = np.array(trajectory)
#         plt.plot(trajectory[:, 0], trajectory[:, 1], linestyle='--', label=f'{name} Trajectory')
#         plt.scatter(trajectory[0, 0], trajectory[0, 1], marker='^', color='g', s=50)  # Start marker
#         plt.scatter(trajectory[-1, 0], trajectory[-1, 1], marker='*', color='r', s=50)  # Goal marker
#
#     # Plot obstacle's trajectory
#     obstacle_trajectory = np.array(obstacle_trajectory)
#     plt.plot(obstacle_trajectory[:, 0], obstacle_trajectory[:, 1], 'r--', label='Obstacle Trajectory')
#
#     # Plot obstacle at each timestep
#     for obs in obstacle_trajectory:
#         circle = plt.Circle(obs, obstacle_size, color='red', fill=True, alpha=0.3)
#         plt.gca().add_patch(circle)
#
#     plt.xlim(x_lim)
#     plt.ylim(y_lim)
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('Dynamic Obstacle and Multiple Aircraft Trajectories')
#     plt.legend()
#     plt.show()
#
#
# # Example usage for multiple aircraft
# if __name__ == "__main__":
#     # Parameters
#     num_nodes = 2000  # Sampled nodes from start to end
#     radius = 10  # Connection threshold of nearby node
#     x_lim = [0, 200]
#     y_lim = [0, 200]
#     obstacle_speed = 3
#
#     # Define multiple aircraft with OD locations in a dictionary
#     # The dictionary has the format: 'aircraft_name': (start_point, goal_point)
#     aircraft_dict = {
#         'AC1': (np.array([20, 180]), np.array([170, 20])),
#         'AC2': (np.array([20, 102]), np.array([185, 102])),
#         'AC3': (np.array([20, 30]), np.array([170, 180])),
#     }
#
#     # Obstacle moves from one point to another within boundary
#     obstacle_start = np.array([135, 160])
#     obstacle_end = np.array([80, 25])
#
#     t_max = 80  # Manually adjust the number of time steps
#
#     # Run the dynamic FMT* algorithm for multiple aircraft
#     aircraft_trajectories, obstacle_trajectory = dynamic_flight_path_multiple(
#         aircraft_dict, num_nodes, x_lim, y_lim, obstacle_start, obstacle_end, t_max, obstacle_speed, radius
#     )
#
#     # Plot the result
#     matplotlib.use('Qt5Agg')
#     plot_dynamic_trajectories(aircraft_trajectories, obstacle_trajectory, 20, x_lim, y_lim)



#________________________________FMT dynamic obstacle, consider other ACs as obstacles if distance < 20__________________good version

import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial import distance
import time
import matplotlib

# Function to check if a point is inside an obstacle
def is_in_obstacle(point, obstacle_pos, size):
    ox, oy = obstacle_pos
    return (point[0] - ox) ** 2 + (point[1] - oy) ** 2 <= size ** 2


# Function to check if a point is inside a nearby aircraft (treated as a dynamic obstacle)
def is_nearby_aircraft(point, aircraft_pos, avoidance_radius=10):
    return np.linalg.norm(np.array(point) - np.array(aircraft_pos)) < avoidance_radius


# Function to find the nearest neighbor nodes to a node
def nearest_neighbors(nodes, target_node, radius):
    distances = distance.cdist(nodes, [target_node])
    return np.where(distances <= radius)[0]


# Function to generate random nodes, avoiding obstacles and nearby aircraft
def generate_random_nodes(num_nodes, x_lim, y_lim, obstacle_pos, obstacle_size, other_aircraft_positions):
    nodes = []
    while len(nodes) < num_nodes:
        x = random.uniform(x_lim[0], x_lim[1])
        y = random.uniform(x_lim[0], x_lim[1])
        if not is_in_obstacle([x, y], obstacle_pos, obstacle_size) and \
                not any(is_nearby_aircraft([x, y], ac_pos) for ac_pos in other_aircraft_positions):
            nodes.append([x, y])
    return np.array(nodes)


# Function to check if there is a collision-free path between two points
def is_collision_free(node1, node2, obstacle_pos, obstacle_size, other_aircraft_positions, step_size=0.05):
    steps = int(np.linalg.norm(np.array(node2) - np.array(node1)) / step_size)
    for i in range(steps):
        interp = node1 + (i / steps) * (np.array(node2) - np.array(node1))
        if is_in_obstacle(interp, obstacle_pos, obstacle_size) or \
                any(is_nearby_aircraft(interp, ac_pos) for ac_pos in other_aircraft_positions):
            return False
    return True


# Simulate obstacle moving 1 unit per timestep
def dynamic_obstacle_movement(start_pos, end_pos, t, obstacle_speed=1):
    total_distance = np.linalg.norm(end_pos - start_pos)
    direction = (end_pos - start_pos) / total_distance
    step_distance = obstacle_speed  # Move 1 unit per timestep
    return start_pos + direction * step_distance * t


# Replan the path at each timestep, considering the dynamic obstacle and other aircraft as obstacles
def plan_path(current_pos, goal, num_nodes, x_lim, y_lim, obstacle_pos, obstacle_size, other_aircraft_positions,
              radius):
    nodes = generate_random_nodes(num_nodes, x_lim, y_lim, obstacle_pos, obstacle_size, other_aircraft_positions)
    nodes = np.vstack([current_pos, nodes, goal])  # Include current and goal in the set of nodes

    tree = {0: None}  # node_index: parent_node_index
    open_set = {0}  # Set of nodes to be explored
    closed_set = set()  # Set of explored nodes

    while open_set:
        current_node = min(open_set, key=lambda node: np.linalg.norm(nodes[node] - goal))
        open_set.remove(current_node)
        closed_set.add(current_node)

        if np.linalg.norm(nodes[current_node] - goal) <= radius:
            tree[len(nodes) - 1] = current_node
            break

        neighbors = nearest_neighbors(nodes, nodes[current_node], radius)
        for neighbor in neighbors:
            if neighbor not in closed_set and is_collision_free(nodes[current_node], nodes[neighbor], obstacle_pos,
                                                                obstacle_size, other_aircraft_positions,
                                                                step_size=0.05):
                if neighbor not in tree:
                    tree[neighbor] = current_node
                    open_set.add(neighbor)

    # Extract the path from the tree
    path = []
    node = len(nodes) - 1  # Goal node
    while node is not None:
        path.append(nodes[node])
        node = tree.get(node)

    return path[::-1]


# Dynamic FMT* algorithm implementation for multiple aircraft with delayed entry and collision avoidance
def dynamic_flight_path_multiple(aircraft_dict_initial, aircraft_dict_late, num_nodes, x_lim, y_lim, obstacle_start,
                                 obstacle_end, t_max, obstacle_speed=1, radius=10, late_timestep=10):
    all_flown_trajectories = {}  # Dictionary to store flown trajectories for each aircraft
    obstacle_trajectory = []  # List to store the obstacle's movement

    # Initialize positions for the initial aircraft
    current_positions = {name: OD[0] for name, OD in aircraft_dict_initial.items()}
    flown_trajectories = {name: [OD[0]] for name, OD in aircraft_dict_initial.items()}

    t = 0  # Timestep counter

    while t < t_max:
        t += 1

        # Move the obstacle by 1 unit per timestep
        obstacle_pos = dynamic_obstacle_movement(obstacle_start, obstacle_end, t, obstacle_speed)
        obstacle_trajectory.append(obstacle_pos)

        # Add the second group of aircraft at the specified timestep (e.g., timestep 10)
        if t == late_timestep:
            current_positions.update({name: OD[0] for name, OD in aircraft_dict_late.items()})
            flown_trajectories.update({name: [OD[0]] for name, OD in aircraft_dict_late.items()})

        # Track progress to ensure we don't have an endless loop
        has_progress = False  # Track whether any aircraft made progress this timestep

        # Loop through each aircraft
        for name, (start, goal) in {**aircraft_dict_initial, **aircraft_dict_late}.items():
            if name not in current_positions:  # If aircraft has been removed after reaching its goal
                continue

            current_pos = current_positions[name]
            if np.linalg.norm(current_pos - goal) > 0:  # Check if the aircraft has reached the goal

                # List of positions of other aircraft (excluding the current aircraft)
                other_aircraft_positions = [pos for other_name, pos in current_positions.items() if other_name != name]

                # Replan the path based on the updated obstacle position and nearby aircraft
                current_path = plan_path(current_pos, goal, num_nodes, x_lim, y_lim, obstacle_pos, obstacle_size=20,
                                         other_aircraft_positions=other_aircraft_positions, radius=radius)

                # Aircraft moves to the next node in the path
                if len(current_path) > 1:
                    next_node = current_path[1]
                    if not np.array_equal(current_pos, next_node):  # Ensure there's actual progress
                        current_positions[name] = next_node
                        flown_trajectories[name].append(next_node)
                        has_progress = True

            # If the aircraft has reached its goal, remove it from current_positions
            if np.linalg.norm(current_positions[name] - goal) <= 1e-3:
                print(f"Aircraft {name} has reached its goal and is removed from the simulation.")
                current_positions.pop(name)

        # If no aircraft made progress and all have not reached their goals, break to prevent infinite loop
        if not has_progress and t >= t_max / 2:  # Add a safeguard to force stopping after half max timesteps
            print("No progress detected for several timesteps. Breaking to avoid infinite loop.")
            break

        # Break the loop when all aircraft have reached their goals
        if all(np.linalg.norm(current_positions[name] - goal) <= 1e-3 for name in current_positions):
            print(f"All aircraft reached their goals by timestep {t}.")
            break

    return flown_trajectories, obstacle_trajectory



# Visualization of the dynamic trajectories for multiple aircraft
def plot_dynamic_trajectories(aircraft_trajectories, obstacle_trajectory, obstacle_size, x_lim, y_lim):
    plt.figure()

    # Plot flown trajectories for each aircraft
    for name, trajectory in aircraft_trajectories.items():
        trajectory = np.array(trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 1], linestyle=':', label=f'{name} trajectory')
        plt.scatter(trajectory[0, 0], trajectory[0, 1], marker='^', color='green', s=50)  # Start marker
        plt.scatter(trajectory[-1, 0], trajectory[-1, 1], marker='*', color='red', s=50)  # Goal marker

    # Add text labels for the starting positions of AR1, AR2, AR3
    plt.text(20, 180, 'AR1', fontsize=10, ha='right')
    plt.text(20, 102, 'AR2', fontsize=10, ha='right')
    plt.text(20, 30, 'AR3', fontsize=10, ha='right')

    # Plot direct lines for each AR with width of 10 units
    plt.plot([20, 170], [180, 20], linestyle='-', color='purple', linewidth=10, alpha=0.1)  # AR1
    plt.plot([20, 185], [102, 102], linestyle='-', color='green', linewidth=10, alpha=0.1)  # AR2
    plt.plot([20, 170], [30, 180], linestyle='-', color='red', linewidth=10, alpha=0.1)     # AR3


    # Plot obstacle's trajectory
    obstacle_trajectory = np.array(obstacle_trajectory)
    # plt.plot(obstacle_trajectory[:, 0], obstacle_trajectory[:, 1], 'r--', label='Obstacle Trajectory')
    plt.plot(obstacle_trajectory[:, 0], obstacle_trajectory[:, 1], 'r--')

    # Plot obstacle at each timestep
    for i, obs in enumerate(obstacle_trajectory):
        if i % 10 == 0:  # plot circle every 5 timesteps
            circle = plt.Circle(obs, obstacle_size, color='red', fill=True, alpha=0.3)
            plt.gca().add_patch(circle)

    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel('Length (nm)')
    plt.ylabel('Width (nm)')
    # plt.title('Dynamic Obstacle and Multiple Aircraft Trajectories with Collision Avoidance')
    # plt.legend()
    plt.show()


# Example usage for multiple aircraft with delayed entry
if __name__ == "__main__":
    # Parameters
    num_nodes = 2000  # Sampled nodes from start to end
    radius = 10  # Connection threshold of nearby node
    x_lim = [0, 200]
    y_lim = [0, 200]

    # Define initial aircraft with OD locations (appearing at timestep 1)
    aircraft_dict_initial = {
        'AC1': (np.array([20, 180]), np.array([170, 20])),
        'AC2': (np.array([20, 102]), np.array([185, 102])),
        'AC3': (np.array([20, 30]), np.array([170, 180])),
    }

    # Define later aircraft with OD locations (appearing at timestep 10)
    aircraft_dict_late = {
        'AC4': (np.array([20, 180]), np.array([170, 20])),
        'AC5': (np.array([20, 102]), np.array([185, 102])),
        'AC6': (np.array([20, 30]), np.array([170, 180])),
    }

    # Obstacle moves from one point to another within boundary
    obstacle_start = np.array([135, 160])
    obstacle_end = np.array([80, 25])

    t_max = 60  # Manually adjust the number of time steps
    late_timestep = 10  # Time step at which late aircraft appear
    obstacle_speed = 2.5


    start_time = time.time()  # Start timing
    # Run the dynamic FMT* algorithm for multiple aircraft with delayed entry and collision avoidance
    aircraft_trajectories, obstacle_trajectory = dynamic_flight_path_multiple(
        aircraft_dict_initial, aircraft_dict_late, num_nodes, x_lim, y_lim, obstacle_start, obstacle_end, t_max,
        obstacle_speed, radius, late_timestep=late_timestep
    )

    end_time = time.time()  # End timing
    print(f"Total simulation time: {end_time - start_time:.2f} seconds")
    # Plot the result
    matplotlib.use('Qt5Agg')
    plot_dynamic_trajectories(aircraft_trajectories, obstacle_trajectory, 30, x_lim, y_lim)


