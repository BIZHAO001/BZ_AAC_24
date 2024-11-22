#_________________________________________________________evaluations__________________________________________________
import csv
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial import distance
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.colors import LinearSegmentedColormap
import time
import matplotlib
import pickle
import os
from scipy.ndimage import gaussian_filter

# Function to check if a point is inside an obstacle
def is_in_obstacle(point, obstacle_pos, size):
    ox, oy = obstacle_pos
    return (point[0] - ox) ** 2 + (point[1] - oy) ** 2 <= size ** 2

# Function to check if a point is inside a nearby aircraft (treated as a dynamic obstacle)
def is_nearby_aircraft(point, aircraft_pos, avoidance_radius=30):
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

# Saving the simulation results to pickle
def save_simulation_results(aircraft_trajectories, obstacle_trajectory):
    current_time = time.strftime("%Y%m%d-%H_%M_%S")
    simu_results = {
        'aircraft_trajectories': aircraft_trajectories,
        'obstacle_trajectory': obstacle_trajectory,
    }
    directory = r'D:/FMT_vs_IDDPG/data'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f'simu_results_{current_time}.pkl'
    file_path = os.path.join(directory, filename)
    with open(file_path, 'wb') as file:
        pickle.dump(simu_results, file)

# Replan the path at each timestep, considering the dynamic obstacle and other aircraft as obstacles
def plan_path(current_pos, goal, num_nodes, x_lim, y_lim, obstacle_pos, obstacle_size, other_aircraft_positions, radius):
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
            if neighbor not in closed_set and is_collision_free(nodes[current_node], nodes[neighbor], obstacle_pos, obstacle_size, other_aircraft_positions, step_size=0.05):
                if neighbor not in tree:
                    tree[neighbor] = current_node
                    open_set.add(neighbor)
    path = []
    node = len(nodes) - 1  # Goal node
    while node is not None:
        path.append(nodes[node])
        node = tree.get(node)
    return path[::-1]

# Dynamic FMT* algorithm for multiple aircraft with delayed entry and multiple groups
def dynamic_flight_path_multiple(aircraft_dict_initial, aircraft_dict_late1, aircraft_dict_late2, num_nodes, x_lim, y_lim, obstacle_start, obstacle_end, t_max, obstacle_speed=1, radius=10, late_timestep1=10, late_timestep2=20):
    all_flown_trajectories = {}  # Dictionary to store flown trajectories for each aircraft
    obstacle_trajectory = []  # List to store the obstacle's movement
    current_positions = {name: OD[0] for name, OD in aircraft_dict_initial.items()}
    flown_trajectories = {name: [OD[0]] for name, OD in aircraft_dict_initial.items()}
    t = 0  # Timestep counter
    while t < t_max:
        t += 1
        obstacle_pos = dynamic_obstacle_movement(obstacle_start, obstacle_end, t, obstacle_speed)
        obstacle_trajectory.append(obstacle_pos)
        if t == late_timestep1:
            current_positions.update({name: OD[0] for name, OD in aircraft_dict_late1.items()})
            flown_trajectories.update({name: [OD[0]] for name, OD in aircraft_dict_late1.items()})
        if t == late_timestep2:
            current_positions.update({name: OD[0] for name, OD in aircraft_dict_late2.items()})
            flown_trajectories.update({name: [OD[0]] for name, OD in aircraft_dict_late2.items()})
        has_progress = False  # Track whether any aircraft made progress this timestep
        for name, (start, goal) in {**aircraft_dict_initial, **aircraft_dict_late1, **aircraft_dict_late2}.items():
            if name not in current_positions:  # If aircraft has been removed after reaching its goal
                continue
            current_pos = current_positions[name]
            if np.linalg.norm(current_pos - goal) > 0:  # Check if the aircraft has reached the goal
                other_aircraft_positions = [pos for other_name, pos in current_positions.items() if other_name != name]
                current_path = plan_path(current_pos, goal, num_nodes, x_lim, y_lim, obstacle_pos, obstacle_size=20, other_aircraft_positions=other_aircraft_positions, radius=radius)
                if len(current_path) > 1:
                    next_node = current_path[1]
                    if not np.array_equal(current_pos, next_node):  # Ensure there's actual progress
                        current_positions[name] = next_node
                        flown_trajectories[name].append(next_node)
                        has_progress = True
            if np.linalg.norm(current_positions[name] - goal) <= 1e-3:
                print(f"Aircraft {name} has reached its goal and is removed from the simulation.")
                current_positions.pop(name)
        if not has_progress and t >= t_max / 2:
            print("No progress detected for several timesteps. Breaking to avoid infinite loop.")
            break
        if all(np.linalg.norm(current_positions[name] - goal) <= 1e-3 for name in current_positions):
            print(f"All aircraft reached their goals by timestep {t}.")
            break
    return flown_trajectories, obstacle_trajectory

# Visualization of the dynamic trajectories for multiple aircraft
def plot_dynamic_trajectories(aircraft_trajectories, obstacle_trajectory, obstacle_size, x_lim, y_lim):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    matplotlib.use('TkAgg')
    fig, ax = plt.subplots(figsize=(8, 8))  # Set the figure size to be square
    alpha_values = np.linspace(0.1, 1.0, len(obstacle_trajectory))
    for name, trajectory in aircraft_trajectories.items():
        trajectory = np.array(trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 1], linestyle=':', label=f'{name} trajectory')
        plt.scatter(trajectory[0, 0], trajectory[0, 1], marker='^', color='green', s=50)  # Start marker
        plt.scatter(trajectory[-1, 0], trajectory[-1, 1], marker='*', color='red', s=50)  # Goal marker

    plt.text(20, 180, 'AR1', fontsize=12, ha='right')
    plt.text(20, 102, 'AR2', fontsize=12, ha='right')
    plt.text(20, 30, 'AR3', fontsize=12, ha='right')

    plt.plot([20, 170], [180, 20], linestyle='-', color='purple', linewidth=10, alpha=0.1)  # AR1
    plt.plot([20, 185], [102, 102], linestyle='-', color='green', linewidth=10, alpha=0.1)  # AR2
    plt.plot([20, 170], [30, 180], linestyle='-', color='red', linewidth=10, alpha=0.1)  # AR3

    obstacle_trajectory = np.array(obstacle_trajectory)
    # plt.plot(obstacle_trajectory[:, 0], obstacle_trajectory[:, 1], 'r--')
    for i, obs in enumerate(obstacle_trajectory):
        if i % 5 == 0:  # plot circle every 10 timesteps
            # circle = plt.Circle(obs, obstacle_size, color='red', fill=True, alpha=0.3)
            # plt.gca().add_patch(circle)

            center_x, center_y = obs[0], obs[1]

            # Generate multiple clusters of random points within the specified range
            num_points_per_cluster = 5000
            num_clusters = 15
            contour_range = 20  # 10, 20, 30

            x_range = (-contour_range/1.5 + 2, contour_range/1.5 + 2)
            y_range = (-contour_range/1.5 + 2, contour_range/1.5 + 2)

            cluster_centers_x = np.random.uniform(center_x + x_range[0], center_x + x_range[1], num_clusters)
            cluster_centers_y = np.random.uniform(center_y + y_range[0], center_y + y_range[1], num_clusters)
            cluster_centers = np.column_stack((cluster_centers_x, cluster_centers_y))

            # Generate points for each cluster with controlled density
            x, y = [], []
            for cx, cy in cluster_centers:
                angles = np.random.uniform(0, 2 * np.pi, num_points_per_cluster)
                radii = np.random.normal(0, 0.1, num_points_per_cluster)  # Decrease spread for higher density
                x.extend(cx + radii * np.cos(angles))
                y.extend(cy + radii * np.sin(angles))
            x = np.array(x)
            y = np.array(y)
            # Create a 2D histogram to serve as the contour data
            margin = 25
            contour_min_x = center_x + x_range[0] - margin
            contour_max_x = center_x + x_range[1] + margin
            contour_min_y = center_y + y_range[0] - margin
            contour_max_y = center_y + y_range[1] + margin
            hist, xedges, yedges = np.histogram2d(x, y, bins=(100, 100),
                                                  range=[[contour_min_x, contour_max_x],
                                                         [contour_min_y, contour_max_y]])
            # Smooth the histogram to create a more organic shape
            hist = gaussian_filter(hist, sigma=5)  # Adjust sigma for better control

            # Create the custom colormap from green to yellow to red
            cmap = LinearSegmentedColormap.from_list('green_yellow_red', ['green', 'yellow', 'red'])
            X, Y = np.meshgrid(xedges[:-1] + 0.5 * (xedges[1] - xedges[0]), yedges[:-1] + 0.5 * (yedges[1] - yedges[0]))
            contour_levels = np.linspace(hist.min(), hist.max(), 10)

            # if i == len(trajectory_eachPlay):
            #     contour = ax.contourf(X, Y, hist, levels=contour_levels, cmap=cmap, alpha=alpha_values[trajectory_idx])
            # else:
            #     # in this loop we only required to draw once.
            #     if contour_drawn[cloud_idx] == False:
            #         contour = ax.contourf(X, Y, hist, levels=contour_levels, cmap=cmap)
            #         contour_drawn[cloud_idx] = True
            contour = ax.contourf(X, Y, hist, levels=contour_levels, cmap=cmap, alpha=alpha_values[i])
            level_color = cmap(
                (contour_levels[1] - contour_levels.min()) / (contour_levels.max() - contour_levels.min()))
            # Extract the outermost contour path and overlay it with a black line

            # if max_time_step == len(trajectory_eachPlay):
            #     outermost_contour = ax.contour(X, Y, hist, levels=[contour_levels[1]], colors=[level_color],
            #                                    linewidths=1, alpha=alpha_values[
            #             trajectory_idx])  # this line must be present to show the boundary fading
            # else:
            #     # in this loop we only required to draw once.
            #     if outline_drawn[cloud_idx] == False:
            #         outermost_contour = ax.contour(X, Y, hist, levels=[contour_levels[1]], colors=[level_color],
            #                                        linewidths=1)
            #         outline_drawn[cloud_idx] = True
            outermost_contour = ax.contour(X, Y, hist, levels=[contour_levels[1]], colors=[level_color],
                                           linewidths=1, alpha=alpha_values[i])
            # Extract the vertices of the outermost contour path
            outermost_path = outermost_contour.collections[0].get_paths()[0]
            vertices = outermost_path.vertices
            x_clip, y_clip = vertices[:, 0], vertices[:, 1]
            # ax.plot(x_clip, y_clip, color="crimson")
            coordinates = np.column_stack((x_clip, y_clip))
            clippath = Path(coordinates)
            patch = PathPatch(clippath, facecolor='none', alpha=alpha_values[i])
            ax.add_patch(patch)
            for c in contour.collections:
                c.set_clip_path(patch)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.gca().set_aspect('equal', adjustable='box')  # Ensure the plot has equal aspect ratio for x and y axes
    plt.xlabel('Length (NM)', fontsize=12)
    plt.ylabel('Width (NM)', fontsize=12)
    # Set tick font sizes
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

def save_figure_with_new_folder(simulation_number, folder_path):
    # Define the path for saving the figure in SVG format
    figure_path = os.path.join(folder_path, f"simu{simulation_number}.svg")  # Save as SVG
    # Save the figure
    plt.savefig(figure_path, format='svg')  # Save as SVG format


def create_simulation_folder():
    base_directory = r'D:/FMT_vs_IDDPG/figure'
    current_time = time.strftime("%Y%m%d-%H_%M_%S")
    new_folder_path = os.path.join(base_directory, current_time)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    return new_folder_path

def run_multiple_simulations(n, num_nodes, radius, x_lim, y_lim, aircraft_dict_initial, aircraft_dict_late1, aircraft_dict_late2, obstacle_start, obstacle_end, t_max, late_timestep1, late_timestep2, obstacle_speed, obstacle_size):
    simulation_folder = create_simulation_folder()
    results_directory = r'D:/FMT_vs_IDDPG/results'
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    current_time = time.strftime("%Y%m%d-%H_%M_%S")
    csv_file_path = os.path.join(results_directory, f"performance_metrics_{current_time}.csv")
    fieldnames = ['Simulation', 'Loss of Separation Aircraft Rate', 'Loss of Separation Obstacle Rate', 'Goal Reach Rate', 'Average Distance Ratio', 'Average Computation Time']
    with open(csv_file_path, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for sim in range(1, n + 1):
            print(f"Running Simulation {sim}/{n}...")
            start_time = time.time()
            aircraft_trajectories, obstacle_trajectory = dynamic_flight_path_multiple(
                aircraft_dict_initial, aircraft_dict_late1, aircraft_dict_late2, num_nodes, x_lim, y_lim, obstacle_start, obstacle_end, t_max, obstacle_speed, radius, late_timestep1=late_timestep1, late_timestep2=late_timestep2
            )
            end_time = time.time()
            computation_times = [end_time - start_time]
            od_dict = {**aircraft_dict_initial, **aircraft_dict_late1, **aircraft_dict_late2}
            performance_results = evaluate_performance(aircraft_trajectories, obstacle_trajectory, obstacle_size, od_dict, computation_times)
            performance_results['Simulation'] = sim
            writer.writerow({
                'Simulation': sim,
                'Loss of Separation Aircraft Rate': performance_results['loss_of_separation_aircraft_rate'],
                'Loss of Separation Obstacle Rate': performance_results['loss_of_separation_obstacle_rate'],
                'Goal Reach Rate': performance_results['goal_reach_rate'],
                'Average Distance Ratio': performance_results['average_distance_ratio'],
                'Average Computation Time': performance_results['average_computation_time']
            })
            plot_dynamic_trajectories(aircraft_trajectories, obstacle_trajectory, obstacle_size, x_lim, y_lim)
            save_figure_with_new_folder(simulation_number=sim, folder_path=simulation_folder)
    print(f"All {n} simulations completed. Performance metrics saved to {csv_file_path}.")

def evaluate_performance(aircraft_trajectories, obstacle_trajectory, obstacle_size, od_dict, computation_times):
    total_aircraft = len(aircraft_trajectories)
    loss_of_separation_aircraft = 0
    loss_of_separation_obstacle = 0
    reached_aircraft = 0
    total_distance_ratio = 0
    # Iterate over timesteps, checking for loss of separation between aircraft
    for t in range(max(len(traj) for traj in aircraft_trajectories.values())):
        aircraft_positions_at_t = []

        # Collect positions of all aircraft at time t
        for traj in aircraft_trajectories.values():
            if t < len(traj):  # Make sure the trajectory has a position at this time step
                aircraft_positions_at_t.append(traj[t])

        # Check pairwise distances between aircraft at time t
        for i in range(len(aircraft_positions_at_t)):
            for j in range(i + 1, len(aircraft_positions_at_t)):  # Avoid checking the same pair twice
                dist = np.linalg.norm(np.array(aircraft_positions_at_t[i]) - np.array(aircraft_positions_at_t[j]))
                if dist < 10:  # Check for loss of separation condition (distance < 10 nm)
                    loss_of_separation_aircraft += 1

    for t in range(len(obstacle_trajectory)):
        obstacle_pos = obstacle_trajectory[t]
        for traj in aircraft_trajectories.values():
            if t < len(traj):
                aircraft_pos = traj[t]
                dist = np.linalg.norm(np.array(aircraft_pos) - np.array(obstacle_pos))
                if dist < 10:
                    loss_of_separation_obstacle += 1
    for ac, (start, goal) in od_dict.items():
        final_position = aircraft_trajectories[ac][-1]
        if np.linalg.norm(np.array(final_position) - np.array(goal)) <= 1e-3:
            reached_aircraft += 1
    goal_reach_rate = reached_aircraft / total_aircraft
    for ac, (start, goal) in od_dict.items():
        shortest_distance = np.linalg.norm(np.array(goal) - np.array(start))
        actual_flight_distance = 0
        for i in range(1, len(aircraft_trajectories[ac])):
            actual_flight_distance += np.linalg.norm(np.array(aircraft_trajectories[ac][i]) - np.array(aircraft_trajectories[ac][i - 1]))
        distance_ratio = actual_flight_distance / shortest_distance if shortest_distance != 0 else 0
        total_distance_ratio += distance_ratio
    avg_distance_ratio = total_distance_ratio / total_aircraft
    total_used_timesteps = max(len(traj) for traj in aircraft_trajectories.values())
    total_computation_time = sum(computation_times)
    avg_computation_time = total_computation_time / total_used_timesteps if total_used_timesteps != 0 else 0
    results = {
        'loss_of_separation_aircraft_rate': loss_of_separation_aircraft / (total_aircraft * max(len(traj) for traj in aircraft_trajectories.values())),
        'loss_of_separation_obstacle_rate': loss_of_separation_obstacle / (total_aircraft * len(obstacle_trajectory)),
        'goal_reach_rate': goal_reach_rate,
        'average_distance_ratio': avg_distance_ratio,
        'average_computation_time': avg_computation_time
    }
    return results

# Main Execution
if __name__ == "__main__":
    num_nodes = 2000
    radius = 10
    x_lim = [0, 200]
    y_lim = [0, 200]
    aircraft_dict_initial = {
        'AC1': (np.array([20, 180]), np.array([170, 20])),
        'AC2': (np.array([20, 102]), np.array([185, 102])),
        'AC3': (np.array([20, 30]), np.array([170, 180])),
    }
    aircraft_dict_late1 = {
        'AC4': (np.array([20, 180]), np.array([170, 20])),
        'AC5': (np.array([20, 102]), np.array([185, 102])),
        'AC6': (np.array([20, 30]), np.array([170, 180])),
    }
    aircraft_dict_late2 = {
        'AC7': (np.array([20, 180]), np.array([170, 20])),
        'AC8': (np.array([20, 102]), np.array([185, 102])),
    }
    obstacle_start = np.array([135, 160])
    obstacle_end = np.array([80, 25])
    t_max = 100
    late_timestep1 = 10
    late_timestep2 = 20
    obstacle_speed = 2.5
    obstacle_size = 25

    run_multiple_simulations(
        n=3,
        num_nodes=num_nodes,
        radius=radius,
        x_lim=x_lim,
        y_lim=y_lim,
        aircraft_dict_initial=aircraft_dict_initial,
        aircraft_dict_late1=aircraft_dict_late1,
        aircraft_dict_late2=aircraft_dict_late2,
        obstacle_start=obstacle_start,
        obstacle_end=obstacle_end,
        t_max=t_max,
        late_timestep1=late_timestep1,
        late_timestep2=late_timestep2,
        obstacle_speed=obstacle_speed,
        obstacle_size=obstacle_size,
    )



    # Run single dynamic FMT* algorithm for multiple aircraft with delayed entry and collision avoidance
    # start_time = time.time()  # Start timing
    # aircraft_trajectories, obstacle_trajectory = dynamic_flight_path_multiple(
    #     aircraft_dict_initial, aircraft_dict_late, num_nodes, x_lim, y_lim, obstacle_start, obstacle_end, t_max,
    #     obstacle_speed, radius, late_timestep=late_timestep
    # )
    # end_time = time.time()  # End timing
    #
    # computation_times = [end_time - start_time]  # Example: Can be updated per timestep in the main loop
    #
    # # Display total simulation time
    # print(f"Total simulation time: {sum(computation_times):.2f} seconds")
    #
    # # Define the OD dict to evaluate performance
    # od_dict = {**aircraft_dict_initial, **aircraft_dict_late}
    #
    # # Evaluations
    # performance_results = evaluate_performance(aircraft_trajectories, obstacle_trajectory, obstacle_size, od_dict,
    #                                            computation_times)
    # print(f"Performance Results: {performance_results}")
    #
    # # Plot the result
    # matplotlib.use('Qt5Agg')
    # plot_dynamic_trajectories(aircraft_trajectories, obstacle_trajectory, obstacle_size, x_lim, y_lim)
    # save_figure_with_new_folder() # save figures
    #
    # # Save the results
    # save_simulation_results(aircraft_trajectories, obstacle_trajectory)
