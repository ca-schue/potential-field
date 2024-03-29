import numpy as np
import matplotlib.pyplot as plt
import warnings

def compute_potential_attraction_repulsion(configuration_space, goal, attraction_weight=3, repulsion_weight=2):

    # Erweiterung um eine Zeile oben und unten für jede Ebene (rotations bleibt unverändert)
    configuration_space_padded = np.pad(configuration_space, ((0, 0), (1, 1), (0, 0)), mode='constant', constant_values=False)

    # Erweiterung um eine Spalte links und rechts für jede Ebene
    configuration_space_padded = np.pad(configuration_space_padded, ((0, 0), (0, 0), (1, 1)), mode='constant', constant_values=False)

    size_rotation, size_y, size_x = configuration_space_padded.shape
    attraction_potential = np.zeros((size_rotation, size_y, size_x))
    repulsion_potential = np.zeros((size_rotation, size_y, size_x))

    for rotation in range(size_rotation):
        for y in range(size_y):
            for x in range(size_x):
                # Abstand zum Ziel
                distance_to_goal = np.sqrt((y - goal[1])**2 + (x - goal[0])**2)

                # Anziehendes Potential zum Ziel (Quadratische Funktion)
                attraction_potential[rotation, y, x] = distance_to_goal**2

                repulsive_potentials_to_all_objects = []
                # Abstand zu Hindernissen in der aktuellen Rotationsebene
                distances_to_obstacles = np.sqrt((y - np.where(configuration_space_padded[rotation] == False)[0])**2 +
                                                 (x - np.where(configuration_space_padded[rotation] == False)[1])**2)
                
                for distance_to_obstacle in distances_to_obstacles:
                    # Abstoßendes Potential von Hindernissen in der aktuellen Rotationsebene (Quadratische Funktion)
                    repulsive_potentials_to_all_objects.append(1 / (repulsion_weight + distance_to_obstacle) if distance_to_obstacle != 0 else 0)

                # Abstand zu Hindernissen in den benachbarten Rotationsebenen
                adjacent_rotations = [(rotation + 1) % size_rotation, (rotation - 1) % size_rotation]
                distances_to_obstacles_adjacent_grids = [(np.sqrt(
                                                        (y - np.where(configuration_space_padded[adj_rotation] == False)[0])**2 +
                                                        (x - np.where(configuration_space_padded[adj_rotation] == False)[1])**2) + (rotation - adj_rotation)**2)
                                                    for adj_rotation in adjacent_rotations]
 
                for distances_to_obstacles_adjacent_grid in distances_to_obstacles_adjacent_grids:
                    for distance_to_obstacles_adjacent in distances_to_obstacles_adjacent_grid:
                        repulsive_potentials_to_all_objects.append(1 / (repulsion_weight + distance_to_obstacles_adjacent) if distance_to_obstacles_adjacent != 0 else 0)
                                
                repulsion_potential[rotation, y, x] = np.max(repulsive_potentials_to_all_objects)
                

    weighted_normalized_attraction_potential = (attraction_potential / np.max(attraction_potential)) * attraction_weight
    weighted_normalized_attraction_potential[~configuration_space_padded] = np.nan
    weighted_normalized_attraction_potential = weighted_normalized_attraction_potential[:, 1:-1, :]
    weighted_normalized_attraction_potential = weighted_normalized_attraction_potential[:, :, 1:-1]

    repulsion_potential[~configuration_space_padded] = np.nan
    repulsion_potential = repulsion_potential[:, 1:-1, :]
    repulsion_potential = repulsion_potential[:, :, 1:-1]


    #total_potential = normalized_attraction_potential + repulsion_potential
    #total_potential[~configuration_space_padded] = np.nan
    #total_potential = total_potential[:, 1:-1, :]
    #total_potential = total_potential[:, :, 1:-1]
    #total_potential[goal[2],goal[1],goal[0]] = 0

    return weighted_normalized_attraction_potential, repulsion_potential


def compute_potential_wavefront(configuration_space, goal):
    rotation_size, size_y, size_x = configuration_space.shape
    goal_x, goal_y, goal_rotation = goal

    potential = np.zeros_like(configuration_space, dtype=float)  # initialize with 0s
    potential[~configuration_space] = np.nan # potential obstacles is undefined

    potential_0 = None

    potential_650 = None

    potential_1250 = None

    queue = [(goal_x, goal_y, goal_rotation, 2)]  # start with the goal
    visited = set([(goal_x, goal_y, goal_rotation)])

    index = 0

    while queue:

        if index == 0:
            potential_0 = np.copy(potential)
        if index == 650:
            potential_650 = np.copy(potential)
        if index == 1250:
            potential_1250 = np.copy(potential)

        index += 1

        current_x, current_y, current_rotation, current_potential = queue.pop(0)

        potential[current_rotation, current_y, current_x] = current_potential      

        # Check neighbours in xy-boundary + not an obstacle + not visited
        for dr, dy, dx in [(0, -1, 0), (0, 1, 0), (0, 0, 1), (0, 0, -1), (-1, 0, 0), (1, 0, 0)]:
            next_rotation = (current_rotation + dr) % rotation_size
            next_y, next_x = current_y + dy, current_x + dx

            if 0 <= next_y < size_y and 0 <= next_x < size_x and \
               potential[next_rotation, next_y, next_x] is not np.nan and \
               potential[next_rotation, next_y, next_x] == 0 and \
               (next_x, next_y, next_rotation) not in visited:
                queue.append((next_x, next_y, next_rotation, current_potential + 1))
                visited.add((next_x, next_y, next_rotation))
    
    unreachable_points = np.argwhere(potential == 0) # wavefront algorithm is convergent!
    if len(unreachable_points) > 0:
        warnings.warn("Unreachable points exist, setting as obstacles: " + str(unreachable_points))
        for unreachable_rotation, unreachable_y, unreachable_x in unreachable_points:
            potential[unreachable_rotation, unreachable_y, unreachable_x] = np.nan
            configuration_space[unreachable_rotation, unreachable_y, unreachable_x] = False
    
    return potential_0, potential_650, potential_1250, potential


def plot_potential_stacked(potential, title, ax, rotation_step, alpha=1):

    potential_plot = np.nan_to_num(potential)

    potential_plot = potential_plot/np.max(potential_plot)
    rotation, size_y, size_x = potential_plot.shape

    # Erzeuge ein Gitter im 3D-Raum
    x = np.arange(size_x)
    y = np.arange(size_y)
    X, Y = np.meshgrid(x, y)

    # Plotte die 3D-imshow für jede Rotationsebene
    for r in range(rotation):
        Z = np.full_like(X, r)  # Z-Koordinate auf Höhe der Rotationsebene
        colors = plt.cm.plasma(potential_plot[r, :, :])
        colors[np.isnan(potential[r, :, :])] = (1,1,1,1)  # Graue Farbe für np.nan
        ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1, alpha=alpha, antialiased=True)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Rotation')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        angles = np.arange(0, 360, rotation_step)    
        ax.set_zticklabels([item for sublist in zip([f'{angle}°' for angle in angles], [''] * len(angles)) for item in sublist])

    ax.set_title(title)
    ax.invert_yaxis()


from matplotlib.colors import Normalize

def plot_potential_slice(potential_slice, title, ax, alpha=1, normalize=True, max_value=None):

    if max_value is None:
        max_value = np.max(np.nan_to_num(potential_slice))

    potential_slice_plot = np.copy(potential_slice)

    cmap = plt.cm.plasma

    if normalize:
        #obstacles = np.isnan(potential_slice)
        #max_pot = np.max(np.nan_to_num(potential_slice))
        #potential_slice_plot[obstacles] = max_pot

        potential_slice_plot = potential_slice_plot / np.max(np.nan_to_num(potential_slice_plot))

        Z = np.ma.masked_where(np.isnan(potential_slice), potential_slice_plot)
        colors = plt.cm.plasma(Z)
    else:
        norm = Normalize(vmin=0, vmax=max_value)  # Benutzerdefinierte Normalisierung
        Z = np.ma.masked_where(np.isnan(potential_slice), potential_slice_plot)
        colors = cmap(norm(Z))

    size_y, size_x = potential_slice_plot.shape

    # Erzeuge ein Gitter im 3D-Raum
    x = np.arange(size_x)
    y = np.arange(size_y)
    X, Y = np.meshgrid(x, y)

    # Erstelle eine benutzerdefinierte Farbkarte    
    ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1, alpha=alpha, antialiased=True)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Potential')
    ax.set_title(title)
    ax.invert_yaxis()