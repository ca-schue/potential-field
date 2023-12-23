import warnings
import numpy as np

def compute_gradients(potential, goal_point):

    obstacles_mask = np.isnan(potential)
    
    # Kopiere die erste und letzte Ebene
    first_layer = potential[0, :, :]
    last_layer = potential[-1, :, :]

    # Füge die kopierten Ebenen oben und unten an
    potential_z_padded = np.concatenate(
        [last_layer[np.newaxis, :, :], potential, first_layer[np.newaxis, :, :]], 
        axis=0)

    # Berechne die Gradienten unter Berücksichtigung der erweiterten Ebenen
    gradient_z, gradient_y, gradient_x = np.gradient(potential_z_padded)

    # Die negierten Gradienten repräsentieren die Kräfte
    force_field_x = -gradient_x[1:-1, :, :]
    force_field_y = -gradient_y[1:-1, :, :]
    force_field_rotation = -gradient_z[1:-1, :, :]

    # Ersetze NaN-Gradienten (an Hindernissen) durch manuell berechneten Gradienten
    force_field_x[obstacles_mask] = np.nan
    force_field_y[obstacles_mask] = np.nan
    force_field_rotation[obstacles_mask] = np.nan

    # Gradienten an Grenzen "clippen"
    force_field_x[:, :, 0][force_field_x[:, :, 0] < 0] = 0  # Setze alle Gradienten am linken Rand auf 0, wenn sie < 0 sind
    force_field_x[:, :, -1][force_field_x[:, :, -1] > 0] = 0  # Setze alle Gradienten am rechten Rand auf 0, wenn sie > 0 sind
    force_field_y[:, 0, :][force_field_y[:, 0, :] < 0] = 0  # Setze alle Gradienten am oberen Rand auf 0, wenn sie < 0 sind
    force_field_y[:, -1, :][force_field_y[:, -1, :] > 0] = 0  # Setze alle Gradienten am unteren Rand auf 0, wenn sie > 0 sind

    return force_field_x, force_field_y, force_field_rotation


def compute_obstacle_gradients(force_field_x, force_field_y, force_field_rotation, potential):
    # This can create new local minima and plateaus:

    obstacle_coordinates = np.argwhere(np.isnan(potential))

    rotations_size = force_field_rotation.shape[0]

    nan_coordinates_force_field_x = np.argwhere(np.isnan(force_field_x))
    non_obstacle_nan_coordinates_force_field_x = set(map(tuple, nan_coordinates_force_field_x)) - set(map(tuple, obstacle_coordinates))
    #print("Number of non-obstacle 'force_field_x == nan' before: " + str(len(non_obstacle_nan_coordinates_force_field_x)))
    for z, y, x in non_obstacle_nan_coordinates_force_field_x:
        if x-1 >= 0 and np.isnan(potential[z, y, x-1]) and x+1 < potential.shape[2] and not np.isnan(potential[z, y, x+1]):
            # x is at "left border"
            gradient_manual_x = potential[z, y, x+1] - potential[z, y, x]
            force_field_x[z, y, x] = -gradient_manual_x if gradient_manual_x <= 0 else 0
        elif x+1 < potential.shape[2] and np.isnan(potential[z, y, x+1]) and x-1 >= 0 and not np.isnan(potential[z, y, x-1]):
            # x is at "right border"
            gradient_manual_x = potential[z, y, x] - potential[z, y, x-1]
            force_field_x[z, y, x] = -gradient_manual_x if gradient_manual_x >= 0 else 0
        elif x-1 < 0 or np.isnan(potential[z, y, x-1]) and x+1 >= potential.shape[2] or np.isnan(potential[z, y, x+1]):
            # is is between borders
            force_field_x[z, y, x] = 0

    nan_coordinates_force_field_x = np.argwhere(np.isnan(force_field_x))
    non_obstacle_nan_coordinates_force_field_x = set(map(tuple, nan_coordinates_force_field_x)) - set(map(tuple, obstacle_coordinates))
    if len(non_obstacle_nan_coordinates_force_field_x) > 0:
        warnings.warn("Unresolved 'force_field_x == nan' at: " + str(non_obstacle_nan_coordinates_force_field_x))
    #else:
        #print("Number of non-obstacle 'force_field_x == nan' after: " + str(len(non_obstacle_nan_coordinates_force_field_x)))

    nan_coordinates_force_field_y = np.argwhere(np.isnan(force_field_y))
    non_obstacle_nan_coordinates_force_field_y = set(map(tuple, nan_coordinates_force_field_y)) - set(map(tuple, obstacle_coordinates))
    #print("Number of non-obstacle 'force_field_y == nan' before: " + str(len(non_obstacle_nan_coordinates_force_field_y)))
    for z, y, x in non_obstacle_nan_coordinates_force_field_y:
        if y-1 >= 0 and np.isnan(potential[z, y-1, x]) and y+1 < potential.shape[1] and not np.isnan(potential[z, y+1, x]):
            # y is at "top border"
            gradient_manual_y = potential[z, y+1, x] - potential[z, y, x]
            force_field_y[z, y, x] = -gradient_manual_y if gradient_manual_y <= 0 else 0   
        elif y+1 < potential.shape[1] and np.isnan(potential[z, y+1, x]) and y-1 >= 0 and not np.isnan(potential[z, y-1, x]):
            # y is at "bottom border"
            gradient_manual_y = potential[z, y, x] - potential[z, y-1, x]
            force_field_y[z, y, x] = -gradient_manual_y if gradient_manual_y >= 0 else 0
        elif y+1 >= potential.shape[1] or np.isnan(potential[z, y+1, x]) and y-1 < 0 or np.isnan(potential[z, y-1, x]):
            # y is between borders
            force_field_y[z, y, x] = 0


    nan_coordinates_force_field_y = np.argwhere(np.isnan(force_field_y))
    non_obstacle_nan_coordinates_force_field_y = set(map(tuple, nan_coordinates_force_field_y)) - set(map(tuple, obstacle_coordinates))
   
    if len(non_obstacle_nan_coordinates_force_field_y) > 0:
        warnings.warn("Unresolved 'force_field_y == nan' at: " + str(non_obstacle_nan_coordinates_force_field_y))
    #else:
        #print("Number of non-obstacle 'force_field_y == nan' after: " + str(len(non_obstacle_nan_coordinates_force_field_y)))

    nan_coordinates_force_field_rotation = np.argwhere(np.isnan(force_field_rotation))
    non_obstacle_nan_coordinates_force_field_rotation = set(map(tuple, nan_coordinates_force_field_rotation)) - set(map(tuple, obstacle_coordinates))
    #print("Number of non-obstacle 'force_field_rotation == nan' before: " + str(len(non_obstacle_nan_coordinates_force_field_rotation)))
    for z, y, x in non_obstacle_nan_coordinates_force_field_rotation:
        if np.isnan(potential[(z+1) % rotations_size, y, x]) and not np.isnan(potential[(z-1) % rotations_size, y, x]):
            # z is at "upper border"
            gradient_manual_z = potential[z, y, x] - potential[(z-1) % rotations_size, y, x] # caution!
            force_field_rotation[z, y, x] = -gradient_manual_z if gradient_manual_z >= 0 else 0  # caution!
        elif np.isnan(potential[(z-1) % rotations_size, y, x]) and not np.isnan(potential[(z+1) % rotations_size, y, x]):
            # z is at "lower border"
            gradient_manual_z = potential[(z+1) % rotations_size, y, x] - potential[z, y, x]
            force_field_rotation[z, y, x] = -gradient_manual_z if gradient_manual_z <= 0 else 0
        elif np.isnan(potential[(z-1) % rotations_size, y, x]) and np.isnan(potential[(z+1) % rotations_size, y, x]):
            # z is between borders
            force_field_rotation[z, y, x] = 0


    nan_coordinates_force_field_rotation = np.argwhere(np.isnan(force_field_rotation))
    non_obstacle_nan_coordinates_force_field_rotation = set(map(tuple, nan_coordinates_force_field_rotation)) - set(map(tuple, obstacle_coordinates))
    if len(non_obstacle_nan_coordinates_force_field_rotation) > 0:
        warnings.warn("Unresolved 'force_field_rotation == nan' at: " + str(non_obstacle_nan_coordinates_force_field_rotation))
    #else:
        #print("Number of non-obstacle 'force_field_rotation == nan' after: " + str(len(non_obstacle_nan_coordinates_force_field_rotation)))

def fix_local_maxima(force_field_x, force_field_y, force_field_rotation, potential, goal_point):
    obstacles_mask = np.isnan(potential)
    goal_x, goal_y, goal_rotation = goal_point

    rotations_size = force_field_rotation.shape[0]

    local_extrema_with_goal_before = np.argwhere((force_field_x == 0) & (force_field_y == 0) & (force_field_rotation == 0) & ~obstacles_mask)
    # Index des Eintrags zum Entfernen finden
    local_extrema_before = np.delete(
        local_extrema_with_goal_before,
        np.where(
            (local_extrema_with_goal_before[:, 0] == goal_rotation) &
            (local_extrema_with_goal_before[:, 1] == goal_y) &
            (local_extrema_with_goal_before[:, 2] == goal_x))[0],
        axis=0)    
    
    for z, y, x in local_extrema_before:

        if x-1 >= 0 and potential[z, y, x-1] is not np.nan and\
           x+1 < force_field_x.shape[2] and potential[z, y, x-1] is not np.nan:
           # due to implementation of "compute_gradients", local x-axis maxima cannot exist at x-axis border or next to x-axis obstacle

            if potential[z, y, x+1] < potential[z, y, x]:
                # local x-axis maxima found at (potential[z,y,x-1] < potential[z,y,x] > potential[z,y,x+1]) 
                gradient_local_max_x = potential[z, y, x+1] - potential[z, y, x] # Naive decision: Go to right neighbour
                force_field_x[z, y, x] = -gradient_local_max_x


        if y-1 >= 0 and potential[z, y-1, x] is not np.nan and\
           y+1 < force_field_x.shape[2] and potential[z, y+1, x] is not np.nan:
           # due to implementation of "compute_gradients", local x-axis maxima cannot exist at y-axis border or next to y-axis obstacle

            if potential[z, y-1, x] < potential[z, y, x]:
                # local maxima in (potential[z,y-1,x] < potential[z,y,x] > potential[z,y+1,x]) found
                gradient_local_min_y = potential[z, y-1, x] - potential[z, y, x] # Naive decision: Go to top neighbour
                force_field_x[z, y, x] = gradient_local_min_y

        if potential[(z+1) % rotations_size, y, x] is not np.nan and\
           potential[(z-1) % rotations_size, y, x] is not np.nan:
           # due to implementation of "compute_gradients", local z-axis maxima cannot exist next to z-axis obstacles

            if potential[(z+1) % rotations_size, y, x] < potential[z, y, x]:
                # local maxima in potential[(z-1) % rotations_size, y, x] < potential[z,y,x] > potential[(z+1) % rotations_size, y, x] found
                gradient_local_min_rotation = potential[(z+1) % rotations_size, y, x] - potential[z, y, x]
                force_field_rotation[z, y, x] = -gradient_local_min_rotation


    local_extrema_after_with_goal = np.argwhere((force_field_x == 0) & (force_field_y == 0) & (force_field_rotation == 0) & ~obstacles_mask)
    local_extrema_after = np.delete(
        local_extrema_after_with_goal,
        np.where(
            (local_extrema_after_with_goal[:, 0] == goal_rotation) &
            (local_extrema_after_with_goal[:, 1] == goal_y) &
            (local_extrema_after_with_goal[:, 2] == goal_x))[0],
        axis=0)    
    if len(local_extrema_after) > 0:
        warnings.warn("Local force minima or plateau at: " + str(local_extrema_after))


def gradient_descent_step(current_position, force_field_x, force_field_y, force_field_rotation, path, goal_point):

    if current_position == goal_point:
        raise Exception("DONE!")

    x, y, rotation = current_position

    force_x = force_field_x[rotation, y, x]
    force_y = force_field_y[rotation, y, x]
    force_rotation = force_field_rotation[rotation, y, x]

    if force_x is np.nan or force_y is np.nan or force_rotation is np.nan:
        raise Exception("Obstacle hit at: x=" + str(x) + ", y=" + str(y) + ", rotation=" + str(rotation))

    # Füge die aktuelle Position zum Pfad hinzu
    path.append((x, y, rotation))

    possible_moves = []

    if abs(force_x) != 0 and abs(force_x) >= abs(force_y) and abs(force_x) >= abs(force_rotation):
        new_x = x + 1 if force_x > 0 else x - 1
        if 0 <= new_x < force_field_y.shape[2]:
            possible_moves.append((new_x, y, rotation))
    
    if abs(force_y) != 0 and abs(force_y) >= abs(force_x) and abs(force_y) >= abs(force_rotation):
        new_y = y + 1 if force_y > 0 else y - 1
        if 0 <= new_y < force_field_y.shape[1]:
            possible_moves.append((x, new_y, rotation))
    
    if abs(force_rotation) != 0 and abs(force_rotation) >= abs(force_x) and abs(force_rotation) >= abs(force_y):
        new_rotation = (rotation + 1) % len(force_field_rotation) if force_rotation > 0 else (rotation - 1) % len(force_field_rotation)
        possible_moves.append((x, y, new_rotation))

    if goal_point in possible_moves:
        return goal_point

    # Überprüfe, ob die neue Position bereits im Pfad enthalten ist
    valid_moves = [move for move in possible_moves if move not in path]

    if valid_moves:
        return min(valid_moves, key=lambda move: abs(force_field_x[move[2], move[1], move[0]]) + abs(force_field_y[move[2], move[1], move[0]]) + abs(force_field_rotation[move[2], move[1], move[0]]))
    else:
        raise Exception("Local mininum or plateau at current location: x=" + str(x) + ", y=" + str(y) + ", rotation=" + str(rotation))


from occupancy_grid import plot_occupancy_grid
from configuration_space import plot_configuration_space

def update_gradient_descent_plots(occupancy_grid, configuration_space, start_point, goal_point, current_position, robot_width, robot_length, rotation_step, ax_occupancy_grid, gradient_decent_plots=False, ax_cs_2D=None, ax_cs_3D=None, force_field_x=None, force_field_y=None, force_field_rotation=None, path=[]):
    plot_occupancy_grid(
        occupancy_grid=occupancy_grid,
        start_point=start_point,
        goal_point=goal_point,
        current_position=current_position,
        robot_width=robot_width,
        robot_length=robot_length,
        plot_axis=ax_occupancy_grid,
        rotation_step=rotation_step,
        path=path,
        ticks=False
    )

    if gradient_decent_plots:
        if ax_cs_3D is not None and force_field_x is not None and force_field_y is not None and force_field_rotation is not None:
            plot_configuration_space(
                configuration_space=configuration_space, 
                force_field_x=force_field_x, 
                force_field_y=force_field_y, 
                force_field_rotation=force_field_rotation, 
                start_point=start_point,
                goal_point=goal_point, 
                current_position=current_position,
                rotation_step=rotation_step,
                path=path,
                ax=ax_cs_3D)

        if ax_cs_2D is not None:
            for index, ax in enumerate(ax_cs_2D):      
                plot_occupancy_grid(
                    occupancy_grid=configuration_space[index], 
                    goal_point=goal_point, 
                    plot_axis=ax, 
                    start_point=start_point,
                    plot_title="Occupancy Grid\n" + str(index * rotation_step) + "° Rotation",
                    ticks=False,
                    current_position=current_position,
                    rotation_step=rotation_step,
                    path=path,
                    active= current_position[2] == index)