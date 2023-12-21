import numpy as np
from scipy.ndimage import rotate

def compute_robot_mask(robot_length, robot_width, rotation_angle):
    no_rotation_mask = np.ones((robot_length, robot_width))
    no_rotation_mask[0, 0] = np.inf
    rotated_mask = rotate(no_rotation_mask, rotation_angle, reshape=True, cval=0, order=0, mode='grid-constant', prefilter=True)

    if len(np.argwhere(np.isinf(rotated_mask))) == 0:

        rotated_mask_nearest = rotate(no_rotation_mask, rotation_angle, reshape=True, cval=0, order=0, mode='nearest', prefilter=True)
        inf_indices = np.argwhere(rotated_mask_nearest == np.inf)
        if len(inf_indices) == 0:
            raise Exception("Inf was interpolated in: " + str(rotated_mask_nearest))
        min_inf_y, min_inf_x = inf_indices[np.argmin(inf_indices[:, 0])]

        zero_indices = np.argwhere(rotated_mask == 0)

        rotated_mask_new_interpolated = np.ones(rotated_mask.shape)

        for y, x in zero_indices:
            rotated_mask_new_interpolated[y, x] = 0

        rotated_mask_new_interpolated[min_inf_y, min_inf_x] = np.inf

        if len(np.argwhere(rotated_mask_new_interpolated == np.inf)) == 0:
            raise Exception("Inf was interpolated in: " + str(rotated_mask_nearest))

        rotated_mask = rotated_mask_new_interpolated

    point_mirrored_mask = np.flipud(np.fliplr(rotated_mask)) # Punktspiegeln
    return point_mirrored_mask


def generate_expanded_occu_grid(occupancy_grid, robot_mask):
    # Rand als Hindernis
    expanded_grid = np.pad(occupancy_grid, 1, constant_values=False)
    generate_expanded_occu_grid = np.copy(expanded_grid)

    # Koordinaten von 1 in der Robot Mask
    robot_mask_ones = np.argwhere(robot_mask == 1)

    inf_y, inf_x = np.argwhere(np.isinf(robot_mask))[0]

    # Relative Koordinaten von 1 in der Robot Mask zu np.inf
    robot_mask_ones_relative = robot_mask_ones - np.array([inf_y, inf_x])

    for one_relative_y, one_relative_x in robot_mask_ones_relative:
        # Iteriere über Hindernisse
        for obstacle_coord in np.argwhere(expanded_grid == False):
            obst_y, obst_x = obstacle_coord

            # Neue Koordinaten nach Anwendung der Relativverschiebung
            new_obst_y = obst_y + one_relative_y
            new_obst_x = obst_x + one_relative_x

            # Überprüfe, ob die neuen Koordinaten innerhalb der Grenzen von expanded_grid liegen
            if 0 <= new_obst_y < expanded_grid.shape[0] and 0 <= new_obst_x < expanded_grid.shape[1]:
                generate_expanded_occu_grid[new_obst_y, new_obst_x] = False

    return generate_expanded_occu_grid[1:-1, 1:-1]  # Entferne zusätzlich hinzugefügte Dimensionen


def compute_configuration_space(robot_length, robot_width, rotation_step, occupancy_grid):
    if 360 % rotation_step != 0:
        raise Exception("Rotationschritt muss Teiler von 360 sein")

    current_rotation = 0
    configuration_space_list = []

    rotations = np.arange(0, 360, rotation_step)

    for current_rotation in rotations:
        current_robot_mask = compute_robot_mask(
            robot_length = robot_length,
            robot_width = robot_width,
            rotation_angle = current_rotation
        )
        current_configuration_space = generate_expanded_occu_grid(
            occupancy_grid=occupancy_grid,
            robot_mask=current_robot_mask
        )
        configuration_space_list.append(current_configuration_space)
        
    configuration_space = np.array(configuration_space_list)
    return configuration_space


def plot_configuration_space(configuration_space, ax, rotation_step, force_field_x=None, force_field_y=None, force_field_rotation=None, start_point=None, goal_point=None, current_position=None, path=[]):
    ax.clear()
    rotation, size_y, size_x = configuration_space.shape

    if force_field_x is not None and force_field_y is not None and force_field_rotation is not None:
        if configuration_space.shape != force_field_x.shape:
            raise Exception("Shape mismatch. configuration_space.shape=" + str(configuration_space.shape) + ", force_field_x.shape=" + str(force_field_x.shape))
        if configuration_space.shape != force_field_y.shape:
            raise Exception("Shape mismatch. configuration_space.shape=" + str(configuration_space.shape) + ", force_field_y.shape=" + str(force_field_y.shape))
        if configuration_space.shape != force_field_rotation.shape:
            raise Exception("Shape mismatch. configuration_space.shape=" + str(configuration_space.shape) + ", force_field_rotation.shape=" + str(force_field_rotation.shape))

        force_field_x_plot = np.nan_to_num(force_field_x)
        force_field_y_plot = np.nan_to_num(force_field_y)
        force_field_rotation_plot = np.nan_to_num(force_field_rotation)
        max_force = np.max(np.sqrt(force_field_x_plot**2 + force_field_y_plot**2 + force_field_rotation_plot**2))

    obstacle_color = 'deepskyblue'
    start_color = 'green'
    goal_color = 'red'
    current_color = 'orange'
    path_color = 'orange'

    for r in range(rotation):
        for y in range(size_y):
            for x in range(size_x):
                if not configuration_space[r, y, x]:
                    # Plotte einen Würfel mit gleichen Abmessungen in allen Richtungen
                    ax.bar3d(x - 0.5, y - 0.5, r, 1, 1, 1, color=obstacle_color, alpha=0.3)
                else:
                    if force_field_x is not None and force_field_y is not None and force_field_rotation is not None:
                        # Berechne die Länge des Kraftvektors für diesen Punkt
                        force_length = np.sqrt(force_field_x_plot[r, y, x]**2 + force_field_y_plot[r, y, x]**2 + force_field_rotation_plot[r, y, x]**2)
                        # Normalisiere den Kraftvektor auf den Bereich [0, 1]
                        length = force_length / max_force if max_force != 0 else 0
                        ax.quiver(x, y, r + 0.5, force_field_x_plot[r, y, x], force_field_y_plot[r, y, x], force_field_rotation_plot[r, y, x], color='blue', length=length, normalize=True)

    if start_point is not None:
        start_x, start_y, start_rotation = start_point
        # Plotte einen Würfel für den Startpunkt
        ax.bar3d(start_x - 0.5, start_y - 0.5, start_rotation, 1, 1, 1, color=start_color, alpha=0.5)

    if goal_point is not None:
        goal_x, goal_y, goal_rotation = goal_point
        # Plotte einen Würfel für den Zielpunkt
        ax.bar3d(goal_x - 0.5, goal_y - 0.5, goal_rotation, 1, 1, 1, color=goal_color, alpha=0.5)

    if current_position is not None:
        current_x, current_y, current_rotation = current_position
        # Plotte einen Würfel für die aktuelle Position
        ax.bar3d(current_x - 0.5, current_y - 0.5, current_rotation, 1, 1, 1, color=current_color, alpha=0.5)

    if path != []:
        for path_x, path_y, path_rotation in path:
            ax.bar3d(path_x - 0.5, path_y - 0.5, path_rotation, 1, 1, 1, color=current_color, alpha=0.2)


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Rotation')

    # Invertiere die Y-Achse
    ax.invert_yaxis()

    # Passe die Skalierung der Z-Achse manuell an
    ax.set_xlim(0, size_x)
    ax.set_ylim(0, size_y)
    ax.set_zlim(0, rotation)

    # Passe die Z-Koordinaten an, um die Würfel über der XY-Ebene zu positionieren
    ax.set_box_aspect([size_x / rotation, size_y / rotation, 1])

    # Passe die Position der Z-Ticks an
    ax.set_zticks(np.arange(0, rotation, 1) + 0.5)
    ax.set_zticklabels([f'{angle}°' for angle in np.arange(0, 360, rotation_step)])

    # Passe die Position der X- und Y-Ticks an
    ax.set_xticks(np.arange(0, size_x, 1))
    ax.set_yticks(np.arange(0, size_y, 1))
    ax.invert_yaxis()