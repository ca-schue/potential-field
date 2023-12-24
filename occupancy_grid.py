import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_occupancy_grid(occupancy_grid, plot_axis, rotation_step, goal_point=None, current_position=None, path=[], robot_width=None, robot_length=None, active=True, plot_title="Occupancy Grid", y_axis_label=None, start_point=None, ticks=True):
    plot_axis.clear()

    if active:
        cmap = ListedColormap(['darkblue', 'white'])     
        cmap_empty = ListedColormap(['white'])        
        start_color = 'green'
        goal_color = 'red'
        path_color = 'darkkhaki'
        current_color = 'orange'
    else:
        cmap = 'gray'
        start_color = 'black'
        goal_color = 'dimgray'
        path_color = 'darkgray'
        current_color = 'gray'
        line_color = 'black'

    if np.count_nonzero(~occupancy_grid) > 0:
        print("np.count_nonzero(occupancy_grid) > 0")
        plot_axis.imshow(occupancy_grid, cmap=cmap, interpolation='none', origin='upper')
    else:
        plot_axis.imshow(occupancy_grid, cmap=cmap_empty, interpolation='none', origin='upper')



    if goal_point is not None:
        goal_x, goal_y, goal_rotation = goal_point
        if robot_width is None or robot_length is None:
            # Darstellung des aktuellen Roboters als x
            if not occupancy_grid[goal_y, goal_x]:
                plot_axis.scatter(goal_x, goal_y, color=goal_color, marker='x', label='Goal Position')
            else:
                plot_axis.scatter(goal_x, goal_y, color=goal_color, marker='o', label='Goal Position')
        else: # Darstellung des aktuellen Roboters als Rechteck
            goal_rect = plt.Rectangle(
                (goal_x-0.5, goal_y-0.5),
                robot_width,
                robot_length,
                rotation_point=(goal_x, goal_y),
                angle=goal_rotation * -rotation_step,  # Rotation in Grad umrechnen
                color=goal_color,
                label='Start Position'
            )
            plot_axis.add_patch(goal_rect)
            plot_axis.scatter(goal_x, goal_y, color='darkred', marker='x', label='Rotation Anchor')

    if start_point is not None:
        start_x, start_y, start_rotation = start_point
        if robot_width is None or robot_length is None:
            # Darstellung des aktuellen Roboters als x
            if not occupancy_grid[start_y, start_x]:
                plot_axis.scatter(start_x, start_y, color=start_color, marker='x', label='Start Position')
            else:
                plot_axis.scatter(start_x, start_y, color=start_color, marker='o', label='Start Position')
        else: # Darstellung des aktuellen Roboters als Rechteck
            start_rect = plt.Rectangle(
                (start_x-0.5,  start_y-0.5),
                robot_width,
                robot_length,
                rotation_point=(start_x, start_y),
                angle=start_rotation * -rotation_step,  # Rotation in Grad umrechnen
                color=start_color,
                label='Start Position'
            )
            plot_axis.add_patch(start_rect)
            plot_axis.scatter(start_x, start_y, color='darkgreen', marker='x', label='Start Position')

    if path != []:
        for path_x, path_y, path_rotation in path:
            if robot_width is None or robot_length is None:
                # Darstellung des aktuellen Roboters als x
                if not occupancy_grid[path_y, path_x]:
                    plot_axis.scatter(path_x, path_y, color=path_color, marker='x', label='Path', alpha=0.3)
                else:
                    plot_axis.scatter(path_x, path_y, color=path_color, marker='o', label='Path', alpha=0.3)
                    # Darstellung des aktuellen Roboters als Rechteck
            else:
                path_rect = plt.Rectangle(
                    (path_x-0.5, path_y-0.5),
                    robot_width,
                    robot_length,
                    rotation_point=(path_x, path_y),
                    angle=path_rotation * -rotation_step,  # Rotation in Grad umrechnen
                    label='Current Position',
                    alpha=0.3,
                    edgecolor='darkkhaki',
                    facecolor='palegoldenrod'
                )
                plot_axis.add_patch(path_rect)
                plot_axis.scatter(path_x, path_y, color='darkgoldenrod', marker='x', label='Rotation Anchor', alpha=0.5)

    if current_position is not None:
        current_x, current_y, current_rotation = current_position
        if robot_width is None or robot_length is None:
            # Darstellung des aktuellen Roboters als x
            if not occupancy_grid[current_y, current_x]:
                plot_axis.scatter(current_x, current_y, color=current_color, marker='x', label='Current Position')
            else:
                plot_axis.scatter(current_x, current_y, color=current_color, marker='o', label='Current Position')
                # Darstellung des aktuellen Roboters als Rechteck
        else:
            current_rect = plt.Rectangle(
                (current_x-0.5, current_y-0.5),
                robot_width,
                robot_length,
                rotation_point=(current_x, current_y),
                angle=current_rotation * -rotation_step,  # Rotation in Grad umrechnen
                color=current_color,
                label='Current Position'
            )
            plot_axis.add_patch(current_rect)
            plot_axis.scatter(current_x, current_y, color='black', marker='x', label='Rotation Anchor')

    plot_axis.grid(False)
    plot_axis.set_title(plot_title)

    # Beschriftung der X-Achse oben
    plot_axis.xaxis.tick_top()

    if ticks:
        plot_axis.set_xticks(range(occupancy_grid.shape[1]))
        plot_axis.set_yticks(range(occupancy_grid.shape[0]))
        plot_axis.grid()
    else:
        plot_axis.set_xticks([])
        plot_axis.set_yticks([])

    if y_axis_label is not None:
        # Optionale Beschriftung Y-Achse
        plot_axis.text(-0.15, 0.5, y_axis_label, rotation='horizontal', va='center', ha='right', transform=plot_axis.transAxes)


def add_obstacle(occupancy_grid, width, length, x, y):
    width_occ_grid = len(occupancy_grid[0])
    height_occ_grid = len(occupancy_grid)

    start_row = y
    end_row = min(height_occ_grid, y + length)
    start_col = x
    end_col = min(width_occ_grid, x + width)

    for row in range(start_row, end_row):
        for col in range(start_col, end_col):
            occupancy_grid[row][col] = False

    return occupancy_grid

def generate_empty_occupancy_grid(length, width):
    empty_occupancy_grid = np.ones((length, width), dtype=bool)

    return empty_occupancy_grid

