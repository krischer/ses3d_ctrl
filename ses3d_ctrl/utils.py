import math
import os

import numpy as np

TEMPLATE_FOLDER = os.path.join(os.path.dirname(__file__), "templates")


def get_template(name):
    """
    Returns the template filename.
    """
    filename = os.path.join(TEMPLATE_FOLDER,
                            name + os.path.extsep + "template")
    if not os.path.exists(filename):
        raise ValueError("Cannot find template '%s'." % name)

    return filename


def read_boxfile(filename):
    """
    Quick and dirty boxfile parsing function.

    :param filename: The filename of the boxfile.
    """
    with open(filename, "rt") as fh:
        lines = fh.readlines()

    lines = [_i.strip() for _i in lines if _i.strip()]

    # First 14 lines are commentary.
    lines = lines[14:]
    global_info = lines[:4]
    procs_in_x = int(global_info[1])
    procs_in_y = int(global_info[2])
    procs_in_z = int(global_info[3])
    lines = lines[5:]

    lines = lines[2:]
    x_dim_min, x_dim_max = map(int, lines[0].split())
    y_dim_min, y_dim_max = map(int, lines[1].split())
    z_dim_min, z_dim_max = map(int, lines[2].split())
    x_min, x_max = map(float, lines[3].split())
    y_min, y_max = map(float, lines[4].split())
    z_min, z_max = map(float, lines[5].split())
    lines = lines[7:]

    while lines:
        lines = lines[2:]
        xdim_min, xdim_max = map(int, lines[0].split())
        ydim_min, ydim_max = map(int, lines[1].split())
        zdim_min, zdim_max = map(int, lines[2].split())
        xmin, xmax = map(float, lines[3].split())
        ymin, ymax = map(float, lines[4].split())
        zmin, zmax = map(float, lines[5].split())
        lines = lines[7:]

        x_dim_min = min(x_dim_min, xdim_min)
        y_dim_min = min(y_dim_min, ydim_min)
        z_dim_min = min(z_dim_min, zdim_min)

        x_dim_max = max(x_dim_max, xdim_max)
        y_dim_max = max(y_dim_max, ydim_max)
        z_dim_max = max(z_dim_max, zdim_max)

        x_min = min(x_min, xmin)
        y_min = min(y_min, ymin)
        z_min = min(z_min, zmin)

        x_max = max(x_max, xmax)
        y_max = max(y_max, ymax)
        z_max = max(z_max, zmax)

    # Convert to degree.
    x_min = math.degrees(x_min)
    y_min = math.degrees(y_min)
    x_max = math.degrees(x_max)
    y_max = math.degrees(y_max)

    return {
        "x_dim_min": x_dim_min,
        "x_dim_max": x_dim_max,
        "y_dim_min": y_dim_min,
        "y_dim_max": y_dim_max,
        "z_dim_min": z_dim_min,
        "z_dim_max": z_dim_max,
        "element_count_x": x_dim_max - x_dim_min,
        "element_count_y": y_dim_max - y_dim_min,
        "element_count_z": z_dim_max - z_dim_min,
        "x_min": x_min,
        "y_min": y_min,
        "z_min": z_min,
        "x_max": x_max,
        "y_max": y_max,
        "z_max": z_max,
        "x_range": abs(x_max - x_min),
        "y_range": abs(y_max - y_min),
        "z_range": abs(z_max - z_min),
        "processors_in_x": procs_in_x,
        "processors_in_y": procs_in_y,
        "processors_in_z": procs_in_z
    }


def write_ses3d_blockfiles(output_folder, x_min, x_max, x_count,
                           y_min, y_max, y_count,
                           z_min, z_max, z_count):
    """
    Write SES3D blockfiles to the specified output folder.

    :param output_folder: The output folder.
    :param x_min: x min value
    :param x_max: x max value
    :param x_count: x step value
    :param y_min: y min value
    :param y_max: y max value
    :param y_count: y step value
    :param z_min: z min value
    :param z_max: z max value
    :param z_count: z step value
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    x = np.linspace(x_min, x_max, x_count)
    y = np.linspace(y_min, y_max, y_count)
    z = np.linspace(z_min, z_max, z_count)

    def write_block(var, name):
        filename = os.path.join(output_folder, "block_%s" % name)
        with open(filename, "wt") as fh:
            fh.write("1\n%i\n" % len(var))
            for _i in var:
                fh.write("%.4f\n" % _i)

    write_block(x, "x")
    write_block(y, "y")
    write_block(z, "z")
