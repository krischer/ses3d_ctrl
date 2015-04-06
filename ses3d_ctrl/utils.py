import math
import os

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
    lines = lines[5:]

    lines = lines[5:]
    x_min, x_max = map(float, lines[0].split())
    y_min, y_max = map(float, lines[1].split())
    z_min, z_max = map(float, lines[2].split())
    lines = lines[4:]

    while lines:
        lines = lines[5:]
        xmin, xmax = map(float, lines[0].split())
        ymin, ymax = map(float, lines[1].split())
        zmin, zmax = map(float, lines[2].split())
        lines = lines[4:]

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
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "x_range": abs(x_max - x_min),
        "y_range": abs(y_max - y_min),
    }
