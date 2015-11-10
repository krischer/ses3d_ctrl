import collections
import os
import re

import numpy as np

from .ses3d_tools.models import SES3DModel


def sum_kernels_spectral_element_grid(kernel_dirs, output_dir,
                                      clip_percentile):
    """
    Sum the kernels on the spectral element grid. This is much more
    efficient and should be identical to first projecting and then
    summing them.

    :param kernel_dirs: List of directories. One for each kernel.
    :param output_dir: The output directory.
    :param clip_percentile: The upper clip percentile to remove singularities.
    """
    pattern = re.compile(r"^(grad_cp_|grad_csh_|grad_csv_|grad_rho_)[0-9]+$")

    contents_per_field = collections.defaultdict(list)
    contents_per_folder_and_gradient = collections.defaultdict(list)

    gradient_types = ("grad_cp_", "grad_csh_", "grad_csv_", "grad_rho_")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for directory in kernel_dirs:
        l = os.listdir(directory)

        # Keep only the valid files.
        l = [_i for _i in l if re.match(pattern, _i)]
        for filename in l:
            contents_per_field[filename].append(directory)

        for grad in gradient_types:
            for filename in l:
                if not filename.startswith(grad):
                    continue
                contents_per_folder_and_gradient[(directory, grad)].append(
                    filename)

    clipping_values = {}

    # We unfortunately have to do two passes. The first to find the
    # percentiles of the various gradients and the second to clip and sum
    # them. The clipping happens before the summing.

    # 1. Calculate the clipping values.
    for key, value in contents_per_folder_and_gradient.items():
        arrays = []
        for filename in value:
            filename = os.path.join(key[0], filename)
            # Omit the first and last value as they are somehow filled with
            # things by Fortran.
            arrays.append(np.fromfile(filename, dtype=np.float32)[1:-1])

        # Temporary memory hog right here! Should not matter in the long run.
        percentile = np.percentile(np.abs(np.concatenate(arrays)),
                                   clip_percentile)
        clipping_values[key] = percentile

    # Sum and clip em!
    for variable, folders in contents_per_field.items():
        summed_data = None
        for folder in folders:
            # Get the corresponding clipping value.
            clipping_value = clipping_values[(folder,
                                              re.sub("\d+$", "", variable))]
            data = np.fromfile(os.path.join(folder, variable),
                               dtype=np.float32)
            # Clip the data.
            np.clip(data[1: -1], -clipping_value, clipping_value)
            if summed_data is None:
                summed_data = data
            else:
                summed_data[1:-1] += data[1:-1]

        summed_data.tofile(os.path.join(output_dir, variable))


def sum_kernels(kernel_dirs, output_dir, clip_percentile=99.9):
    """
    Sums up all kernels found in the kernel directories and write the summed
    kernel to output_dir.

    :param kernel_dirs: A list of directories containing kernels in the
        SES3D block format.
    :param output_dir: The output directory. Must not yet exist.
    :param clip_percentile: The upper clip percentile to remove singularities.
    """
    if os.path.exists(output_dir):
        raise ValueError("Directory '%s' already exists." % output_dir)

    grad_csv = None
    grad_csh = None
    grad_rho = None
    grad_cp = None

    import time

    for directory in kernel_dirs:
        a = time.time()
        # Read to temporary structure.
        temp_grad_csv = SES3DModel()
        temp_grad_csh = SES3DModel()
        temp_grad_rho = SES3DModel()
        temp_grad_cp = SES3DModel()
        temp_grad_csv.read(directory, filename="gradient_csv")
        temp_grad_csh.read(directory, filename="gradient_csh")
        temp_grad_rho.read(directory, filename="gradient_rho")
        temp_grad_cp.read(directory, filename="gradient_cp")
        b = time.time()
        print "Time for reading:", b - a

        a = time.time()
        # Clip them to avoid singularities.
        temp_grad_csv.clip_percentile(clip_percentile)
        temp_grad_csh.clip_percentile(clip_percentile)
        temp_grad_rho.clip_percentile(clip_percentile)
        temp_grad_cp.clip_percentile(clip_percentile)
        b = time.time()
        print "Time for clipping:", b - a

        a = time.time()
        # Sum them up.
        if grad_csv is None and grad_csh is None and grad_rho is None and \
                grad_cp is None:
            grad_csv = temp_grad_csv
            grad_csh = temp_grad_csh
            grad_rho = temp_grad_rho
            grad_cp = temp_grad_cp
        else:
            grad_csv = grad_csv + temp_grad_csv
            grad_csh = grad_csh + temp_grad_csh
            grad_rho = grad_rho + temp_grad_rho
            grad_cp = grad_cp + temp_grad_cp
        b = time.time()
        print "Time for summing:", b - a

    a = time.time()
    # Write the summed kernels.
    os.makedirs(output_dir)
    grad_csv.write(output_dir, "gradient_csv")
    grad_csh.write(output_dir, "gradient_csh")
    grad_rho.write(output_dir, "gradient_rho")
    grad_cp.write(output_dir, "gradient_cp")
    b = time.time()
    print "Time for writing:", b - a
