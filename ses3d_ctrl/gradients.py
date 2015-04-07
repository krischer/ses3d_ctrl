import os

from .ses3d_tools.models import SES3DModel


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
