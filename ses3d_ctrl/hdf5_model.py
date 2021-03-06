#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import io
import json
import math
import os
import shutil
import struct

import h5py
import numpy as np
import obspy
import xarray

# LASIF can already deal with the binary SES3D models. Thus we can utilize
# this here!
from lasif.ses3d_models import RawSES3DModelHandler
from lasif.scripts.lasif_cli import _find_project_comm


def binary_ses3d_to_hdf5_model(input_folder, lasif_project, output_filename):
    """
    Function converting a binary SES3D model consisting of many files to an
    HDF5 model.

    Requires access to a LASIF project that determines the potentially
    rotated geometry. Not super clean but workable.

    :param input_folder: The folder containing the input model.
    :param lasif_project: The folder with the LASIF project.
    :param output_filename: The output filename.
    """
    assert not os.path.exists(output_filename), \
        "'%s' already exists" % output_filename
    _dirname = os.path.dirname(output_filename)
    if not os.path.exists(_dirname) and _dirname:
        os.makedirs(_dirname)

    # We need the project to get the domain definition which stores the
    # rotation settings.
    comm = _find_project_comm(lasif_project, read_only_caches=True)

    if any(_i.startswith("grad_") for _i in os.listdir(input_folder)):
        model_type = "kernel"
    else:
        model_type = "earth_model"

    m = RawSES3DModelHandler(
        directory=input_folder, domain=comm.project.domain,
        model_type=model_type)

    f = h5py.File(output_filename)

    try:
        data_group = f.create_group("data")

        if model_type == "earth_model":
            # We will also store A, C, and Q which we don't invert for but
            # have to take into account in any case.
            components = ["vp", "vsh", "vsv", "rho", "A", "C"]
            # Q might not exist.
            if "Q" in m.components:
                components.append("Q")
        elif model_type == "kernel":
            components = ["grad_cp", "grad_csh", "grad_csv", "grad_rho"]
        else:
            raise NotImplementedError

        # Make it compatible with seismopt.
        rename_dict = {
            "grad_cp": "vp",
            "grad_csh": "vsh",
            "grad_csv": "vsv",
            "grad_rho": "rho",
        }

        for c in components:
            m.parse_component(c)
            _d = xarray.DataArray(
                np.require(m.parsed_components[c],
                           requirements=["C_CONTIGUOUS"]),
                coords=[90.0 - m.collocation_points_lats[::-1],
                        m.collocation_points_lngs,
                        (6371.0 - m.collocation_points_depth) * 1000.0],
                dims=["colatitude", "longitude", "radius_in_m"])

            # Write to HDF5 file.
            if c in rename_dict:
                c = rename_dict[c]

            data = _d.data

            # Make sure it is g\cm^3 in the hdf5 files.
            if c == "rho":
                if data.mean() > 1000.0:
                    data /= 1000.0

            data_group[c] = np.require(data, dtype=np.float32)

            data_group[c].attrs["variable_name"] = \
                np.string_((c + "\x00").encode())

        # Write coordinate axes.
        f["coordinate_0"] = np.require(_d.colatitude.data, dtype=np.float32)
        f["coordinate_0"].attrs["name"] = \
            np.string_(("colatitude" + "\x00").encode())
        f["coordinate_1"] = np.require(_d.longitude.data, dtype=np.float32)
        f["coordinate_1"].attrs["name"] = \
            np.string_(("longitude" + "\x00").encode())
        f["coordinate_2"] = np.require(_d.radius_in_m.data, dtype=np.float32)
        f["coordinate_2"].attrs["name"] = \
            np.string_(("radius_in_m" + "\x00").encode())

        # Create dimension scales.
        for c in components:
            if c in rename_dict:
                c = rename_dict[c]
            f["data"][c].dims[0].label = "colatitude"
            f["data"][c].dims[1].label = "longitude"
            f["data"][c].dims[2].label = "radius_in_m"

            f["data"][c].dims.create_scale(f["coordinate_0"], "values")
            f["data"][c].dims.create_scale(f["coordinate_1"], "values")
            f["data"][c].dims.create_scale(f["coordinate_2"], "values")

            f["data"][c].dims[0].attach_scale(f["coordinate_0"])
            f["data"][c].dims[1].attach_scale(f["coordinate_1"])
            f["data"][c].dims[2].attach_scale(f["coordinate_2"])

        # Also add some meta information.
        _meta = f.create_group("_meta")
        model_name = os.path.split(os.path.normpath(os.path.abspath(
            input_folder)))[-1]
        _meta.attrs["model_name"] = np.string_((model_name + "\x00").encode())

        # Everything needed to reconstruct the domain objects.
        _domain = _meta.create_group("domain")
        d = comm.project.domain
        _domain.attrs["min_longitude"] = d.min_longitude
        _domain.attrs["max_longitude"] = d.max_longitude
        _domain.attrs["min_latitude"] = d.min_latitude
        _domain.attrs["max_latitude"] = d.max_latitude
        _domain.attrs["min_depth_in_km"] = d.min_depth_in_km
        _domain.attrs["max_depth_in_km"] = d.max_depth_in_km
        _domain.attrs["rotation_axis"] = d.rotation_axis
        _domain.attrs["rotation_angle_in_degree"] = d.rotation_angle_in_degree
        _domain.attrs["boundary_width_in_degree"] = d.boundary_width_in_degree

        # We also need to store the boxfile.
        _meta.create_dataset("boxfile",
                             data=np.fromfile(m.boxfile, dtype=np.uint8))
    finally:
        try:
            f.close()
        except:
            pass


def hdf5_model_to_binary_ses3d_model(input_filename, output_folder):
    with h5py.File(input_filename, "r") as f:
        _hdf5_model_to_binary_ses3d_model(f=f,
                                          output_folder=output_folder)


def _hdf5_model_to_binary_ses3d_model(f, output_folder):
    lpd = 4

    assert not os.path.exists(output_folder), \
        "Folder '%s' already exists." % output_folder
    os.makedirs(output_folder)

    with io.BytesIO(f["_meta"]["boxfile"].value.tostring()) as buf:
        setup = _read_boxfile(buf)
        # Also write to output folder
        buf.seek(0, 0)
        with io.open(os.path.join(output_folder, "boxfile"), "wb") as fh:
            fh.write(buf.read())

    data = {}

    # SES3D internally expects a density in kg/m^3 - The hdf5 files might
    # have g/cm^3.
    rho = f["data"]["rho"][:]
    if rho.mean() < 1000:
        rho *= 1000.0
    data["rhoinv"] = 1.0 / rho

    data["mu"] = (f["data"]["vsh"][:] * 1000) ** 2 / data["rhoinv"]
    data["lambda"] = \
        (f["data"]["vp"][:] * 1000) ** 2 / data["rhoinv"] - 2 * data["mu"]
    data["A"] = (f["data"]["A"][:])
    data["B"] = (f["data"]["vsv"][:] * 1000) ** 2 / data["rhoinv"] - data["mu"]
    data["C"] = (f["data"]["C"][:])
    # Q might now always be given.
    if "Q" in f["data"]:
        data["Q"] = (f["data"]["Q"][:])

    for key in sorted(data.keys()):
        for number, domain in enumerate(setup["subdomains"]):
            x_min, x_max = domain["boundaries_x"]
            y_min, y_max = domain["boundaries_y"]
            z_min, z_max = domain["boundaries_z"]

            # Minimum indices
            x_min, y_min, z_min = \
                [lpd * _j for _j in (x_min, y_min, z_min)]
            # Maximum indices
            x_max, y_max, z_max = \
                [lpd * (_j + 1) for _j in (x_max, y_max, z_max)]

            _d = data[key][x_min: x_max + 1,
                           y_min: y_max + 1,
                           z_min: z_max + 1]
            # Invert last components.
            _d = _d[:, :, ::-1]
            # Reduplicate the GLL points.
            for _i in xrange(3):
                _s = _d.shape[_i]
                left_idx = np.arange(_s - lpd)[::lpd]
                right_idx = np.arange(_s + lpd)[lpd + 1::lpd]
                if _i == 0:
                    _t = [_d[_l:_r, :, :]
                          for _l, _r in zip(left_idx, right_idx)]
                elif _i == 1:
                    _t = [_d[:, _l:_r, :]
                          for _l, _r in zip(left_idx, right_idx)]
                elif _i == 2:
                    _t = [_d[:, :, _l:_r]
                          for _l, _r in zip(left_idx, right_idx)]
                else:
                    raise NotImplementedError

                _d = np.concatenate(_t, axis=_i)
            # Reshape to restore 6 dimensional layout.
            _d = np.require(_d, requirements=["C_CONTIGUOUS"])
            shape = (domain["index_x_count"], lpd + 1,
                     domain["index_y_count"], lpd + 1,
                     domain["index_z_count"], lpd + 1)
            _d = _d.reshape(shape, order="C")
            # Roll to retrieve original SES3D memory order.
            _d = np.rollaxis(_d, 2, 1)
            _d = np.rollaxis(_d, 4, 2)
            _d = np.require(_d, requirements=["F_CONTIGUOUS"])

            filename = os.path.join(output_folder, "%s%i" % (key, number))
            with io.open(filename, "wb") as fh:
                fh.write(struct.pack("<I", 520000))
                fh.write(_d.tobytes(order="F"))
                fh.write(struct.pack("<I", 520000))


def _read_boxfile(fh):
    """
    Copied straight from LASIF.
    """
    setup = {"subdomains": []}

    # The first 14 lines denote the header
    lines = fh.readlines()[14:]
    # Strip lines and remove empty lines.
    lines = [_i.strip() for _i in lines if _i.strip()]

    # The next 4 are the global CPU distribution.
    setup["total_cpu_count"] = int(lines.pop(0))
    setup["cpu_count_in_x_direction"] = int(lines.pop(0))
    setup["cpu_count_in_y_direction"] = int(lines.pop(0))
    setup["cpu_count_in_z_direction"] = int(lines.pop(0))

    if set(lines[0]) == set("-"):
        lines.pop(0)
    # Small sanity check.
    if setup["total_cpu_count"] != setup["cpu_count_in_x_direction"] * \
            setup["cpu_count_in_y_direction"] * \
            setup["cpu_count_in_z_direction"]:
        msg = ("Invalid boxfile. Total and individual processor "
               "counts do not match.")
        raise ValueError(msg)

    # Now parse the rest of file which contains the subdomains.
    def subdomain_generator(data):
        """
        Simple generator looping over each defined box and yielding
        a dictionary for each.

        :param data: The text.
        """
        while data:
            subdom = {}
            # Convert both indices to 0-based indices
            subdom["single_index"] = int(data.pop(0)) - 1
            subdom["multi_index"] = map(lambda x: int(x) - 1,
                                        data.pop(0).split())
            subdom["boundaries_x"] = map(int, data.pop(0).split())
            subdom["boundaries_y"] = map(int, data.pop(0).split())
            subdom["boundaries_z"] = map(int, data.pop(0).split())
            # Convert radians to degree.
            subdom["physical_boundaries_x"] = map(
                    lambda x: math.degrees(float(x)), data.pop(0).split())
            subdom["physical_boundaries_y"] = map(
                    lambda x: math.degrees(float(x)), data.pop(0).split())
            # z is in meter.
            subdom["physical_boundaries_z"] = \
                map(float, data.pop(0).split())
            for component in ("x", "y", "z"):
                idx = "boundaries_%s" % component
                index_count = subdom[idx][1] - subdom[idx][0] + 1
                subdom["index_%s_count" % component] = index_count
                # The boxfiles are slightly awkward in that the indices
                # are not really continuous. For example if one box
                # has 22 as the last index, the first index of the next
                # box will also be 22, even though it should be 23. The
                # next snippet attempts to fix this deficiency.
                offset = int(round(subdom[idx][0] /
                                   float(index_count - 1)))
                subdom[idx][0] += offset
                subdom[idx][1] += offset
            # Remove separator_line if existent.
            if set(lines[0]) == set("-"):
                lines.pop(0)
            yield subdom
    # Sort them after with the single index.
    setup["subdomains"] = sorted(list(subdomain_generator(lines)),
                                 key=lambda x: x["single_index"])
    # Do some more sanity checks.
    if len(setup["subdomains"]) != setup["total_cpu_count"]:
        msg = ("Invalid boxfile. Number of processors and subdomains "
               "to not match.")
        raise ValueError(msg)
    for component in ("x", "y", "z"):
        idx = "index_%s_count" % component
        if len(set([_i[idx] for _i in setup["subdomains"]])) != 1:
            msg = ("Invalid boxfile. Unequal %s index count across "
                   "subdomains.") % component
            raise ValueError(msg)

    # Now generate the absolute indices for the whole domains.
    for component in ("x", "y", "z"):
        setup["boundaries_%s" % component] = (
            min([_i["boundaries_%s" % component][0]
                 for _i in setup["subdomains"]]),
            max([_i["boundaries_%s" %
                    component][1] for _i in setup["subdomains"]]))
        setup["physical_boundaries_%s" % component] = (
            min([_i["physical_boundaries_%s" % component][0] for
                 _i in setup["subdomains"]]),
            max([_i["physical_boundaries_%s" % component][1] for _i in
                 setup["subdomains"]]))

    return setup


def plot_hdf5_model(filename, plot_type="horizontal", *args, **kwargs):
    with h5py.File(filename, "r") as f:
        if plot_type == "horizontal":
            _plot_hdf5_model_horizontal(f=f, *args, **kwargs)
        elif plot_type == "vertical":
            _plot_hdf5_model_vertical(f=f, *args, **kwargs)
        else:
            raise NotImplementedError


def _plot_hdf5_model_vertical(f, component, output_filename, vmin=None,
                              vmax=None):
    import matplotlib.cm
    import matplotlib.pylab as plt

    data = xarray.DataArray(
        f["data"][component][:], [
            ("latitude", 90.0 - f["coordinate_0"][:]),
            ("longitude", f["coordinate_1"][:]),
            ("radius", f["coordinate_2"][:] / 1000.0)])

    plt.style.use('seaborn-pastel')

    plt.figure(figsize=(32, 18))

    plt.suptitle("Component %s - File %s" % (component, output_filename),
                 fontsize=20)

    count = 12
    lats = plt.linspace(data["latitude"].min(), data["latitude"].max(),
                        count)
    lngs = plt.linspace(data["longitude"].min(), data["longitude"].max(),
                        count)

    import lasif.colors
    my_colormap = lasif.colors.get_colormap(
        "tomo_full_scale_linear_lightness")

    # Overwrite colormap things if given.
    if vmin is not None and vmax is not None:
        min_val_plot = vmin
        max_val_plot = vmax
    else:
        mean = data.mean()
        max_diff = max(abs(mean - data.min()),
                       abs(data.max() - mean))
        min_val_plot = mean - max_diff
        max_val_plot = mean + max_diff
        # Plotting essentially constant models.
        min_delta = 0.001 * abs(max_val_plot)
        if (max_val_plot - min_val_plot) < min_delta:
            max_val_plot = max_val_plot + min_delta
            min_val_plot = min_val_plot - min_delta

    for _i in range(count):
        plt.subplot(4, count // 2, _i + 1)

        x, y = np.meshgrid(data.longitude, data.radius)

        plot_data = data.sel(latitude=lats[_i], method="nearest")
        plot_data = np.ma.masked_invalid(plot_data.data)

        # Plot.
        plt.pcolormesh(
            x, y, plot_data.T,
            cmap=my_colormap, vmin=min_val_plot, vmax=max_val_plot,
            shading="flat")

        # make a colorbar and title
        plt.colorbar()
        plt.title("@Latitude: " + str(lats[_i]))


    for _i in range(count):
        plt.subplot(4, count // 2, count + _i + 1)

        x, y = np.meshgrid(data.latitude, data.radius)

        plot_data = data.sel(longitude=lngs[_i], method="nearest")
        plot_data = np.ma.masked_invalid(plot_data.data)

        # Plot.
        plt.pcolormesh(
            x, y, plot_data.T,
            cmap=my_colormap, vmin=min_val_plot, vmax=max_val_plot,
            shading="flat")

        # make a colorbar and title
        plt.colorbar()
        plt.title("@Longitude: " + str(lngs[_i]))


    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output_filename, dpi=150)
    plt.close()




def _plot_hdf5_model_horizontal(f, component, output_filename,
                                vmin=None, vmax=None):
    import matplotlib.cm
    import matplotlib.pylab as plt

    data = xarray.DataArray(
        f["data"][component][:], [
            ("latitude", 90.0 - f["coordinate_0"][:]),
            ("longitude", f["coordinate_1"][:]),
            ("radius", f["coordinate_2"][:] / 1000.0)])

    plt.style.use('seaborn-pastel')

    from lasif.domain import RectangularSphericalSection
    domain = RectangularSphericalSection(**dict(f["_meta"]["domain"].attrs))

    plt.figure(figsize=(32, 18))

    depth_position_map = {
        50: (0, 0),
        100: (0, 1),
        150: (1, 0),
        250: (1, 1),
        400: (2, 0),
        600: (2, 1)
    }

    for depth, location in depth_position_map.items():
        ax = plt.subplot2grid((3, 5), location)
        radius = 6371.0 - depth

        # set up a map and colourmap
        m = domain.plot(ax=ax, resolution="c", skip_map_features=True)

        import lasif.colors
        my_colormap = lasif.colors.get_colormap(
                "tomo_full_scale_linear_lightness")

        from lasif import rotations

        x, y = np.meshgrid(data.longitude, data.latitude)

        x_shape = x.shape
        y_shape = y.shape

        lat_r, lon_r = rotations.rotate_lat_lon(
                y.ravel(), x.ravel(),
                domain.rotation_axis,
                domain.rotation_angle_in_degree)

        x, y = m(lon_r, lat_r)

        x.shape = x_shape
        y.shape = y_shape

        plot_data = data.sel(radius=radius, method="nearest")
        plot_data = np.ma.masked_invalid(plot_data.data)

        # Overwrite colormap things if given.
        if vmin is not None and vmax is not None:
            min_val_plot = vmin
            max_val_plot = vmax
        else:
            mean = plot_data.mean()
            max_diff = max(abs(mean - plot_data.min()),
                           abs(plot_data.max() - mean))
            min_val_plot = mean - max_diff
            max_val_plot = mean + max_diff
            # Plotting essentially constant models.
            min_delta = 0.001 * abs(max_val_plot)
            if (max_val_plot - min_val_plot) < min_delta:
                max_val_plot = max_val_plot + min_delta
                min_val_plot = min_val_plot - min_delta

        # Plot.
        im = m.pcolormesh(
                x, y, plot_data,
                cmap=my_colormap, vmin=min_val_plot, vmax=max_val_plot,
                shading="gouraud")

        # make a colorbar and title
        m.colorbar(im, "right", size="3%", pad='2%')
        plt.title(str(depth) + ' km')


    # Depth based statistics.
    plt.subplot2grid((3, 5), (0, 4), rowspan=3)
    plt.title("Depth statistics")
    mean = data.mean(axis=(0, 1))
    std = data.std(axis=(0, 1))
    _min = data.min(axis=(0, 1))
    _max = data.max(axis=(0, 1))

    plt.fill_betweenx(data.radius, mean - std, mean + std,
                      label="std", color="#FF3C83")
    plt.plot(mean, data.radius, label="mean", color="k", lw=2)
    plt.plot(_min, data.radius, color="grey", label="min")
    plt.plot(_max, data.radius, color="grey", label="max")
    plt.legend(loc="best")
    plt.xlabel("Value")
    plt.ylabel("Radius")

    plt.hlines(data.radius, plt.xlim()[0], plt.xlim()[1], color="0.8",
               zorder=-10, linewidth=0.5)

    # Roughness plots.
    plt.subplot2grid((3, 5), (0, 2))
    _d = np.abs(data.diff("latitude", n=1)).sum("latitude").data
    plt.title("Roughness in latitude direction, Total: %g" % _d.sum())
    plt.pcolormesh(data.longitude.data, data.radius.data,
                   _d.T, cmap=matplotlib.cm.viridis)
    try:
        plt.colorbar()
    except:
        pass
    plt.xlabel("Longitude")
    plt.ylabel("Radius")

    plt.subplot2grid((3, 5), (1, 2))
    _d = np.abs(data.diff("longitude", n=1)).sum("longitude").data
    plt.title("Roughness in longitude direction. Total: %g" % data.sum())
    plt.pcolormesh(data.latitude.data, data.radius.data, _d.T,
                   cmap=matplotlib.cm.viridis)
    try:
        plt.colorbar()
    except:
        pass
    plt.xlabel("Latitude")
    plt.ylabel("Radius")

    plt.subplot2grid((3, 5), (2, 2))
    _d = np.abs(data.diff("radius", n=1)).sum("radius").data
    plt.title("Roughness in radius direction. Total: %g" % _d.sum())
    plt.pcolormesh(data.longitude.data, data.latitude.data,
                   _d, cmap=matplotlib.cm.viridis)
    try:
        plt.colorbar()
    except:
        pass
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # L2
    plt.subplot2grid((3, 5), (0, 3))
    _d = (data ** 2).sum("latitude").data
    plt.title("L2 Norm in latitude direction, Total: %g" % _d.sum())
    plt.pcolormesh(data.longitude.data, data.radius.data,
                   _d.T, cmap=matplotlib.cm.viridis)
    try:
        plt.colorbar()
    except:
        pass
    plt.xlabel("Longitude")
    plt.ylabel("Radius")

    plt.subplot2grid((3, 5), (1, 3))
    _d = (data ** 2).sum("longitude").data
    plt.title("L2 Norm in longitude direction, Total: %g" % _d.sum())
    plt.pcolormesh(data.latitude.data, data.radius.data, _d.T,
                   cmap=matplotlib.cm.viridis)
    try:
        plt.colorbar()
    except:
        pass
    plt.xlabel("Latitude")
    plt.ylabel("Radius")

    plt.subplot2grid((3, 5), (2, 3))
    _d = (data ** 2).sum("radius").data
    plt.title("L2 Norm in radius direction, Total: %g" % _d.sum())
    plt.pcolormesh(data.longitude.data, data.latitude.data,
                   _d, cmap=matplotlib.cm.viridis)
    try:
        plt.colorbar()
    except:
        pass
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.suptitle("Component %s - File %s" % (component, output_filename),
                 fontsize=20)

    plt.tight_layout(rect=(0, 0, 1, 0.95))

    plt.savefig(output_filename, dpi=150)
    plt.close()


def taper_and_precondition_hdf5_model(
        input_filename, output_filename, taper_colatitude_offset_in_km,
        taper_colatitude_width_in_km, taper_longitude_offset_in_km,
        taper_longitude_width_in_km, taper_depth_offset_in_km,
        taper_depth_width_in_km, scaling_file):
    # Make a copy of the file and then modify in-place.
    assert not os.path.exists(output_filename), "File '%s' already exists." % \
        output_filename

    shutil.copy2(input_filename, output_filename)

    with h5py.File(output_filename, "r+") as f:
        _taper_and_precondition_hdf5_model(
            f=f,
            taper_colatitude_offset_in_km=taper_colatitude_offset_in_km,
            taper_colatitude_width_in_km=taper_colatitude_width_in_km,
            taper_longitude_offset_in_km=taper_longitude_offset_in_km,
            taper_longitude_width_in_km=taper_longitude_width_in_km,
            taper_depth_offset_in_km=taper_depth_offset_in_km,
            taper_depth_width_in_km=taper_depth_width_in_km,
            scaling_file=scaling_file)


def _taper_and_precondition_hdf5_model(f, taper_colatitude_offset_in_km,
                                       taper_colatitude_width_in_km,
                                       taper_longitude_offset_in_km,
                                       taper_longitude_width_in_km,
                                       taper_depth_offset_in_km,
                                       taper_depth_width_in_km,
                                       scaling_file):

    # Read the scaling file and make sure it plays nice with the gradient at
    # hand.
    with io.open(scaling_file, "rb") as fh:
        scaling = json.load(fh)
    np.testing.assert_allclose(scaling["radius"], f["coordinate_2"][:])
    scaling = np.array(scaling["weights"], dtype=np.float32)

    fac = 111.19492664455873
    colatitude_in_km = f["coordinate_0"][:] * fac
    longitude_in_km = f["coordinate_1"][:] * fac
    radius_in_km = f["coordinate_2"][:] / 1000.0

    # Convert into distance from either end.
    for _i in [colatitude_in_km, longitude_in_km]:
        _i[:] = np.fmin(_i - _i.min(), _i.max() - _i)
    # In the radial direction we only taper at the bottom.
    radius_in_km -= radius_in_km.min()

    # Apply the offsets
    colatitude_in_km -= taper_colatitude_offset_in_km
    longitude_in_km -= taper_longitude_offset_in_km
    radius_in_km -= taper_depth_offset_in_km

    # Apply the taper width
    colatitude_in_km /= taper_colatitude_width_in_km
    longitude_in_km /= taper_longitude_width_in_km
    radius_in_km /= taper_depth_width_in_km

    # Clip
    longitude_in_km = longitude_in_km.clip(min=0.0, max=1.0)
    colatitude_in_km = colatitude_in_km.clip(min=0.0, max=1.0)
    radius_in_km = radius_in_km.clip(min=0.0, max=1.0)

    # Apply Hanning taper. This finalizes the taper we have to multiply the
    # data with.
    for x in [longitude_in_km, colatitude_in_km, radius_in_km]:
        x[:] = 0.5 * (1.0 - np.cos(x * np.pi))

    # Apply the tapers.
    for name, data in f["data"].items():
        data = data[:]
        data *= colatitude_in_km[:, np.newaxis, np.newaxis]
        data *= longitude_in_km[np.newaxis, :, np.newaxis]
        data *= radius_in_km[np.newaxis, np.newaxis, :]
        # Apply the depth weighting.
        data *= scaling[np.newaxis, np.newaxis, :]
        f["data"][name][:] = data


def determine_depth_scaling(input_filename, output_filename, max_kernel_value):
    with h5py.File(input_filename, mode="r") as f:
        _determine_depth_scaling(f=f,
                                 output_filename=output_filename,
                                 max_kernel_value=max_kernel_value)


def _determine_depth_scaling(f, output_filename, max_kernel_value):
    all_scales = []

    for data in f["data"].values():
        data = data[:]
        # Zeros mess with everything - replace with the smallest
        # non-zero number!
        data[data == 0] = np.abs(data[data != 0]).min()

        # Damping factor - the higher the damping the lesser the effect of
        # the depth scaling.
        damp = 0.1

        m = np.max(np.abs(data))
        fac = np.zeros(data.shape[-1])
        for _i in range(len(fac)):
            fac[_i] = 1.0 / (damp * m + np.abs(data[:, :, _i]).max())

        all_scales.append(fac)

    s = np.sum(all_scales, axis=0)

    import scipy.signal
    # Smooth a tiny bit to avoid wild oscillations.
    w = scipy.signal.gaussian(5, 3)
    w /= w.sum()

    # Scale this for funsies.
    s /= s.min()

    # Avoid boundary effects.
    l = len(s)
    s = np.concatenate([np.ones_like(s) * s[0], s, np.ones_like(s) * s[-1]])

    smooth_s = np.convolve(s, w, mode="same")
    # Cut out the original segment.
    smooth_s = smooth_s[l:-l]
    s = s[l:-l]

    # Abuse ObsPy to taper a bit at both ends.
    smooth_s = obspy.Trace(data=smooth_s).taper(
        max_percentage=0.2, type="cosine", side="left").data.clip(min=1.0)

    # Get the max absolute value in depth for the vsv kernel.
    max_vsv = np.abs(f["data"]["vsv"][:]).max(axis=(0, 1))

    factor = max_kernel_value / (smooth_s * max_vsv).max()

    import matplotlib.pyplot as plt
    plt.style.use("ggplot")

    y = f["coordinate_2"][:] / 1000.0

    plt.subplot(141)
    m = max_vsv
    plt.plot(m, y)
    plt.xlim(-0.1 * m.ptp(), 1.1 * m.max())
    plt.ylim(y[0], y[-1])
    plt.title("max abs vsv")

    plt.subplot(142)
    plt.plot(s, y)
    plt.ylim(y[0], y[-1])
    plt.title("raw")

    plt.subplot(143)
    plt.plot(smooth_s, y)
    plt.xlim(0, smooth_s.max() * 1.5)
    plt.ylim(y[0], y[-1])
    plt.title("smoothed")

    plt.subplot(144)
    m = smooth_s * factor * max_vsv
    plt.plot(m, y)
    plt.xlim(-0.1 * max_kernel_value, 1.1 * max_kernel_value)
    plt.ylim(y[0], y[-1])
    plt.title("after")

    plt.suptitle("Factor: %s" % str(factor))

    output = {
        "radius": [float(i) for i in f["coordinate_2"][:]],
        "weights": [float(i) for i in smooth_s * factor]
    }

    with io.open(output_filename, "wb") as fh:
        json.dump(output, fh)

    plt.show()
