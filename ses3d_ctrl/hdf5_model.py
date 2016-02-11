#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import io
import math
import os
import struct

import h5py
import numpy as np
import xray

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

    m = RawSES3DModelHandler(
        directory=input_folder, domain=comm.project.domain,
        model_type="earth_model")

    f = h5py.File(output_filename)

    try:
        data_group = f.create_group("data")

        # We will also store A, C, and Q which we don't invert for but have to
        # take into account in any case.
        components = ["vp", "vsh", "vsv", "rho", "A", "C", "Q"]
        for c in components:
            m.parse_component(c)
            _d = xray.DataArray(
                np.require(m.parsed_components[c],
                           requirements=["C_CONTIGUOUS"]),
                coords=[90.0 - m.collocation_points_lats[::-1],
                        m.collocation_points_lngs,
                        (6371.0 - m.collocation_points_depth) * 1000.0],
                dims=["colatitude", "longitude", "radius_in_m"])

            # Write to HDF5 file.
            data_group[c] = np.require(_d.data, dtype=np.float32)
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
    data["rhoinv"] = 1.0 / f["data"]["rho"][:]
    data["mu"] = (f["data"]["vsh"][:] * 1000) ** 2 / data["rhoinv"]
    data["lambda"] = \
        (f["data"]["vp"][:] * 1000) ** 2 / data["rhoinv"] - data["mu"]
    data["A"] = (f["data"]["A"][:])
    data["B"] = (f["data"]["vsv"][:] * 1000) ** 2 / data["rhoinv"] - data["mu"]
    data["C"] = (f["data"]["C"][:])
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
