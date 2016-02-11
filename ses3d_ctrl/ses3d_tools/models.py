"""
Modified copy from the SES3D tools.

This version only deals with models that have a single domain and utilizes
the xray package.
"""
from __future__ import absolute_import

import os

import numpy as np
import xray

from .import colormaps as cm
from .import rotation as rot
from . import Q_models as q


class SES3DModel(object):
    """
    class for reading, writing, plotting and manipulating and ses3d model
    """
    def copy(self):
        """ Copy a model
        """
        res = SES3DModel()
        res.data = self.data.copy()
        res._filename = self._filename
        return res

    def __mul__(self, factor):
        """
        override left-multiplication of an ses3d model by a scalar factor
        """
        res = self.copy()
        res.data *= factor
        return res

    def __add__(self, other_model):
        """
        override addition of two ses3d models
        """
        res = self.copy()
        res.data += self.data
        return res

    def read(self, directory, filename, verbose=False, blockfile_folder=None):
        """
        Read a SES3D Model.
        """
        if not blockfile_folder:
            blockfile_folder = directory

        self._filename = os.path.abspath(os.path.join(directory, filename))

        # Read blockfiles.
        blocks = {}
        for var, f_name in zip(["x", "y", "z"],
                               ["block_x", "block_y", "block_z"]):
            blocks[var] = np.loadtxt(os.path.join(blockfile_folder, f_name),
                           dtype=np.float32)
            assert blocks[var][0] == 1, "Only one subvolume allowed."

            # Also make sure it contains exactly as many coordinates as
            # expected.
            assert len(blocks[var][2:]) == blocks[var][1]
            blocks[var] = blocks[var][2:]

            # Also the latest values is never used! This is consitent with
            # the add_perturbations function and all the projection functions
            # in SES3D.
            blocks[var] = blocks[var][:-1]

        radius = blocks["z"]
        longitude = blocks["y"]
        latitude = 90.0 - blocks["x"]

        with open(os.path.join(directory, filename)) as fh:
            # Seems to be about the fastest way to read possibly NaN ASCII
            # floating point values.
            data = np.array(fh.read().strip().split('\n'),
                            dtype=np.float32)[2:]
            data = np.reshape(data, (len(latitude), len(longitude),
                                     len(radius)))

        # Convert to xray data array.
        self.data = xray.DataArray(data, [("latitude", latitude),
                                          ("longitude", longitude),
                                          ("radius", radius)])

    def write(self, directory, filename, verbose=False):
        """
        write ses3d model to a file

        write(self,directory,filename,verbose=False):
        """
        with open(os.path.join(directory, filename), 'w') as fh:
            fh.write("1\n")

            nx = self.data.latitude.shape[0]
            ny = self.data.longitude.shape[0]
            nz = self.data.radius.shape[0]
            fh.write(str(nx * ny * nz) + '\n')

            for idx in xrange(nx):
                for idy in xrange(ny):
                    for idz in xrange(nz):
                        fh.write(str(self.data.data[idx, idy, idz]) + '\n')

    def norm(self):
        """
        # Compute the L2 norm.
        """
        N = 0.0

        # Loop over subvolumes.

        for n in np.arange(self.nsubvol):

            # Size of the array.
            nx = len(self.m[n].lat) - 1
            ny = len(self.m[n].lon) - 1
            nz = len(self.m[n].r) - 1

            # Compute volume elements.
            dV = np.zeros(np.shape(self.m[n].v))
            theta = (90.0 - self.m[n].lat) * np.pi / 180.0

            dr = self.m[n].r[1] - self.m[n].r[0]
            dphi = (self.m[n].lon[1] - self.m[n].lon[0]) * np.pi / 180.0
            dtheta = theta[1] - theta[0]

            for idx in np.arange(nx):
                for idy in np.arange(ny):
                    for idz in np.arange(nz):
                        dV[idx, idy, idz] = theta[idx] * (self.m[n].r[idz])**2

            dV = dr * dtheta * dphi * dV

            # Integrate.
            N += np.sum(dV * (self.m[n].v)**2)

        # Finish.
        return np.sqrt(N)

    def clip_percentile(self, percentile):
        """
        Clip the upper percentiles of the model. Particularly useful to
        remove the singularities in sensitivity kernels.
        """
        # Loop over subvolumes to find the percentile.
        percentile_list = []

        for n in np.arange(self.nsubvol):
            percentile_list.append(
                np.percentile(np.abs(self.m[n].v), percentile))

        percent = np.max(percentile_list)

        # Clip the values above the percentile.
        for n in np.arange(self.nsubvol):
            idx = np.nonzero(np.greater(np.abs(self.m[n].v), percent))
            self.m[n].v[idx] = np.sign(self.m[n].v[idx]) * percent

    def smooth_horizontal(self, sigma, filter_type='gauss'):
        """
        smooth_horizontal(self,sigma,filter='gauss')

        Experimental function for smoothing in horizontal directions.

        filter_type: gauss (Gaussian smoothing), neighbour (average over
            neighbouring cells)
        sigma: filter width (when filter_type='gauss') or iterations (when
            filter_type='neighbour')

        WARNING: Currently, the smoothing only works within each subvolume.
        The problem of smoothing across subvolumes without having excessive
        algorithmic complexity and with fast compute times, awaits
        resolution ... .
        """
        # Smoothing by averaging over neighbouring cells.
        from scipy.ndimage.filters import gaussian_filter

        width_latitude = 120
        width_longitude = 120
        width_radius = 30

        # Convert the width in samples.
        sigma = (
            width_latitude / 111.0 / abs(np.diff(self.data.latitude.data)[0]),
            width_longitude / 111.0 / abs(np.diff(
                self.data.longitude.data)[0]),
            width_radius / abs(np.diff(self.data.radius.data)[0]))

        self.data.data = gaussian_filter(self.data.data, sigma=sigma,
                                         mode="nearest")
        return

        if filter_type == 'neighbour':

            for _ in range(int(sigma)):
                filtered_data = self.data.copy()
                for i in np.arange(1, self.data.latitude.shape[0] - 1):
                    for j in np.arange(1, self.data.longitude.shape[0] - 1):
                        filtered_data.data[i, j, :] = (
                          self.data.data[i, j, :] +
                          self.data.data[i + 1, j, :] +
                          self.data.data[i - 1, j, :] +
                          self.data.data[i, j + 1, :] +
                          self.data.data[i, j - 1, :]) / 5.0
                self.data = filtered_data
        else:
            raise NotImplementedError

        return

        # Loop over subvolumes.
        for n in np.arange(self.nsubvol):

            v_filtered = self.m[n].v

            # Size of the array.
            nx = len(self.m[n].lat) - 1
            ny = len(self.m[n].lon) - 1
            nz = len(self.m[n].r) - 1

            # Gaussian smoothing.

            if filter_type == 'gauss':

                # Estimate element width.
                r = np.mean(self.m[n].r)
                dx = r * np.pi * (self.m[n].lat[0] - self.m[n].lat[1]) / 180.0

                # Colat and lon fields for the small Gaussian.
                dn = 3 * np.ceil(sigma / dx)

                nx_min = np.round(float(nx) / 2.0) - dn
                nx_max = np.round(float(nx) / 2.0) + dn

                ny_min = np.round(float(ny) / 2.0) - dn
                ny_max = np.round(float(ny) / 2.0) + dn

                lon, colat = np.meshgrid(
                    self.m[n].lon[ny_min:ny_max],
                    90.0 - self.m[n].lat[nx_min:nx_max])
                colat = np.pi * colat / 180.0
                lon = np.pi * lon / 180.0

                # Volume element.
                dy = r * np.pi * \
                    np.sin(colat) * \
                    (self.m[n].lon[1] - self.m[n].lon[0]) / 180.0
                dV = dx * dy

                # Unit vector field.
                x = np.cos(lon) * np.sin(colat)
                y = np.sin(lon) * np.sin(colat)
                z = np.cos(colat)

                # Make a Gaussian centred in the middle of the grid.
                i = np.round(float(nx) / 2.0) - 1
                j = np.round(float(ny) / 2.0) - 1

                colat_i = np.pi * (90.0 - self.m[n].lat[i]) / 180.0
                lon_j = np.pi * self.m[n].lon[j] / 180.0

                x_i = np.cos(lon_j) * np.sin(colat_i)
                y_j = np.sin(lon_j) * np.sin(colat_i)
                z_k = np.cos(colat_i)

                # Compute the Gaussian.
                G = x * x_i + y * y_j + z * z_k
                G = G / np.max(np.abs(G))
                G = r * np.arccos(G)
                G = np.exp(-0.5 * G**2 / sigma**2) / (2.0 * np.pi * sigma**2)

                # Move the Gaussian across the field.

                for i in np.arange(dn + 1, nx - dn - 1):
                    for j in np.arange(dn + 1, ny - dn - 1):
                        for k in np.arange(nz):

                            v_filtered[i, j, k] = np.sum(
                                self.m[n].v[i - dn:i + dn, j - dn:j + dn, k]
                                * G * dV)

            # Smoothing by averaging over neighbouring cells.

            elif filter_type == 'neighbour':

                for iteration in np.arange(int(sigma)):  # NOQA
                    for i in np.arange(1, nx - 1):
                        for j in np.arange(1, ny - 1):

                            v_filtered[i, j, :] = (
                                self.m[n].v[i, j, :] +
                                self.m[n].v[i + 1, j, :] +
                                self.m[n].v[i - 1, j, :] +
                                self.m[n].v[i, j + 1, :] +
                                self.m[n].v[i, j - 1, :]) / 5.0

                self.m[n].v = v_filtered

    def smooth_horizontal_adaptive(self, sigma):
        """
        Apply horizontal smoothing with adaptive smoothing length.
        """
        # Find maximum smoothing length.

        sigma_max = []

        for n in np.arange(self.nsubvol):

            sigma_max.append(np.max(sigma.m[n].v))

        # Loop over subvolumes.

        for n in np.arange(self.nsubvol):

            # Size of the array.
            nx = len(self.m[n].lat) - 1
            ny = len(self.m[n].lon) - 1
            nz = len(self.m[n].r) - 1

            # Estimate element width.
            r = np.mean(self.m[n].r)
            dx = r * np.pi * (self.m[n].lat[0] - self.m[n].lat[1]) / 180.0

            # Colat and lon fields for the small Gaussian.
            dn = 2 * np.round(sigma_max[n] / dx)

            nx_min = np.round(float(nx) / 2.0) - dn
            nx_max = np.round(float(nx) / 2.0) + dn

            ny_min = np.round(float(ny) / 2.0) - dn
            ny_max = np.round(float(ny) / 2.0) + dn

            lon, colat = np.meshgrid(
                self.m[n].lon[ny_min:ny_max],
                90.0 - self.m[n].lat[nx_min:nx_max], dtype=float)
            colat = np.pi * colat / 180.0
            lon = np.pi * lon / 180.0

            # Volume element.
            dy = r * np.pi * \
                np.sin(colat) * (self.m[n].lon[1] - self.m[n].lon[0]) / 180.0
            dV = dx * dy

            # Unit vector field.
            x = np.cos(lon) * np.sin(colat)
            y = np.sin(lon) * np.sin(colat)
            z = np.cos(colat)

            # Make a Gaussian centred in the middle of the grid.
            i = np.round(float(nx) / 2.0) - 1
            j = np.round(float(ny) / 2.0) - 1

            colat_i = np.pi * (90.0 - self.m[n].lat[i]) / 180.0
            lon_j = np.pi * self.m[n].lon[j] / 180.0

            x_i = np.cos(lon_j) * np.sin(colat_i)
            y_j = np.sin(lon_j) * np.sin(colat_i)
            z_k = np.cos(colat_i)

            # Distance from the central point.
            G = x * x_i + y * y_j + z * z_k
            G = G / np.max(np.abs(G))
            G = r * np.arccos(G)

            # Move the Gaussian across the field.
            v_filtered = self.m[n].v

            for i in np.arange(dn + 1, nx - dn - 1):
                for j in np.arange(dn + 1, ny - dn - 1):
                    for k in np.arange(nz):

                        # Compute the actual Gaussian.
                        s = sigma.m[n].v[i, j, k]

                        if (s > 0):

                            GG = np.exp(-0.5 * G**2 / s**2) / \
                                (2.0 * np.pi * s**2)

                            # Apply filter.
                            v_filtered[i, j, k] = np.sum(
                                self.m[n].v[i - dn:i + dn, j - dn:j + dn, k] *
                                GG * dV)

            self.m[n].v = v_filtered

    def ref2relax(self, qmodel='cem', nrelax=3):
        """
        ref2relax(qmodel='cem', nrelax=3)

        Compute relaxed velocities from velocities at 1 s reference period.

        Assuming that the current velocity model is given at the reference
        period 1 s, ref2relax computes the relaxed velocities. They may then
        be written to a file.

        For this conversion, the relaxation parameters from the /INPUT/relax
        file are taken.

        Currently implemented Q models (qmodel): cem, prem, ql6 .

        nrelax is the number of relaxation mechnisms.
        """
        # Read the relaxation parameters from the relax file.

        tau_p = np.zeros(nrelax)
        D_p = np.zeros(nrelax)

        fid = open('../INPUT/relax', 'r')

        fid.readline()

        for n in range(nrelax):
            tau_p[n] = float(fid.readline().strip())

        fid.readline()

        for n in range(nrelax):
            D_p[n] = float(fid.readline().strip())

        fid.close()

        # Loop over subvolumes.

        for k in np.arange(self.nsubvol):

            # nx = len(self.m[k].lat) - 1
            # ny = len(self.m[k].lon) - 1
            nz = len(self.m[k].r) - 1

            # Loop over radius within the subvolume.
            for idz in np.arange(nz):

                # Compute Q.
                if qmodel == 'cem':
                    Q = q.q_cem(self.m[k].r[idz])
                elif qmodel == 'ql6':
                    Q = q.q_ql6(self.m[k].r[idz])
                elif qmodel == 'prem':
                    Q = q.q_prem(self.m[k].r[idz])

                # Compute A and B for the reference period of 1 s.

                A = 1.0
                B = 0.0
                w = 2.0 * np.pi

                tau = 1.0 / Q

                for n in range(nrelax):
                    A += tau * \
                        D_p[n] * (w**2) * (tau_p[n]**2) / \
                        (1.0 + (w**2) * (tau_p[n]**2))
                    B += tau * D_p[n] * w * tau_p[n] / \
                        (1.0 + (w**2) * (tau_p[n]**2))

                conversion_factor = (A + np.sqrt(A**2 + B**2)) / (A**2 + B**2)
                conversion_factor = np.sqrt(0.5 * conversion_factor)

                # Correct velocities.

                self.m[k].v[:, :, idz] = conversion_factor * \
                    self.m[k].v[:, :, idz]

    def convert_to_vtk(self, directory, filename, verbose=False):
        """
        convert ses3d model to vtk format for plotting with Paraview, VisIt,
        ... .

        convert_to_vtk(self,directory,filename,verbose=False):
        """
        # preparatory steps

        nx = np.zeros(self.nsubvol, dtype=int)
        ny = np.zeros(self.nsubvol, dtype=int)
        nz = np.zeros(self.nsubvol, dtype=int)
        N = 0

        for n in np.arange(self.nsubvol):
            nx[n] = len(self.m[n].lat)
            ny[n] = len(self.m[n].lon)
            nz[n] = len(self.m[n].r)
            N = N + nx[n] * ny[n] * nz[n]

        # open file and write header

        fid = open(os.path.join(directory, filename), 'w')

        if verbose is True:
            print 'write to file %s' % os.path.join(directory, filename)

        fid.write('# vtk DataFile Version 3.0\n')
        fid.write('vtk output\n')
        fid.write('ASCII\n')
        fid.write('DATASET UNSTRUCTURED_GRID\n')

        # write grid points

        fid.write('POINTS ' + str(N) + ' float\n')

        for n in np.arange(self.nsubvol):

            if verbose is True:
                print 'writing grid points for subvolume ' + str(n)

            for i in np.arange(nx[n]):
                for j in np.arange(ny[n]):
                    for k in np.arange(nz[n]):

                        theta = 90.0 - self.m[n].lat[i]
                        phi = self.m[n].lon[j]

                        # rotate coordinate system
                        if self.phi != 0.0:
                            theta, phi = rot.rotate_coordinates(
                                self.n, -self.phi, theta, phi)

                            # transform to cartesian coordinates and write
                            # to file
                            theta = theta * np.pi / 180.0
                            phi = phi * np.pi / 180.0

                            r = self.m[n].r[k]
                            x = r * np.sin(theta) * np.cos(phi)
                            y = r * np.sin(theta) * np.sin(phi)
                            z = r * np.cos(theta)

                            fid.write(
                                str(x) + ' ' + str(y) + ' ' + str(z) + '\n')

        # write connectivity

        n_cells = 0

        for n in np.arange(self.nsubvol):
            n_cells = n_cells + (nx[n] - 1) * (ny[n] - 1) * (nz[n] - 1)

        fid.write('\n')
        fid.write('CELLS ' + str(n_cells) + ' ' + str(9 * n_cells) + '\n')

        count = 0

        for n in np.arange(self.nsubvol):

            if verbose is True:
                print 'writing conectivity for subvolume ' + str(n)

            for i in np.arange(1, nx[n]):
                for j in np.arange(1, ny[n]):
                    for k in np.arange(1, nz[n]):

                        a = count + k + \
                            (j - 1) * nz[n] + (i - 1) * ny[n] * nz[n] - 1
                        b = count + k + \
                            (j - 1) * nz[n] + (i - 1) * ny[n] * nz[n]
                        c = count + k + \
                            (j) * nz[n] + (i - 1) * ny[n] * nz[n] - 1
                        d = count + k + (j) * nz[n] + (i - 1) * ny[n] * nz[n]
                        e = count + k + \
                            (j - 1) * nz[n] + (i) * ny[n] * nz[n] - 1
                        f = count + k + (j - 1) * nz[n] + (i) * ny[n] * nz[n]
                        g = count + k + (j) * nz[n] + (i) * ny[n] * nz[n] - 1
                        h = count + k + (j) * nz[n] + (i) * ny[n] * nz[n]

                        fid.write('8 ' + str(a) + ' ' + str(b) + ' ' + str(c)
                                  + ' ' + str(d) + ' ' + str(e) + ' ' + str(f)
                                  + ' ' + str(g) + ' ' + str(h) + '\n')

            count = count + nx[n] * ny[n] * nz[n]

        # write cell types

        fid.write('\n')
        fid.write('CELL_TYPES ' + str(n_cells) + '\n')

        for n in np.arange(self.nsubvol):

            if verbose is True:
                print 'writing cell types for subvolume ' + str(n)

            for i in np.arange(nx[n] - 1):
                for j in np.arange(ny[n] - 1):
                    for k in np.arange(nz[n] - 1):

                        fid.write('11\n')

        # write data

        fid.write('\n')
        fid.write('POINT_DATA ' + str(N) + '\n')
        fid.write('SCALARS scalars float\n')
        fid.write('LOOKUP_TABLE mytable\n')

        for n in np.arange(self.nsubvol):

            if verbose is True:
                print 'writing data for subvolume ' + str(n)

            idx = np.arange(nx[n])
            idx[nx[n] - 1] = nx[n] - 2

            idy = np.arange(ny[n])
            idy[ny[n] - 1] = ny[n] - 2

            idz = np.arange(nz[n])
            idz[nz[n] - 1] = nz[n] - 2

            for i in idx:
                for j in idy:
                    for k in idz:

                        fid.write(str(self.m[n].v[i, j, k]) + '\n')

        # clean up

        fid.close()

    def plot_slice(self, depth, min_val_plot=None, max_val_plot=None,
                   colormap='tomo', res='i', save_under=None, verbose=False,
                   lasif_folder=None, vmin=None, vmax=None):
        """
        plot horizontal slices through an ses3d model

        plot_slice(self,depth,colormap='tomo',res='i',save_under=None,
        verbose=False)

        depth=depth in km of the slice
        colormap='tomo','mono'
        res=resolution of the map, admissible values are: c, l, i, h f
        save_under=save figure as *.png with the filename "save_under".
        Prevents plotting of the slice.
        """
        import matplotlib.cm
        from matplotlib.colors import LogNorm
        import matplotlib.pylab as plt

        plt.style.use('seaborn-pastel')


        if not lasif_folder:
            raise NotImplementedError

        from lasif.scripts.lasif_cli import _find_project_comm
        comm = _find_project_comm(lasif_folder, read_only_caches=False)

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
            m = comm.project.domain.plot(ax=ax)

            import lasif.colors
            my_colormap = lasif.colors.get_colormap(
                "tomo_full_scale_linear_lightness")

            from lasif import rotations

            x, y = np.meshgrid(self.data.longitude, self.data.latitude)

            x_shape = x.shape
            y_shape = y.shape

            lat_r, lon_r = rotations.rotate_lat_lon(
                y.ravel(), x.ravel(),
                comm.project.domain.rotation_axis,
                comm.project.domain.rotation_angle_in_degree)

            x, y = m(lon_r, lat_r)

            x.shape = x_shape
            y.shape = y_shape

            plot_data = self.data.sel(radius=radius, method="nearest")
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
                min_delta = 0.01 * abs(max_val_plot)
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
        mean = self.data.mean(axis=(0, 1))
        std = self.data.std(axis=(0, 1))
        _min = self.data.min(axis=(0, 1))
        _max = self.data.max(axis=(0, 1))

        plt.fill_betweenx(self.data.radius, mean - std, mean + std,
                          label="std", color="#FF3C83")
        plt.plot(mean, self.data.radius, label="mean", color="k", lw=2)
        plt.plot(_min, self.data.radius, color="grey", label="min")
        plt.plot(_max, self.data.radius, color="grey", label="max")
        plt.legend(loc="best")
        plt.xlabel("Value")
        plt.ylabel("Radius")

        # Roughness plots.
        plt.subplot2grid((3, 5), (0, 2))
        data = np.abs(self.data.diff("latitude", n=1)).sum("latitude").data
        plt.title("Roughness in latitude direction, Total: %g" % data.sum())
        plt.pcolormesh(self.data.longitude.data, self.data.radius.data,
                       data.T, cmap=matplotlib.cm.viridis,
                       norm=LogNorm(data.max() * 1E-2, data.max()))
        plt.colorbar()
        plt.xlabel("Longitude")
        plt.ylabel("Radius")

        plt.subplot2grid((3, 5), (1, 2))
        data = np.abs(self.data.diff("longitude", n=1)).sum("longitude").data
        plt.title("Roughness in longitude direction. Total: %g" % data.sum())
        plt.pcolormesh(self.data.latitude.data, self.data.radius.data, data.T,
                       cmap=matplotlib.cm.viridis,
                       norm=LogNorm(data.max() * 1E-2, data.max()))
        plt.colorbar()
        plt.xlabel("Latitude")
        plt.ylabel("Radius")

        plt.subplot2grid((3, 5), (2, 2))
        data = np.abs(self.data.diff("radius", n=1)).sum("radius").data
        plt.title("Roughness in radius direction. Total: %g" % data.sum())
        plt.pcolormesh(self.data.longitude.data, self.data.latitude.data,
                       data, cmap=matplotlib.cm.viridis)
        plt.colorbar()
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        # L2
        plt.subplot2grid((3, 5), (0, 3))
        data = (self.data ** 2).sum("latitude").data
        plt.title("L2 Norm in latitude direction, Total: %g" % data.sum())
        plt.pcolormesh(self.data.longitude.data, self.data.radius.data,
                       data.T, cmap=matplotlib.cm.viridis)
        plt.colorbar()
        plt.xlabel("Longitude")
        plt.ylabel("Radius")

        plt.subplot2grid((3, 5), (1, 3))
        data = (self.data ** 2).sum("longitude").data
        plt.title("L2 Norm in longitude direction, Total: %g" % data.sum())
        plt.pcolormesh(self.data.latitude.data, self.data.radius.data, data.T,
                       cmap=matplotlib.cm.viridis)
        plt.colorbar()
        plt.xlabel("Latitude")
        plt.ylabel("Radius")

        plt.subplot2grid((3, 5), (2, 3))
        data = (self.data ** 2).sum("radius").data
        plt.title("L2 Norm in radius direction, Total: %g" % data.sum())
        plt.pcolormesh(self.data.longitude.data, self.data.latitude.data,
                       data, cmap=matplotlib.cm.viridis)
        plt.colorbar()
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        plt.suptitle("File %s" % self._filename, fontsize=20)

        plt.tight_layout(rect=(0, 0, 1, 0.95))

        # save image if wanted
        if save_under is None:
            plt.show()
        else:
            plt.savefig(save_under, dpi=150)
            plt.close()

    def plot_threshold(self, val, min_val_plot, max_val_plot, colormap='tomo',
                       verbose=False):
        """
        plot depth to a certain threshold value 'val' in an ses3d model


        plot_threshold(val,min_val_plot,max_val_plot,colormap='tomo',
                        verbose=False):
        val=threshold value
        min_val_plot, max_val_plot=minimum and maximum values of the colour
        scale colormap='tomo','mono'
        """
        import matplotlib.pylab as plt
        from mpl_toolkits.basemap import Basemap
        # set up a map and colourmap

        if self.global_regional == 'regional':
            m = Basemap(projection='merc', llcrnrlat=self.lat_min,
                        urcrnrlat=self.lat_max, llcrnrlon=self.lon_min,
                        urcrnrlon=self.lon_max, lat_ts=20, resolution="m")
            m.drawparallels(
                np.arange(self.lat_min, self.lat_max, self.d_lon),
                labels=[1, 0, 0, 1])
            m.drawmeridians(
                np.arange(self.lon_min, self.lon_max, self.d_lat),
                labels=[1, 0, 0, 1])
        elif self.global_regional == 'global':
            m = Basemap(projection='ortho', lon_0=self.lon_centre,
                        lat_0=self.lat_centre, resolution="l")
            m.drawparallels(np.arange(-80.0, 80.0, 10.0), labels=[1, 0, 0, 1])
            m.drawmeridians(
                np.arange(-170.0, 170.0, 10.0), labels=[1, 0, 0, 1])

        m.drawcoastlines()
        m.drawcountries()

        m.drawmapboundary(fill_color=[1.0, 1.0, 1.0])

        if colormap == 'tomo':
            my_colormap = cm.make_colormap(
                {0.0: [0.1, 0.0, 0.0], 0.2: [0.8, 0.0, 0.0],
                 0.3: [1.0, 0.7, 0.0], 0.48: [0.92, 0.92, 0.92],
                 0.5: [0.92, 0.92, 0.92], 0.52: [0.92, 0.92, 0.92],
                 0.7: [0.0, 0.6, 0.7], 0.8: [0.0, 0.0, 0.8],
                 1.0: [0.0, 0.0, 0.1]})
        elif colormap == 'mono':
            my_colormap = cm.make_colormap(
                {0.0: [1.0, 1.0, 1.0], 0.15: [1.0, 1.0, 1.0],
                 0.85: [0.0, 0.0, 0.0], 1.0: [0.0, 0.0, 0.0]})

        # loop over subvolumes
        for k in np.arange(self.nsubvol):

            depth = np.zeros(np.shape(self.m[k].v[:, :, 0]))

            nx = len(self.m[k].lat)
            ny = len(self.m[k].lon)

            # find depth

            r = self.m[k].r
            r = 0.5 * (r[0:len(r) - 1] + r[1:len(r)])

            for idx in np.arange(nx - 1):
                for idy in np.arange(ny - 1):

                    n = self.m[k].v[idx, idy, :] >= val
                    depth[idx, idy] = 6371.0 - np.max(r[n])

            # rotate coordinate system if necessary

            lon, lat = np.meshgrid(self.m[k].lon[0:ny], self.m[k].lat[0:nx])

            if self.phi != 0.0:

                lat_rot = np.zeros(np.shape(lon), dtype=float)
                lon_rot = np.zeros(np.shape(lat), dtype=float)

                for idx in np.arange(nx):
                    for idy in np.arange(ny):

                        colat = 90.0 - lat[idx, idy]

                        lat_rot[idx, idy], lon_rot[idx, idy] = \
                            rot.rotate_coordinates(self.n, -self.phi, colat,
                                                   lon[idx, idy])
                        lat_rot[idx, idy] = 90.0 - lat_rot[idx, idy]

                lon = lon_rot
                lat = lat_rot

                # convert to map coordinates and plot

            x, y = m(lon, lat)
            im = m.pcolor(x, y, depth, cmap=my_colormap, vmin=min_val_plot,
                          vmax=max_val_plot)

        m.colorbar(im, "right", size="3%", pad='2%')
        plt.title('depth to ' + str(val) + ' km/s [km]')
        plt.show()
