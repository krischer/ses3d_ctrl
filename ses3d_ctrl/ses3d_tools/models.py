"""
Modiefied copy from the SES3D tools.
"""
from __future__ import absolute_import

import os

import numpy as np

from .import colormaps as cm
from .import rotation as rot
from . import Q_models as q


class SES3DSubModel(object):
    """
    class defining an ses3d submodel
    """
    def __init__(self):

        # coordinate lines
        self.lat = np.zeros(1)
        self.lon = np.zeros(1)
        self.r = np.zeros(1)

        # rotated coordinate lines
        self.lat_rot = np.zeros(1)
        self.lon_rot = np.zeros(1)

        # field
        self.v = np.zeros((1, 1, 1))


class SES3DModel(object):
    """
    class for reading, writing, plotting and manipulating and ses3d model
    """
    def __init__(self):
        """
        initiate the SES3DModel class

        initiate list of submodels and read rotation_parameters.txt
        """
        self.nsubvol = 0
        self.lat_min = 0.0
        self.lat_max = 0.0
        self.lon_min = 0.0
        self.lon_max = 0.0
        self.lat_centre = 0.0
        self.lon_centre = 0.0
        self.global_regional = "global"

        self.m = []

        # read rotation parameters
        # Currently disabled as not needed for ses3d_ctrl.
        # fid = open('rotation_parameters.txt', 'r')
        # fid.readline()
        # self.phi = float(fid.readline().strip())
        # fid.readline()
        # line = fid.readline().strip().split(' ')
        # self.n = np.array([float(line[0]), float(line[1]), float(line[2])])
        # fid.close()
        self.phi = 0
        self.n = np.ones(3)

    #########################################################################
    # copy models
    #########################################################################

    def copy(self):
        """ Copy a model
        """

        res = SES3DModel()

        res.nsubvol = self.nsubvol

        res.lat_min = self.lat_min
        res.lat_max = self.lat_max
        res.lon_min = self.lon_min
        res.lon_max = self.lon_max
        res.lat_centre = self.lat_centre
        res.lon_centre = self.lon_centre
        res.phi = self.phi

        res.n = self.n

        res.global_regional = self.global_regional
        res.d_lon = self.d_lon
        res.d_lat = self.d_lat

        for k in np.arange(self.nsubvol):

            subvol = SES3DSubModel()

            subvol.lat = self.m[k].lat
            subvol.lon = self.m[k].lon
            subvol.r = self.m[k].r

            subvol.lat_rot = self.m[k].lat_rot
            subvol.lon_rot = self.m[k].lon_rot

            subvol.v = self.m[k].v

            res.m.append(subvol)

        return res

    def __rmul__(self, factor):
        """ override left-multiplication of an ses3d model by a scalar factor
        """

        res = SES3DModel()

        res.nsubvol = self.nsubvol

        res.lat_min = self.lat_min
        res.lat_max = self.lat_max
        res.lon_min = self.lon_min
        res.lon_max = self.lon_max
        res.lat_centre = self.lat_centre
        res.lon_centre = self.lon_centre
        res.phi = self.phi

        res.n = self.n

        res.global_regional = self.global_regional
        res.d_lon = self.d_lon
        res.d_lat = self.d_lat

        for k in np.arange(self.nsubvol):

            subvol = SES3DSubModel()

            subvol.lat = self.m[k].lat
            subvol.lon = self.m[k].lon
            subvol.r = self.m[k].r

            subvol.lat_rot = self.m[k].lat_rot
            subvol.lon_rot = self.m[k].lon_rot

            subvol.v = factor * self.m[k].v

            res.m.append(subvol)

        return res

    def __add__(self, other_model):
        """
        override addition of two ses3d models
        """
        res = SES3DModel()

        res.nsubvol = self.nsubvol
        res.lat_min = self.lat_min
        res.lat_max = self.lat_max
        res.lon_min = self.lon_min
        res.lon_max = self.lon_max
        res.lat_centre = self.lat_centre
        res.lon_centre = self.lon_centre
        res.phi = self.phi
        res.n = self.n
        res.global_regional = self.global_regional
        res.d_lon = self.d_lon
        res.d_lat = self.d_lat

        for k in np.arange(self.nsubvol):

            subvol = SES3DSubModel()

            subvol.lat = self.m[k].lat
            subvol.lon = self.m[k].lon
            subvol.r = self.m[k].r

            subvol.lat_rot = self.m[k].lat_rot
            subvol.lon_rot = self.m[k].lon_rot

            subvol.v = self.m[k].v + other_model.m[k].v

            res.m.append(subvol)

        return res

    def read(self, directory, filename, verbose=False):
        """ read an ses3d model from a file

        read(self,directory,filename,verbose=False):
        """
        # read block files
        fid_x = open(os.path.join(directory, 'block_x'), 'r')
        fid_y = open(os.path.join(directory, 'block_y'), 'r')
        fid_z = open(os.path.join(directory, 'block_z'), 'r')

        if verbose is True:
            print "read block files:"
            print "\t%s" % os.path.join(directory, 'block_x')
            print "\t%s" % os.path.join(directory, 'block_y')
            print "\t%s" % os.path.join(directory, 'block_z')

        dx = np.array(fid_x.read().strip().split('\n'), dtype=float)
        dy = np.array(fid_y.read().strip().split('\n'), dtype=float)
        dz = np.array(fid_z.read().strip().split('\n'), dtype=float)

        fid_x.close()
        fid_y.close()
        fid_z.close()

        # read coordinate lines

        self.nsubvol = int(dx[0])

        if verbose is True:
            print 'number of subvolumes: ' + str(self.nsubvol)

        idx = np.zeros(self.nsubvol, dtype=int) + 1
        idy = np.zeros(self.nsubvol, dtype=int) + 1
        idz = np.zeros(self.nsubvol, dtype=int) + 1

        for k in np.arange(1, self.nsubvol, dtype=int):
            idx[k] = int(dx[idx[k - 1]]) + idx[k - 1] + 1
            idy[k] = int(dy[idy[k - 1]]) + idy[k - 1] + 1
            idz[k] = int(dz[idz[k - 1]]) + idz[k - 1] + 1

        for k in np.arange(self.nsubvol, dtype=int):
            subvol = SES3DSubModel()
            subvol.lat = 90.0 - dx[(idx[k] + 1):(idx[k] + 1 + dx[idx[k]])]
            subvol.lon = dy[(idy[k] + 1):(idy[k] + 1 + dy[idy[k]])]
            subvol.r = dz[(idz[k] + 1):(idz[k] + 1 + dz[idz[k]])]
            self.m.append(subvol)

        # compute rotated version of the coordinate lines

        if self.phi != 0.0:

            for k in np.arange(self.nsubvol, dtype=int):

                nx = len(self.m[k].lat)
                ny = len(self.m[k].lon)

                self.m[k].lat_rot = np.zeros([nx, ny])
                self.m[k].lon_rot = np.zeros([nx, ny])

                for idx in np.arange(nx):
                    for idy in np.arange(ny):
                        self.m[k].lat_rot[idx, idy], \
                            self.m[k].lon_rot[idx, idy] = \
                            rot.rotate_coordinates(self.n, -self.phi,
                                                   90.0 - self.m[k].lat[idx],
                                                   self.m[k].lon[idy])
                        self.m[k].lat_rot[idx, idy] = 90.0 - \
                            self.m[k].lat_rot[idx, idy]

        else:

            for k in np.arange(self.nsubvol, dtype=int):

                self.m[k].lat_rot, self.m[k].lon_rot = np.meshgrid(
                    self.m[k].lat, self.m[k].lon)
                self.m[k].lat_rot = self.m[k].lat_rot.T
                self.m[k].lon_rot = self.m[k].lon_rot.T

        # read model volume

        fid_m = open(os.path.join(directory, filename), 'r')

        if verbose is True:
            print 'read model file: %s' % os.path.join(directory, filename)

        v = np.array(fid_m.read().strip().split('\n'), dtype=float)

        fid_m.close()

        # assign values

        idx = 1
        for k in np.arange(self.nsubvol):

            n = int(v[idx])
            nx = len(self.m[k].lat) - 1
            ny = len(self.m[k].lon) - 1
            nz = len(self.m[k].r) - 1

            self.m[k].v = v[(idx + 1):(idx + 1 + n)].reshape(nx, ny, nz)

            idx = idx + n + 1

        # decide on global or regional model

        self.lat_min = 90.0
        self.lat_max = -90.0
        self.lon_min = 180.0
        self.lon_max = -180.0

        for k in np.arange(self.nsubvol):
            if np.min(self.m[k].lat_rot) < self.lat_min:
                self.lat_min = np.min(self.m[k].lat_rot)
            if np.max(self.m[k].lat_rot) > self.lat_max:
                self.lat_max = np.max(self.m[k].lat_rot)
            if np.min(self.m[k].lon_rot) < self.lon_min:
                self.lon_min = np.min(self.m[k].lon_rot)
            if np.max(self.m[k].lon_rot) > self.lon_max:
                self.lon_max = np.max(self.m[k].lon_rot)

        if ((self.lat_max - self.lat_min) > 90.0 or
                (self.lon_max - self.lon_min) > 90.0):
            self.global_regional = "global"

            self.lat_centre = (self.lat_max + self.lat_min) / 2.0
            self.lon_centre = (self.lon_max + self.lon_min) / 2.0

        else:
            self.global_regional = "regional"

        self.d_lat = 5.0
        self.d_lon = 5.0

    def write(self, directory, filename, verbose=False):
        """
        write ses3d model to a file

        write(self,directory,filename,verbose=False):
        """
        fid_m = open(os.path.join(directory, filename), 'w')

        if verbose is True:
            print 'write to file %s' % os.path.join(directory, filename)

        fid_m.write(str(self.nsubvol) + '\n')

        for k in np.arange(self.nsubvol):

            nx = len(self.m[k].lat) - 1
            ny = len(self.m[k].lon) - 1
            nz = len(self.m[k].r) - 1

            fid_m.write(str(nx * ny * nz) + '\n')

            for idx in np.arange(nx):
                for idy in np.arange(ny):
                    for idz in np.arange(nz):

                        fid_m.write(str(self.m[k].v[idx, idy, idz]) + '\n')

        fid_m.close()

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
                   lasif_folder=None):
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
        import matplotlib.pylab as plt
        from mpl_toolkits.basemap import Basemap

        radius = 6371.0 - depth

        # set up a map and colourmap

        if self.global_regional == 'regional':
            m = Basemap(projection='merc', llcrnrlat=self.lat_min,
                        urcrnrlat=self.lat_max, llcrnrlon=self.lon_min,
                        urcrnrlon=self.lon_max, lat_ts=20, resolution=res)
            m.drawparallels(
                np.arange(self.lat_min, self.lat_max, self.d_lon),
                labels=[1, 0, 0, 1])
            m.drawmeridians(
                np.arange(self.lon_min, self.lon_max, self.d_lat),
                labels=[1, 0, 0, 1])
        elif self.global_regional == 'global':
            m = Basemap(projection='ortho', lon_0=self.lon_centre,
                        lat_0=self.lat_centre, resolution=res)
            m.drawparallels(np.arange(-80.0, 80.0, 10.0), labels=[1, 0, 0, 1])
            m.drawmeridians(
                np.arange(-170.0, 170.0, 10.0), labels=[1, 0, 0, 1])

        m.drawcoastlines(linewidth=3.0)
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
            my_colormap = cm.make_colormap({
                0.0: [1.0, 1.0, 1.0],
                0.15: [1.0, 1.0, 1.0],
                0.85: [0.0, 0.0, 0.0],
                1.0: [0.0, 0.0, 0.0]})

        # loop over subvolumes to collect information

        x_list = []
        y_list = []
        idz_list = []
        N_list = []

        if lasif_folder:
            from lasif.scripts.lasif_cli import _find_project_comm

            comm = _find_project_comm(lasif_folder, read_only_caches=True)

        for k in np.arange(self.nsubvol):

            nx = len(self.m[k].lat)
            ny = len(self.m[k].lon)

            r = self.m[k].r

            # collect subvolumes within target depth

            if (max(r) >= radius) and (min(r) < radius):

                N_list.append(k)

                r = r[0:len(r) - 1]
                idz = min(
                    np.where(min(np.abs(r - radius)) == np.abs(r - radius))[0])
                if idz == len(r):
                    idz -= idz
                idz_list.append(idz)

                if verbose is True:
                    print 'true plotting depth: ' + str(6371.0 - r[idz]) + \
                        ' km'

                if lasif_folder:
                    from lasif import rotations
                    lon = self.m[k].lon_rot[0:nx - 1, 0:ny - 1]
                    lat = self.m[k].lat_rot[0:nx - 1, 0:ny - 1]

                    lat_r, lon_r = rotations.rotate_lat_lon(
                        lat.ravel(), lon.ravel(),
                        comm.project.domain.rotation_axis,
                        comm.project.domain.rotation_angle_in_degree)

                    lat_r.shape = lat.shape
                    lon_r.shape = lon.shape

                    x, y = m(lon_r, lat_r)

                else:
                    x, y = m(self.m[k].lon_rot[0:nx - 1, 0:ny - 1],
                             self.m[k].lat_rot[0:nx - 1, 0:ny - 1])
                x_list.append(x)
                y_list.append(y)

        # make a (hopefully) intelligent colour scale

        if min_val_plot is None:

            if len(N_list) > 0:

                # compute some diagnostics

                min_list = []
                max_list = []
                percentile_list = []

                for k in np.arange(len(N_list)):

                    min_list.append(
                        np.min(self.m[N_list[k]].v[:, :, idz_list[k]]))
                    max_list.append(
                        np.max(self.m[N_list[k]].v[:, :, idz_list[k]]))
                    percentile_list.append(
                        np.percentile(np.abs(
                            self.m[N_list[k]].v[:, :, idz_list[k]]), 99.0))

                minval = np.min(min_list)
                maxval = np.max(max_list)
                percent = np.max(percentile_list)

                # min and max roughly centred around zero

                if (minval * maxval < 0.0):
                    max_val_plot = percent
                    min_val_plot = -max_val_plot

                # min and max not centred around zero

                else:
                    max_val_plot = maxval
                    min_val_plot = minval

        # loop over subvolumes to plot

        for k in np.arange(len(N_list)):
            im = m.pcolormesh(
                x_list[k], y_list[k], self.m[N_list[k]].v[:, :, idz_list[k]],
                cmap=my_colormap, vmin=min_val_plot, vmax=max_val_plot)

            # if colormap=='mono':
            # cs=m.contour(x_list[k],y_list[k],self.m[N_list[k]].v[:,:,
            # idz_list[k]], colors='r',linewidths=1.0)
            # plt.clabel(cs,colors='r')

        # make a colorbar and title
        m.colorbar(im, "right", size="3%", pad='2%')
        plt.title(str(depth) + ' km')

        # save image if wanted
        if save_under is None:
            plt.show()
        else:
            plt.savefig(save_under + '.png', format='png', dpi=200)
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
