#!/usr/bin/env python
# -*- coding: utf-8 -*-
import abc
import datetime
import io
import os
import six
import subprocess
import uuid

SES_3D_CONF_TEMPLATE = os.path.join(os.path.dirname(__file__), os.path.pardir,
                                    "data", "ses3d_conf.h.template")


SES3D_EXECUTABLE = "MAIN/ses3d"
SES3D_SOURCES = [
    "SOURCE/ses3d_comm.c",
    "SOURCE/ses3d_diag.c",
    "SOURCE/ses3d_dist.c",
    "SOURCE/ses3d_evolution.c",
    "SOURCE/ses3d_global.c",
    "SOURCE/ses3d_grad.c",
    "SOURCE/ses3d_init.c",
    "SOURCE/ses3d_input.c",
    "SOURCE/ses3d_lib.c",
    "SOURCE/ses3d_main.c",
    "SOURCE/ses3d_output.c",
    "SOURCE/ses3d_util.c"]


class SiteConfig(six.with_metaclass(abc.ABCMeta)):
    def __init__(self, working_directory):
        self._working_dir = working_directory

    @abc.abstractproperty
    def compiler(self):
        pass

    @abc.abstractproperty
    def mpi_compiler(self):
        pass

    @abc.abstractproperty
    def mpi_compiler_flags(self):
        pass

    @abc.abstractproperty
    def compiler_flags(self):
        pass

    @property
    def working_dir(self):
        return os.path.expandvars(os.path.expanduser(self._working_dir))

    def get_new_working_directory(self):
        time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        directory = "%s_%s" % (time_str, str(uuid.uuid4()))
        directory = os.path.join(self.working_dir, directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

    def compile_ses3d(self, cwd, nx_max, ny_max, nz_max, lpd, fw_lpd, maxnt,
                      maxnr, pml_count):
        """
        Compile SES3D patching the config file beforehand.

        :param cwd: The directory with the SES3D source.
        :param nx_max: nx_max setting of SES3D.
        :param ny_max: ny_max setting of SES3D.
        :param nz_max: nz_max setting of SES3D.
        :param lpd: The polynomial degree for the simulation.
        :param fw_lpd: The polynomial degree at which the forward fields are
            stored.
        :param maxnt: The maximum number of time steps.
        :param maxnr: The maximum number of receivers.
        :param pml_count: The amount of boundary layers.
        """
        # Copy and fill ses3d_conf.h template.
        filename = os.path.join(cwd, "SOURCE", "ses3d_conf.h")

        with io.open(SES_3D_CONF_TEMPLATE, "rt") as fh:
            with io.open(filename, "wt") as fh2:
                fh2.write(fh.read().format(
                    NX_MAX=nx_max,
                    NY_MAX=ny_max,
                    NZ_MAX=nz_max,
                    LPD=lpd,
                    FW_LPD=fw_lpd,
                    MAXNT=maxnt,
                    MAXNR=maxnr,
                    PML=pml_count))

        args = [self.mpi_compiler]
        args.extend(self.mpi_compiler_flags)
        args.append("-o")
        args.append(SES3D_EXECUTABLE)
        args.extend(SES3D_SOURCES)
        subprocess.check_call(args, cwd=cwd)
