#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import abc
import arrow
import click
import datetime
import enum
import io
import os
import six
import subprocess
import uuid

from ..utils import get_template


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


@enum.unique
class Status(enum.Enum):
    running = 0
    unknown = 1
    finished = 2
    cancelled = 3


class SiteConfig(six.with_metaclass(abc.ABCMeta)):
    """
    Abstract base class for a site configuration.

    Subclass this to add support for different computers and machines to
    SES3D control. Subclasses will have to implement all the abstract
    methods and properties.

    On each machine, a job will be identified by a job name.
    """
    executable = os.path.basename(SES3D_EXECUTABLE)
    executable_path = os.path.dirname(SES3D_EXECUTABLE)

    def __init__(self, working_directory, log_directory):
        """
        :param working_directory: The main working directory for the site.
        :param log_directory:  The log directory for the site. Usually
            within the working directory.
        """
        self._working_dir = working_directory
        self._log_directory = log_directory

    @abc.abstractproperty
    def mpi_compiler(self):
        """
        Name of the MPI compiler.
        """
        pass

    @abc.abstractproperty
    def mpi_compiler_flags(self):
        """
        Flags for the MPI compiler.
        """
        pass

    @abc.abstractmethod
    def _get_status(self, job_name):
        """
        Returns the status of the given job in form of a
        :class:`~ses3d_ctrl.sites.site_config.Status` value.

        :param job_name: The job name.
        """
        pass

    @abc.abstractmethod
    def _cancel_job(self, job_name):
        """
        Cancel the job if it is running by whatever means necessary/supported.

        Return True if the job got cancelled successfully, otherwise False.

        :param job_name: The job name.
        """
        pass

    @abc.abstractmethod
    def _run_ses3d(self, job_name, cpu_count):
        """
        Launch SES3D for the given job and cpu count.

        :param job_name: The name of the job.
        :param cpu_count: The number of CPUs.
        """
        pass

        pass

    def get_stdout_file(self, job_name):
        """
        Return the filename where stdout is piped to.

        In the future this might need to become an abstract method with
        implementation in the subclasses.

        :param job_name: The name of the job.
        """
        return os.path.join(self.get_log_dir(job_name), "stdout")

    def get_stderr_file(self, job_name):
        """
        Return the filename where stderr is piped to.

        In the future this might need to become an abstract method with
        implementation in the subclasses.

        :param job_name: The name of the job.
        """
        return os.path.join(self.get_log_dir(job_name), "stderr")

    def get_ses3d_run_dir(self, job_name):
        """
        Returns the MAIN directory of SES3D where the executable should be
        launched from.

        :param job_name: The name of the job.
        """
        return os.path.join(self.working_dir, job_name,
                            self.executable_path)

    @property
    def working_dir(self):
        return os.path.expandvars(os.path.expanduser(self._working_dir))

    def get_log_dir(self, job_name):
        log_dir = os.path.join(self._log_directory, job_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def get_status_filename(self, job_name):
        return os.path.join(self.get_log_dir(job_name), "__STATUS")

    def set_status(self, job_name, status):
        """
        Set the status of the given job.

        :param job_name: The name of the job.
        :param status: Status as a string or an enumeration value.
        """
        # Make sure the status is valid.
        if not isinstance(status, Status):
            status = getattr(Status, status.lower())

        # Write the file.
        with io.open(self.get_status_filename(job_name), "wb") as fh:
            fh.write("%s---%s" % (arrow.utcnow().isoformat(), status.name))

    def get_status(self, job_name):
        """
        Get the status and last changed time of a job. If the job is
        running, it will once again check the state of the job.

        :param job_name: The name of the job.
        """
        with io.open(self.get_status_filename(job_name), "rb") as fh:
            time, status = fh.readline().split("---")
        time = arrow.get(time)
        status = getattr(Status, status.lower())

        # Update in case status is running.
        if status == Status.running:
            status = self._get_status(job_name)
            time = arrow.utcnow()
            self.set_status(job_name, status)
        return {"time": time, "status": status}

    def run_ses3d(self, job_name, cpu_count):
        """
        Run SES3D on the given number of CPUs.

        :param job_name: The name of the job.
        :param cpu_count: The number CPU cores.
        """
        self.set_status(job_name=job_name, status=Status.running)
        self._run_ses3d(job_name=job_name, cpu_count=cpu_count)

    def cancel_job(self, job_name):
        """
        Cancel the given job and set the status if necessary.

        :param job_name: The name of the job.
        """
        cancelled = self._cancel_job(job_name=job_name)
        if cancelled:
            self.set_status(job_name=job_name, status=Status.cancelled)
        else:
            click.echo("Job '%s' could not be cancelled." % job_name)

    def get_new_working_directory(self):
        """
        Gets a new working directory for a job. The name of the directory is
        also the name of the job.
        """
        time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        directory = "%s_%s" % (time_str, str(uuid.uuid4()).split("-")[0])
        directory = os.path.join(self.working_dir, directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

    def compile_ses3d(self, cwd, nx_max, ny_max, nz_max, lpd, fw_lpd, maxnt,
                      maxnr, pml_count, pml_limit=-1):
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

        with io.open(get_template("ses3d_conf"), "rt") as fh:
            with io.open(filename, "wt") as fh2:
                fh2.write(fh.read().format(
                    NX_MAX=nx_max,
                    NY_MAX=ny_max,
                    NZ_MAX=nz_max,
                    LPD=lpd,
                    FW_LPD=fw_lpd,
                    MAXNT=maxnt,
                    MAXNR=maxnr,
                    PML=pml_count,
                    PML_LIMIT=pml_limit))

        args = [self.mpi_compiler]
        args.extend(self.mpi_compiler_flags)
        args.append("-o")
        args.append(SES3D_EXECUTABLE)
        args.extend(SES3D_SOURCES)
        subprocess.check_call(args, cwd=cwd)
