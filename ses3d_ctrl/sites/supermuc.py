import click
import io
import os
import subprocess

from .site_config import SiteConfig, Status
from ..utils import get_template


class JobNotFoundError(ValueError):
    pass


class SuperMuc(SiteConfig):
    site_name = "supermuc"

    @property
    def mpi_compiler(self):
        return "mpicc"

    @property
    def mpi_compiler_flags(self):
        return ["-std=c99"]

    @property
    def fortran_compiler(self):
        return "gfortran"

    @property
    def c_compiler(self):
        return "gcc"

    @property
    def c_compiler_flags(self):
        return []

    @property
    def fortran_compiler_flags(self):
        # gfortran cuts lines after 132 chars by default...
        return ["-ffree-line-length-none"]

    def _cancel_job(self, job_name):
        status = self.get_status(job_name)

        if status == Status.finished:
            click.echo("Job already finished.")
            return False
        elif status == Status.cancelled:
            click.echo("Job already cancelled.")
            return False
        elif status == Status.unknown:
            click.echo("Job status unknown. Cannot cancel ...")
            return False

        job_id = self.get_job_info(job_name=job_name)["job_id"]

        click.echo(subprocess.check_output("llcancel %s" % job_id, shell=True))
        return True

    def get_job_info(self, job_name):
        """
        Returns some basic information about the job from the load leveller.

        :param job_name: The name of the job.
        """
        output = subprocess.check_output("llq -u $(whoami) -f %jn %st %id",
                                         shell=True)
        # Skip first two and last line.
        output = output.splitlines()[2:-1]
        for line in output:
            line = line.strip()
            if not line:
                continue
            name, status, job_id = line.split()
            if not job_name.startswith(name):
                continue
            break
        else:
            raise JobNotFoundError("Job not found")

        if status == "I":
            status = Status.waiting
        elif status == "R":
            status = Status.running
        else:
            click.echo("Status '%s' not known." % status)
            status = Status.unknown

        return {"status": status, "job_id": job_id}

    def _get_status(self, job_name):
        # First check if the job is done by reading stdout file. The file
        # might not yet exists, thus catch an IOError.
        try:
            if self._stdout_inidicates_job_finished(job_name):
                return Status.finished
        except IOError:
            pass

        try:
            info = self.get_job_info(job_name)
        except JobNotFoundError:
            return Status.unknown

        return info["status"]

    def _run_ses3d(self, job_name, cpu_count, wall_time, email):
        """
        Launch the job on supermuc.

        :param job_name: The name of the job.
        :param cpu_count: The number of CPUs to use.
        :param wall_time: The wall time in hours.
        :param email: The email address to send notification to.
        """
        if cpu_count <= 512:
            job_class = "micro"
        elif cpu_count <= 8192:
            job_class = "general"
        else:
            raise ValueError("The job would require to run on more than one "
                             "island. This is probably not desired thus not "
                             "supported right now.")

        # Wall time calculation.
        hours, minutes = wall_time // 1, (wall_time % 1) * 60
        minutes, seconds = minutes // 1, (minutes % 1) * 60
        seconds = seconds // 1
        hours = int(hours)
        minutes = int(minutes)
        seconds = int(seconds)
        wall_time_str = "%.2i:%.2i:%.2i" % (hours, minutes, seconds)

        submit_script = os.path.join(self.working_dir, job_name,
                                     "submit_job.sh")

        with io.open(get_template("supermuc_job_file"), "rt") as fh:
            with io.open(submit_script, "wt") as fh2:
                fh2.write(
                    fh.read().format(
                        job_class=job_class,
                        number_of_cores=cpu_count,
                        job_name=job_name,
                        stdout=self.get_stdout_file(job_name),
                        stderr=self.get_stderr_file(job_name),
                        email=email,
                        initial_dir=self.get_ses3d_run_dir(job_name),
                        executable=self.executable,
                        wall_time=wall_time_str
                    ))

        # Launch the submit script.
        click.echo(
            subprocess.check_output(["llsubmit",
                                     os.path.abspath(submit_script)]))
