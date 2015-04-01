import io

from .site_config import SiteConfig
from ..utils import get_template


class SuperMuc(SiteConfig):
    site_name = "supermuc"

    @property
    def mpi_compiler(self):
        return "mpicc"

    @property
    def mpi_compiler_flags(self):
        return ["-std=c99"]

    def _run_ses3d(self, job_name, cpu_count):
        """
        Launch the job on supermuc.

        :param job_name: The name of the job.
        :param cpu_count: The number of CPUs to use.
        """
        if cpu_count <= 512:
            job_class = "micro"
        elif cpu_count <= 8192:
            job_class = "general"
        else:
            raise ValueError("The job would require to run on more than one "
                             "island. This is probably not desired thus not "
                             "supported right now.")

        with io.open(get_template("supermuc_job_file"), "rt") as fh:
            fh.read().format(
                job_class=job_class,
                number_of_cores=cpu_count,
                job_name=job_name,
                stdout=self.get_stdout_file(job_name),
                stderr=self.get_stderr_file(job_name),
                email=self.config.email,
                initial_dir=self.get_ses3d_run_dir(job_name),
                executable=self.executable
            )
