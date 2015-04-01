import io
import os
import psutil
import subprocess
import sys

from .site_config import SiteConfig


class LocalGCC(SiteConfig):
    site_name = "local_gcc"

    @property
    def mpi_compiler(self):
        return "mpicc"

    @property
    def mpi_compiler_flags(self):
        return ["-std=c99"]

    def get_pid_file(self, job_name):
        return os.path.join(self.get_log_dir(job_name), "PID")

    def get_stdout_file(self, job_name):
        return os.path.join(self.get_log_dir(job_name), "stdout")

    def get_stderr_file(self, job_name):
        return os.path.join(self.get_log_dir(job_name), "stderr")

    def _stdout_inidicates_job_finished(self, job_name):
        with open(self.get_stdout_file(job_name), "r") as fh:
            # Read the last line.
            fh.seek(-1024, 2)
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                last_line = line
        if last_line.startswith("SES3D_R07_B: End:"):
            return True
        return False

    def _get_pid(self, job_name):
        with open(self.get_pid_file(job_name), "rt") as fh:
            return int(fh.readline().strip())

    def _process_is_running(self, job_name):
        """
        Determines if the process for the given job and log directory is still
        running.
        """
        # Get the job based on the pid.
        try:
            p = psutil.Process(self._get_pid(job_name))
            status = p.status()
        except psutil.NoSuchProcess:
            return False

        # Make sure it the correct process.
        if set([_i.name() for _i in p.get_children()]) != {"ses3d"}:
            return False

        # Make sure it has the correct working dir.
        if p.getcwd() != os.path.join(self.working_dir, job_name,
                                      self.executable_path):
            return False

        if status == "running":
            return True
        else:
            raise NotImplementedError("Unknown process status '%s'." % status)

    def _cancel_job(self, job_name):
        if not self._process_is_running(job_name):
            return
        try:
            p = psutil.Process(self._get_pid(job_name))
        except psutil.NoSuchProcess:
            return
        p.terminate()

    def _get_status(self, job_name):
        if self._process_is_running(job_name):
            if self._stdout_inidicates_job_finished(job_name):
                return "FINISHED"
            else:
                return "RUNNING"
        else:
            if self._stdout_inidicates_job_finished(job_name=job_name):
                return "FINISHED"
            else:
                return "UNKNOWN"

    def _run_ses3d(self, job_name, cpu_count):
        # Adapted from
        # http://code.activestate.com/recipes/
        # 66012-fork-a-daemon-process-on-unix/
        # do the UNIX double-fork magic, see Stevens' "Advanced
        # Programming in the UNIX Environment" for details (ISBN 0201563177)
        try:
            pid = os.fork()
            if pid > 0:
                # exit first parent
                sys.exit(0)
        except OSError, e:
            print >>sys.stderr, "fork #1 failed: %d (%s)" % (e.errno,
                                                             e.strerror)
            sys.exit(1)

        # decouple from parent environment
        os.chdir("/")
        os.setsid()
        os.umask(0)

        # do second fork
        try:
            pid = os.fork()
            if pid > 0:
                sys.exit(0)
        except OSError, e:
            print >>sys.stderr, "fork #2 failed: %d (%s)" % (e.errno,
                                                             e.strerror)
            sys.exit(1)

        # start the daemon main loop
        with io.open(self.get_stdout_file(job_name), "wb") as stdout:
            with io.open(self.get_stderr_file(job_name), "wb") as stderr:
                p = subprocess.Popen(
                    ["mpirun", "-n", str(cpu_count), self.executable],
                    cwd=os.path.join(self.working_dir, job_name,
                                     self.executable_path),
                    stdout=stdout, stderr=stderr)

        pid_file = self.get_pid_file(job_name)
        with open(pid_file, "wt") as fh:
            fh.write(str(p.pid))
