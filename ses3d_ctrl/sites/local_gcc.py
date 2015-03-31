import io
import os
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

    @property
    def compiler(self):
        return "gcc"

    @property
    def compiler_flags(self):
        return ["-std=c99"]

    def get_pid_file(self, log_dir):
        return os.path.join(log_dir, "PID")

    def get_stdout_file(self, log_dir):
        return os.path.join(log_dir, "stdout")

    def get_stderr_file(self, log_dir):
        return os.path.join(log_dir, "stderr")

    def _get_status(self, job_name, log_dir):
        # First read the stdout file and check if it reached the end.
        with open(self.get_stdout_file(log_dir), "r") as fh:
            # Read the last line.
            fh.seek(-1024, 2)
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                last_line = line
        if last_line.startswith("SES3D_R07_B: End:"):
            return "DONE"
        return "RUNNING"

    def _run_ses3d(self, job_name, log_dir, cpu_count):
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
        with io.open(self.get_stdout_file(log_dir), "wb") as stdout:
            with io.open(self.get_stderr_file(log_dir), "wb") as stderr:
                p = subprocess.Popen(
                    ["mpirun", "-n", str(cpu_count), self.executable],
                    cwd=os.path.join(self.working_dir, job_name,
                                     self.executable_path),
                    stdout=stdout, stderr=stderr)

        pid_file = self.get_pid_file(log_dir)
        with open(pid_file, "wt") as fh:
            fh.write(str(p.pid))

