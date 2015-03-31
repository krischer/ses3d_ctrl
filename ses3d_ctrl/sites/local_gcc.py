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
