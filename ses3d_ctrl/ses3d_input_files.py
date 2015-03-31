import glob
import io
import os

import numpy as np


class SES3DInputFiles(object):
    """
    Simple object representing SES3D input files.
    """
    def __init__(self, folder):
        if not os.path.exists(folder):
            raise ValueError("Folder '%s' does not exist." % folder)

        self.folder = folder

        files = ["setup", "relax", "stf"]
        files = {_i: os.path.join(folder, _i) for _i in files}

        for filename in files.values():
            if not os.path.exists(filename):
                raise ValueError("File '%s' does not exist" % filename)

        self.setup = self.parse_setup_file(files["setup"])

        # Find all events and associated receiver files.
        events = [_i for _i in glob.glob(os.path.join(folder, "event_*")) if
                  os.path.basename(_i) != "event_list"]
        recfiles = glob.glob(os.path.join(folder, "recfile_*"))

        events = {os.path.basename(_i).lstrip("event_"): _i for _i in events}
        recfiles = {os.path.basename(_i).lstrip("recfile_"): _i
                    for _i in recfiles}

        if set(events.keys()) != set(recfiles.keys()):
            raise ValueError("Event and receiver files in folder '%s' don't "
                             "match.")

        self.stf = self.parse_stf(files["stf"])

        self.events = \
            {name: {"filename": filename,
                    "receiver_file": recfiles[name],
                    "receiver_count": self.get_receiver_count(recfiles[name]),
                    "contents": self.parse_event_file(filename)}
             for name, filename in events.items()}

        # Some very basic checks.
        if self.max_nt > len(self.stf):
            raise ValueError(
                "The biggest event wants to run for %i timesteps, "
                "the STF only has %i timesteps." % (self.max_nt,
                                                    len(self.stf)))

    @property
    def max_nt(self):
        return max([_i["contents"]["nt"] for _i in self.events.values()])

    @property
    def max_receivers(self):
        return max([_i["receiver_count"] for _i in self.events.values()])

    def _parse_input_file(self, filename, contents):
        values = {}
        idx = 0
        with io.open(filename, "rt") as fh:
            for line in fh:
                line = line.strip()
                if line.endswith("=========="):
                    continue
                line = line.split()
                # No need to store the adjoint path. We'll change it in any
                # case.
                if len(line) == 1:
                    continue
                values[contents[idx][0]] = contents[idx][1](line[0])
                idx += 1
        return values

    def parse_setup_file(self, filename):
        contents = [
            ("theta_min", float),
            ("theta_max", float),
            ("phi_min", float),
            ("phi_max", float),
            ("z_min", float),
            ("z_max", float),
            ("is_diss", int),
            ("model_type", int),
            ("nx_global", int),
            ("ny_global", int),
            ("nz_global", int),
            ("lpd", int),
            ("px", int),
            ("py", int),
            ("pz", int),
            ("adjoint_flag", int),
            ("samp_ad", int)]
        return self._parse_input_file(filename, contents)

    def parse_event_file(self, filename):
        contents = [
            ("nt", int),
            ("dt", float),
            ("xxs", float),
            ("yys", float),
            ("zzs", float),
            ("srctype", int),
            ("M_theta_theta", float),
            ("M_phi_phi", float),
            ("M_r_r", float),
            ("M_theta_phi", float),
            ("M_theta_r", float),
            ("M_phi_r", float),
            ("ssamp", int),
            ("output_displacement", int)]
        return self._parse_input_file(filename, contents)

    def get_receiver_count(self, filename):
        with io.open(filename, "rt") as fh:
            return int(fh.readline().strip())

    def parse_stf(self, filename):
        return np.loadtxt(filename, dtype=np.float64)
