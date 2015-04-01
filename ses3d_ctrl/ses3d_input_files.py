import copy
import glob
import io
import os
import shutil

import numpy as np

from . import utils


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
        self.relaxation_times, self.relaxation_weights = \
            self.parse_relax(files["relax"])

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

    def merge(self, other):
        """
        Creates a new SES3DInputFile object merging self and other if they are
        compatible.

        Will raise otherwise.
        """
        # Events must not overlap.
        if set(self.events.keys()).intersection(set(other.events.keys())):
            raise ValueError("Objects must not share events names.")

        # Setup must be identical!
        if self.setup != other.setup:
            raise ValueError("Objects must have the same setup.")

        # Same for the stf and the relaxation settings.
        np.testing.assert_allclose(
            self.stf, other.stf, atol=self.stf.ptp() * 1E-7,
            err_msg="Objects must have the same source time function.")

        np.testing.assert_allclose(
            self.relaxation_times, other.relaxation_times,
            atol=self.stf.ptp() * 1E-7,
            err_msg="Objects must have the same relaxation times.")

        np.testing.assert_allclose(
            self.relaxation_weights, other.relaxation_weights,
            atol=self.stf.ptp() * 1E-7,
            err_msg="Objects must have the same relaxation weights.")

        # Copy and create merged_object. Only the events need to be merged. The
        # rest has to be identical to even reach this point in the code.
        merged = copy.deepcopy(self)
        merged.events.update(copy.deepcopy(other.events))

        return merged

    @property
    def max_nt(self):
        return max([_i["contents"]["nt"] for _i in self.events.values()])

    @property
    def max_receivers(self):
        return max([_i["receiver_count"] for _i in self.events.values()])

    def check_model_compatibility(self, px, py, pz):
        """
        Checks the compatibility of a given model with the input files.

        :param px: number of processors in x direction
        :param py: number of processors in y direction
        :param pz: number of processors in z direction
        """
        if px != self.setup["px"]:
            raise ValueError(
                "Model requires %i processors in x direction, "
                "input files specify %i." % (px, self.setup["px"]))

        if py != self.setup["py"]:
            raise ValueError(
                "Model requires %i processors in y direction, "
                "input files specify %i." % (py, self.setup["py"]))

        if pz != self.setup["pz"]:
            raise ValueError(
                "Model requires %i processors in z direction, "
                "input files specify %i." % (pz, self.setup["pz"]))

    def _parse_input_file(self, filename, contents):
        values = {}
        idx = 0
        with io.open(filename, "rt") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.endswith("=========="):
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

    def parse_relax(self, filename):
        with io.open(filename, "rt") as fh:
            fh.readline()
            relaxation_times = [float(fh.readline()) for _ in range(3)]
            fh.readline()
            weights = [float(fh.readline()) for _ in range(3)]
        return np.array(relaxation_times), np.array(weights)

    def write(self, output_folder, waveform_output_folder,
              adjoint_output_folder):
        # Assert the folder exists, but that it is empty.
        if not os.path.exists(output_folder):
            raise ValueError("Folder '%s' does not exist" % output_folder)

        if os.listdir(output_folder):
            raise ValueError("Folder '%s' is not empty" % output_folder)

        # Write the stf.
        output = ["#"] * 4
        output.extend(["%e" % _i for _i in self.stf])
        output.append("")
        with io.open(os.path.join(output_folder, "stf"), "wt") as fh:
            fh.write(u"\n".join(output))

        # Write the relaxation parameters.
        with io.open(utils.get_template("relax"), "rt") as fh:
            with io.open(os.path.join(output_folder, "relax"), "wt") as fh2:
                fh2.write(fh.read().format(
                    relaxation_times=u"\n".join(
                        map(str, self.relaxation_times)),
                    weights=u"\n".join(map(str, self.relaxation_weights))
                ))

        # Write setup.
        setup = copy.deepcopy(self.setup)
        setup["adjoint_folder"] = adjoint_output_folder

        with io.open(utils.get_template("setup"), "rt") as fh:
            with io.open(os.path.join(output_folder, "setup"), "wt") as fh2:
                fh2.write(fh.read().format(**setup))

        # Write the event and receiver files.
        for event_name, contents in self.events.items():
            # Copy the receiver file.
            filename = os.path.join(output_folder, "recfile_%s" % event_name)
            shutil.copy(contents["receiver_file"], filename)

            event = copy.deepcopy(contents["contents"])
            event["output_directory"] = os.path.join(waveform_output_folder,
                                                     event_name)

            # Write the event_file.
            filename = os.path.join(output_folder, "event_%s" % event_name)
            with io.open(utils.get_template("event"), "rt") as fh:
                with io.open(filename, "wt") as fh2:
                    fh2.write(fh.read().format(**event))

        # Write the event_list file.
        with io.open(os.path.join(output_folder, "event_list"), "wt") as fh:
            fh.write(u"%i                   ! n_events = number of events\n" %
                     len(self.events))
            for name in self.events.keys():
                fh.write(u"%s\n" % name)
            fh.write(u"\n")
