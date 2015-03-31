import io

def _parse_input_file(filename, contents):
    values = {}
    idx = 0
    with io.open(filename, "rt") as fh:
        for line in fh:
            line = line.strip()
            if line.endswith("=========="):
                continue
            line = line.split()
            # No need to store the adjoint path. We'll change it in any case.
            if len(line) == 1:
                continue
            values[contents[idx][0]] = contents[idx][1](line[0])
            idx += 1
    return values


def parse_setup_file(filename):
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
    return _parse_input_file(filename, contents)


def parse_event_file(filename):
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
    return _parse_input_file(filename, contents)


def get_receiver_count(filename):
    with io.open(filename, "rt") as fh:
        return int(fh.readline().strip())
