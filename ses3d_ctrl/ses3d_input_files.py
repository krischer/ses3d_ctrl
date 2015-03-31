import io

def parse_setup_file(filename):
    setup = {}
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
            setup[contents[idx][0]] = contents[idx][1](line[0])

    return setup
