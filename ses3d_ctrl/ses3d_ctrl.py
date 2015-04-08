#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import click
import glob
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile

from . import utils
from .gradients import sum_kernels
from .sites import available_sites
from .sites.site_config import Status
from .ses3d_input_files import SES3DInputFiles


CONFIG_FILE_PATH = os.path.expanduser("~/.ses3d_ctrl.json")

DEFAULT_CONFIG = {
    "root_working_dir": "~/ses3d_ctrl_working_directory",
    "adjoint_dir": "/tmp/SES3D_TEMP_ADJOINT",
    "site_name": "local_gcc",
    "email": "change_me@www.org"
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SES3D_PATH = os.path.join(DATA_DIR, "ses3d_r07_b.tgz")
SES3D_MD5_CHECKSUM = "3834d425da0d49439e753fe481620c3d"


def _read_config_file():
    with io.open(CONFIG_FILE_PATH, "r") as fh:
        data = json.load(fh)
    default_keys = set(DEFAULT_CONFIG.keys())
    keys = set(data.keys())
    if default_keys != keys:
        raise ValueError("Config file '%s' must contain exactly these keys:\n"
                         "\t%s" % (CONFIG_FILE_PATH, default_keys))
    for key, value in data.items():
        if isinstance(value, (str, unicode, bytes)):
            data[key] = os.path.expanduser(os.path.expandvars(value))

    if not os.path.exists(data["adjoint_dir"]):
        os.makedirs(data["adjoint_dir"])
    return data


def _write_default_config_file():
    with io.open(CONFIG_FILE_PATH, "wb") as fh:
        json.dump(DEFAULT_CONFIG, fh,
                  sort_keys=True,
                  indent=4,
                  separators=(",", ": "))


def _check_ses3d_md5():
    """
    Checks the hash of the SES3D archive.
    :return:
    """
    # Use checksum to assert the file is correct.
    with io.open(SES3D_PATH, "rb") as fh:
        md5 = hashlib.md5(fh.read()).hexdigest()

    if md5 != SES3D_MD5_CHECKSUM:
        raise ValueError("md5 of the SES3D archive is not %s" %
                         SES3D_MD5_CHECKSUM)


class Config(object):
    """
    Config passed to all subcommands.
    """
    def update(self, items):
        for key, value in items.items():
            setattr(self, key, value)
        self.prepare()

    def prepare(self):
        if not os.path.exists(self.root_working_dir):
            os.makedirs(self.root_working_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.waveform_dir):
            os.makedirs(self.waveform_dir)
        if not os.path.exists(self.kernel_dir):
            os.makedirs(self.kernel_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if self.site_name not in available_sites:
            raise ValueError("Site '%s' is not available. Available sites: %s"
                             % sorted(available_sites.keys()))

        self.site = available_sites[self.site_name](
            working_directory=self.root_working_dir,
            log_directory=self.log_dir)

    def list_runs(self):
        build_dirs = set(
            [_i for _i in os.listdir(self.root_working_dir)
             if not _i.startswith("__") and
             os.path.isdir(os.path.join(self.root_working_dir, _i))])
        available_logs = set(os.listdir(self.log_dir))

        # Anything that has a log folder is a valid run. A build dir without a
        # corresponding log file is dangerous.
        diff = build_dirs.difference(available_logs)
        for d in diff:
            click.secho("Folder '%s' has no corresponding log entry. "
                        "Please remove the folder." % os.path.abspath(
                            os.path.join(self.root_working_dir, d)),
                        fg="red")
        return sorted(available_logs)

    @property
    def model_dir(self):
        return os.path.join(self.root_working_dir, "__MODELS")

    @property
    def log_dir(self):
        return os.path.join(self.root_working_dir, "__LOGS")

    @property
    def waveform_dir(self):
        return os.path.join(self.root_working_dir, "__WAVEFORMS")

    @property
    def kernel_dir(self):
        return os.path.join(self.root_working_dir, "__KERNELS")

    def list_models(self):
        return [_i for _i in os.listdir(self.model_dir)
                if os.path.isdir(os.path.join(self.model_dir, _i))]

    def get_model_path(self, name):
        if name not in self.list_models():
            raise ValueError("Model %s not found" % name)
        return os.path.join(self.model_dir, name)

    def get_model_settings(self, name):
        path = self.get_model_path(name)
        boxfile = os.path.join(path, "boxfile")
        with io.open(boxfile, "rt") as fh:
            for _ in range(15):
                fh.readline()
            px = int(fh.readline().strip())
            py = int(fh.readline().strip())
            pz = int(fh.readline().strip())
        return {"px": px, "py": py, "pz": pz}

    def __str__(self):
        ret_str = (
            "SES3D Ctrl Config "
            "(config file: '{config_file}')"
        )
        ret_str = ret_str.format(config_file=CONFIG_FILE_PATH)
        for key in DEFAULT_CONFIG:
            ret_str += "\n\t{key}: {value}".format(
                key=key, value=getattr(self, key))
        return ret_str

pass_config = click.make_pass_decorator(Config, ensure=True)


@click.group()
@pass_config
def cli(config):
    """
    SES3D Ctrl.

    Small suite assisting in launching and managing a large number of SES3D
    runs.
    """
    if not os.path.exists(CONFIG_FILE_PATH):
        _write_default_config_file()
        print("No config file discovered. Created one at '%s'. Please edit it "
              "and restart the program." % (CONFIG_FILE_PATH))
        sys.exit(0)
    config.update(_read_config_file())


@cli.command()
@pass_config
def info(config):
    """
    Print some basic information.
    """
    click.echo(config)


def _progress(msg, warn=False):
    """
    Consistent and "pretty" progress logs.
    """
    click.echo(click.style(" -> ", fg="blue"), nl=False)
    if warn:
        fg = "red"
    else:
        fg = "green"
    click.echo(click.style(msg, fg=fg))


@cli.command()
@click.option("--model", type=str, required=True,
              help="The model to use for the run.")
@click.option("--lpd", type=int, default=4, show_default=True,
              help="Degree of the Lagrange polynomials for the simulation")
@click.option("--fw-lpd", type=int, default=1, show_default=True,
              help="Polynomial degree for storing the forward field")
@click.option("--pml-count", type=int, default=3, show_default=True,
              help="Number of PMLs at each boundary.")
@click.option("--pml-limit", type=int, default=10000, show_default=True,
              help="Number of time steps for which PMLs are enabled.")
@click.option("--parallel-events", type=int, default=1, show_default=True,
              help="The number of events to run in parallel.")
@click.option("--wall-time-per-event", type=float, default=0.5,
              help="Wall time per event in hours. Only needed for some sites.")
@click.argument("input_files_folders", type=click.Path(), nargs=-1)
@pass_config
def run_forward(config, model, input_files_folders, lpd, fw_lpd, pml_count,
                pml_limit, wall_time_per_event, parallel_events):
    """
    Run a simulation for the chosen input files. If multiple input file folders
    are given, they will be merged.
    """
    model = model.lower()
    if model not in config.list_models():
        raise ValueError("Model '%s' not known. Available models:\n%s" % (
            model, "\n\t".join(config.list_models())))

    # Read and merge all input files.
    _progress("Parsing, checking, and merging input files ...")
    input_files = [SES3DInputFiles(_i) for _i in input_files_folders]
    ip = input_files.pop()

    for _i in input_files:
        ip = ip.merge(_i)
    input_files = ip

    # Check the compatibility with the model.
    m = config.get_model_settings(model)
    input_files.check_model_compatibility(**m)

    if len(input_files.events) % parallel_events != 0:
        raise ValueError("The total number of events must be a multiple of "
                         "the number of parallel events.")

    # Now we now the total number of events, let's calculate the total wall
    # time and the number of CPUs required.
    s = input_files.setup
    wall_time_in_hours = wall_time_per_event * \
        float(len(input_files.events)) / float(parallel_events)
    cpu_count = s["px"] * s["py"] * s["pz"] * parallel_events

    # Get a new working directory for a forward run.
    cwd = config.site.get_new_working_directory(job_type="fw")
    run_name = os.path.basename(cwd)
    _progress("Initializing run '%s' ..." % run_name)

    # Untar SES3D to that working directory.
    _progress("Extracting SES3D ...")

    _check_ses3d_md5()

    with tarfile.open(SES3D_PATH, "r:gz") as fh:
        for member in fh.getmembers():
            if not member.isreg():
                continue
            member.name = os.path.sep.join(member.name.split(os.path.sep)[1:])
            fh.extract(member, cwd)

    _progress("Compiling SES3D ...")
    config.site.compile_ses3d(
        cwd=cwd,
        nx_max=s["nx_global"] // s["px"],
        ny_max=s["ny_global"] // s["py"],
        nz_max=s["nz_global"] // s["pz"],
        maxnt=input_files.max_nt, maxnr=input_files.max_receivers,
        lpd=lpd, fw_lpd=fw_lpd, pml_count=pml_count, pml_limit=pml_limit)

    _progress("Copying input files ...")

    input_file_dir = os.path.join(cwd, "INPUT")
    if os.path.exists(input_file_dir):
        shutil.rmtree(input_file_dir)
    os.makedirs(input_file_dir)

    # Directory where the waveforms will be stored. Must be relative for SES3D.
    waveform_folder = os.path.join(config.waveform_dir, run_name)
    if not os.path.exists(waveform_folder):
        os.makedirs(waveform_folder)
    waveform_folder = os.path.relpath(waveform_folder,
                                      os.path.join(cwd, "MAIN"))

    input_files.write(
        output_folder=input_file_dir,
        waveform_output_folder=waveform_folder,
        adjoint_output_folder=os.path.join(os.path.abspath(
            config.adjoint_dir), run_name))

    _progress("Copying model ...")
    model_folder = os.path.join(cwd, "MODELS", "MODELS")
    if os.path.exists(model_folder):
        shutil.rmtree(model_folder)
    shutil.copytree(config.get_model_path(model), model_folder)

    _progress("Launching SES3D on %i cores ..." % cpu_count)
    config.site.run_ses3d(job_name=run_name, cpu_count=cpu_count,
                          wall_time=wall_time_in_hours,
                          email=config.email)


@cli.command()
@click.option("--fw_run", type=str, required=True,
              help="The name of the associated forward run")
@click.option("--parallel-events", type=int, default=1, show_default=True,
              help="The number of events to run in parallel.")
@click.option("--wall-time-per-event", type=float, default=0.5,
              help="Wall time per event in hours. Only needed for some sites.")
@click.argument("adjoint_source_folders", type=click.Path(), nargs=-1)
@pass_config
def run_adjoint(config, fw_run, parallel_events,
                wall_time_per_event, adjoint_source_folders):
    """
    Runs a reverse (adjoint) simulation for a given forward run.

    Each event used in the forward simulation must have adjoint sources. The
    event name must be part of the folder name.
    """
    if not fw_run.endswith("_fw"):
        raise ValueError("Forward run name must end with '_fw'.")

    status = config.site.get_status(fw_run)
    if status["status"] != Status.finished:
        raise ValueError("Forward run %s not yet finished." % fw_run)

    # Safer than str.replace(). The validity of the run name has been
    # asserted before.
    bw_run = fw_run[:-3] + "_bw"
    fw_run_folder = config.site.get_working_dir_name(fw_run)
    bw_run_folder = config.site.get_working_dir_name(bw_run)

    if os.path.exists(bw_run_folder):
        raise ValueError("Backwards run already exists: %s" % bw_run_folder)

    _progress("Matching adjoint sources with events from forward run ...")
    # Time to parse the input files and get a list of events.
    input_files = SES3DInputFiles(os.path.join(fw_run_folder, "INPUT"))

    if len(input_files.events) % parallel_events != 0:
        raise ValueError("The total number of events must be a multiple of "
                         "the number of parallel events.")

    # Now we now the total number of events, let's calculate the total wall
    # time and the number of CPUs required.
    s = input_files.setup
    wall_time_in_hours = wall_time_per_event * \
        float(len(input_files.events)) / float(parallel_events)
    cpu_count = s["px"] * s["py"] * s["pz"] * parallel_events

    events_fw_run = list(input_files.events.keys())

    for folder in adjoint_source_folders:
        if not os.path.exists(os.path.join(folder, "ad_srcfile")):
            raise ValueError("Folder '%s' has not 'ad_srcfile'." % folder)

    # Now find the adjoint source folder for each event.
    events_bw_run = {}
    for event in events_fw_run:
        adj_srcs = [_i for _i in adjoint_source_folders if event in
                    os.path.basename(_i)]
        if not adj_srcs:
            raise ValueError("Could not find adjoint sources for event %s."
                             % event)
        if len(adj_srcs) > 1:
            raise ValueError(
                "More than one potential folder with adjoint sources for "
                "events %s found: \n\t%s" % (event, "\n\t".join(adj_srcs)))
        events_bw_run[event] = adj_srcs[0]

    _progress("Copying forward run folder ...")
    shutil.copytree(fw_run_folder, bw_run_folder)

    _progress("Copying adjoint sources ...")
    adjoint_folder = os.path.join(bw_run_folder, "ADJOINT")
    if os.path.exists(adjoint_folder):
        shutil.rmtree(adjoint_folder)
    os.makedirs(adjoint_folder)

    for event_name, folder in events_bw_run.items():
        shutil.copytree(folder, os.path.join(adjoint_folder, event_name))

    _progress("Creating input files ...")
    # Set adjoint flag to 2, meaning an adjoint reverse simulation
    input_files.setup["adjoint_flag"] = 2

    # Directory where the kernels will be stored. Must be relative for SES3D.
    kernel_folder = os.path.join(config.kernel_dir, bw_run)
    if not os.path.exists(kernel_folder):
        os.makedirs(kernel_folder)
    kernel_folder = os.path.relpath(kernel_folder,
                                    os.path.join(bw_run_folder, "MAIN"))

    input_file_dir = os.path.join(bw_run_folder, "INPUT")
    if os.path.exists(input_file_dir):
        shutil.rmtree(input_file_dir)
    os.makedirs(input_file_dir)

    input_files.write(
        output_folder=input_file_dir,
        waveform_output_folder=kernel_folder,
        adjoint_output_folder=os.path.join(os.path.abspath(
            config.adjoint_dir), fw_run))

    _progress("Launching SES3D on %i cores ..." % cpu_count)
    config.site.run_ses3d(job_name=bw_run, cpu_count=cpu_count,
                          wall_time=wall_time_in_hours,
                          email=config.email)


@cli.command()
@click.argument("model-name", type=str)
@pass_config
def remove_model(config, model_name):
    """
    Delete a model.
    """
    shutil.rmtree(config.get_model_path(model_name.lower()))


@cli.command()
@click.option("--name", type=str, required=True,
              help="Case insensitive name of the model")
@click.argument("model_folder",
                type=click.Path(exists=True, readable=True, resolve_path=True,
                                dir_okay=True))
@pass_config
def add_model(config, name, model_folder):
    """
    Add a model to SES3D ctrl.
    """
    model_name = name.lower().replace(" ", "_")
    if model_name in config.list_models():
        raise ValueError("Model '%s' already exists" % model_name)
    # Assert the folder has a boxfile.
    if not os.path.exists(os.path.join(model_folder, "boxfile")):
        raise ValueError("Folder has no boxfile. Not a model")

    shutil.copytree(model_folder, os.path.join(config.model_dir, model_name))


@cli.command()
@pass_config
def list_models(config):
    """
    Lists available models.
    """
    models = config.list_models()
    if not models:
        click.echo("No models available")
        return
    for model in sorted(models):
        click.echo("\t%s" % model)


@cli.command()
@pass_config
def list_runs(config):
    """
    Lists available runs.
    """
    runs = config.list_runs()
    if not runs:
        click.echo("No runs available")
        return
    for run in sorted(runs):
        click.echo("\t%s" % run)


@cli.command()
@pass_config
def status(config):
    """
    Print the status of all runs.
    """
    line_fmt = "{job_number:35s}{status:20s}{updated:25s}"
    click.echo(line_fmt.format(
        job_number="JOB NUMBER", status="STATUS", updated="UPDATED"))
    click.echo("=" * 80)
    for run in config.list_runs():
        status = config.site.get_status(run)
        click.echo(line_fmt.format(
            job_number=run, status=status["status"].name.upper(),
            updated=status["time"].humanize()))


@cli.command()
@pass_config
@click.argument("job-number", type=str)
def cancel(config, job_number):
    """
    Cancel a certain job.
    """
    if job_number not in config.list_runs():
        raise ValueError("Job not known")
    config.site.cancel_job(job_number)


@cli.command()
@pass_config
@click.argument("job-name", type=str)
def force_remove_job(config, job_name):
    """
    Forcefully removes everything from a job regardless no matter if its still
    running or not.
    """
    if job_name not in config.list_runs():
        raise ValueError("Job not known.")
    log_dir = config.site.get_log_dir(job_name)
    waveform_dir = os.path.join(config.waveform_dir, job_name)
    job_dir = os.path.join(config.root_working_dir, job_name)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    if os.path.exists(waveform_dir):
        shutil.rmtree(waveform_dir)
    if os.path.exists(job_dir):
        shutil.rmtree(job_dir)


@cli.command()
@pass_config
def clean(config):
    """
    Delete all traces of jobs that are neither running nor finished.
    """
    for run in config.list_runs():
        status = config.site.get_status(run)["status"]
        if status == Status.running or status == Status.finished:
            continue

        potential_folders = [
            os.path.join(config.root_working_dir, run),
            os.path.join(config.site.get_log_dir(run)),
            os.path.join(config.waveform_dir, run)]

        for folder in potential_folders:
            if not os.path.exists(folder):
                continue
            if not os.path.isdir(folder):
                continue
            shutil.rmtree(folder)


@cli.command()
@pass_config
def cd(config):
    """
    Prints the root working dir to stdour use with "$ cd `agere cd`".
    """
    click.echo(config.root_working_dir)


@cli.command()
@pass_config
def tail(config):
    """
    Tails the output of running jobs.
    """
    all_files = []
    for run in config.list_runs():
        status = config.site.get_status(run)["status"]
        if status != Status.running:
            continue
        all_files.append(config.site.get_stderr_file(run))
        all_files.append(config.site.get_stdout_file(run))

    if not all_files:
        click.echo("No active runs")
        return

    all_files = [_i for _i in all_files if os.path.exists(_i)]

    os.system("tail -f %s" % " ".join(all_files))


@cli.command()
@pass_config
@click.option("-n", type=int, default=1, show_default=True,
              help="Get results for the n-th last output. 1 means last "
                   "output, 2 the second last, and so on.")
def ls_output(config, n):
    """
    List the output of finished runs.
    """
    runs = []

    for run in config.list_runs():
        info = config.site.get_status(run)
        if info["status"] != Status.finished:
            continue
        runs.append((run, info["time"]))

    # Sort by reverse time.
    runs = sorted(runs, key=lambda x: x[1], reverse=True)
    if n > len(runs):
        raise ValueError("Only %i runs available." % len(runs))

    run = runs[n - 1][0]
    click.echo("Waveforms:")
    waveform_folder = os.path.join(config.waveform_dir, run)
    contents = sorted(os.listdir(waveform_folder))
    for _i in contents:
        click.echo("\t%s" % os.path.join(waveform_folder, _i))


@cli.command()
@click.option('--compress', is_flag=True,
              help="Optionally compress (gzip level 1) the data. Slower but "
                   "results in smaller files.")
@pass_config
def tar_waveforms(config, compress):
    """
    Tar the resulting waveforms of the finished run.
    """
    runs = []
    for run in config.list_runs():
        status = config.site.get_status(run)["status"]
        # Only consider finished jobs.
        if status != Status.finished:
            continue
        waveform_folder = os.path.join(config.waveform_dir, run)

        # Either plain tar or tar.gz file.
        tar_filename = os.path.join(config.waveform_dir, "%s.tar" % run)
        tar_gz_filename = os.path.join(config.waveform_dir, "%s.tar.gz" % run)

        if os.path.exists(tar_filename) or os.path.exists(tar_gz_filename) \
                or not os.path.exists(waveform_folder):
            continue

        if compress:
            filename = tar_gz_filename
        else:
            filename = tar_filename

        runs.append((run, waveform_folder, filename))

    for _i, run in enumerate(runs):
        if _i:
            click.echo("\n")
        _progress("Tarring waveforms for run %i of %i ..." % (_i + 1,
                                                              len(runs)))

        # First find all files to be able to have a nice progressbar.
        all_files = []
        for root, _, files in os.walk(run[1]):
            if not files:
                continue
            for filename in files:
                all_files.append(os.path.join(root, filename))

        if not all_files:
            _progress("No files found. No archive will be created.", warn=True)
            continue

        relpath = os.path.join(config.waveform_dir, run[0])

        if compress:
            # This still compress quite a bit but takes much less
            # computations time so probably worth it.
            flags = {"mode": "w:gz", "compresslevel": 1}
        else:
            flags = {"mode": "w"}

        with click.progressbar(all_files) as files:
            with tarfile.open(run[2], **flags) as tf:
                for filename in files:
                    tf.add(filename,
                           arcname=os.path.relpath(filename, relpath))
        _progress("Created archive %s" % run[2])


@cli.command()
@click.option('--iteration-name', type=str, required=True,
              help="The iteration name")
@click.option('--lasif-project', type=click.Path(exists=True, dir_okay=True),
              required=True, help="The LASIF project root")
@click.argument("archives", type=click.Path(exists=True, file_okay=True),
                nargs=-1)
@pass_config
def unpack_waveforms(config, iteration_name, lasif_project, archives):
    """
    Unpacks the waveforms in the archives to the corresponding LASIF project.
    """
    config_file = os.path.join(lasif_project, "config.xml")
    if not os.path.exists(config_file):
        raise ValueError("%s does not contain a valid LASIF project." %
                         lasif_project)

    event_folder = os.path.join(lasif_project, "EVENTS")
    if not os.path.exists(event_folder):
        raise ValueError("%s does not contain an EVENTS folder." %
                         lasif_project)
    synthetics_folder = os.path.join(lasif_project, "SYNTHETICS")
    if not os.path.exists(synthetics_folder):
        raise ValueError("%s does not contain a SYNTHETICS folder." %
                         lasif_project)

    # Get the events in the LASIF project.
    events = [os.path.splitext(os.path.basename(_i))[0] for _i in glob.glob(
              os.path.join(event_folder, "*.xml"))]

    # Get events that already have data for the iteration in question.
    events_with_data = []
    long_iteration_name = "ITERATION_%s" % iteration_name
    for event in events:
        folder = os.path.join(synthetics_folder, event, long_iteration_name)
        if not os.path.exists(folder):
            continue
        contents = os.listdir(folder)
        if contents:
            events_with_data.append(event)

    # Loop over all
    existing_folders = []
    total_count = 0
    for _i, archive in enumerate(archives):
        count = 0
        _progress("Unpacking archive %i of %i [%s]" % (_i + 1, len(archives),
                                                       archive))
        with tarfile.open(archive, "r") as tf:
            for info in tf:
                if not info.isfile():
                    continue
                split_name = os.path.split(info.name)
                if len(split_name) != 2:
                    raise ValueError("%s in archive %s is not valid" % (
                        info.name, archive))
                event_name = split_name[0]
                if event_name not in events:
                    raise ValueError(
                        "Event %s part of archive %s is not part of the LASIF "
                        "project." % (event_name, archive))
                if event_name in events_with_data:
                    raise ValueError(
                        "Event %s, part of archive %s already has data in "
                        "LASIF for iteration %s" % (event_name, archive,
                                                    iteration_name))

                folder_name = os.path.join(synthetics_folder, event_name,
                                           long_iteration_name)
                # Cache the availability query.
                if folder_name not in existing_folders:
                    if not os.path.exists(folder_name):
                        os.makedirs(folder_name)
                    existing_folders.append(folder_name)

                filename = os.path.join(folder_name, os.path.basename(
                                        info.name))
                with open(filename, "wb") as fh:
                    fh.write(tf.extractfile(info).read())

                count += 1
                total_count += 1

                if not count % 1000:
                    _progress("\tUnpacked %i files ..." % count)

    _progress("Extracted %i files." % total_count)

@cli.command()
@click.option("--model", type=str, required=True,
              help="The model to use for the run.")
@click.option('--verbose', is_flag=True, show_default=True,
              help="Controls the verbosity of the output")
@click.option("--output_folder", type=click.Path(), required=True,
              help="The output folder")
@click.argument("kernels", type=click.Path(exists=True, dir_okay=True),
                nargs=-1)
@pass_config
def generate_gradients(config, model, verbose, output_folder, kernels):
    """
    Projects all kernels and the model from the SES3D spectral element grid
    to a regular grid, then sums all kernels and saves everything in a
    folder ready for the optimization.
    """
    model = model.lower()
    if model not in config.list_models():
        raise ValueError("Model '%s' not known. Available models:\n%s" % (
            model, "\n\t".join(config.list_models())))
    model_path = config.get_model_path(model)
    boxfile = os.path.join(model_path, "boxfile")

    dims = utils.read_boxfile(boxfile)

    if os.path.exists(output_folder):
        raise ValueError("Folder %s already exists" % output_folder)

    # Write the blockfiles.
    _progress("Creating block files ...")
    blockfile_folder = os.path.join(output_folder, "MODELS", "MODELS_3D")
    os.makedirs(blockfile_folder)

    buf = 0.02
    # Currently each element will be projected to 5 points per dimension.
    utils.write_ses3d_blockfiles(
        output_folder=blockfile_folder,
        x_min=dims["x_min"] - buf * dims["x_range"],
        x_max=dims["x_max"] + buf * dims["x_range"],
        x_count=dims["element_count_x"] * 5,
        y_min=dims["y_min"] - buf * dims["y_range"],
        y_max=dims["y_max"] + buf * dims["y_range"],
        y_count=dims["element_count_y"] * 5,
        # blockfiles are in km whereas boxfiles are in m...
        z_min=dims["z_min"] / 1000.0 - buf * dims["z_range"] / 1000.0,
        z_max=dims["z_max"] / 1000.0 + buf * dims["z_range"] / 1000.0,
        z_count=dims["element_count_z"] * 5,
    )

    # Copy the boxfiles.
    _progress("Copying boxfile ...")
    boxfile_folder = os.path.join(output_folder, "MODELS", "MODELS")
    os.makedirs(boxfile_folder)
    shutil.copy(boxfile, os.path.join(boxfile_folder, "boxfile"))

    _progress("Extracting and compiling the Fortran code ...")
    _check_ses3d_md5()

    src_and_binary_folder = os.path.join(output_folder, "SRC")
    os.makedirs(src_and_binary_folder)

    filenames = ["ses3d_modules.f90", "project_kernel.f90",
                 "project_model.f90"]

    with tarfile.open(SES3D_PATH, "r:gz") as tf:
        for member in tf.getmembers():
            name = member.name
            if "TOOLS/SOURCE" not in name:
                continue
            fname = os.path.basename(name)
            if fname not in filenames:
                continue
            with open(os.path.join(src_and_binary_folder, fname), "wb") as fh:
                fh.write(tf.extractfile(member).read())

    # Now read the project_kernel.f90 file and patch the character limit.
    # XXX: Fix this if SES3D updates eventually.
    fname = os.path.join(src_and_binary_folder, "project_kernel.f90")
    with open(fname, "r") as fh:
        src_code = fh.read()
    with open(fname, "w") as fh:
        src_code = src_code.replace(
            "character(len=140) :: fn_grad, fn_output",
            "character(len=512) :: fn_grad, fn_output")
        src_code = src_code.replace(
            "character(len=60) :: junk, fn",
            "character(len=512) :: junk, fn")
        fh.write(src_code)

    # Patch the project_model.f90 file.
    fname = os.path.join(src_and_binary_folder, "project_model.f90")
    with open(fname, "r") as fh:
        src_code = fh.read()
    with open(fname, "w") as fh:
        src_code = src_code.replace(
            "character(len=60) :: junk, fn, cit",
            "character(len=512) :: junk, fn, cit")
        src_code = src_code.replace(
            "character(len=140) :: fn_model, fn_output",
            "character(len=512) :: fn_model, fn_output")
        fh.write(src_code)

    # Also patch the modules file.
    nx_max = dims["element_count_x"] / dims["processors_in_x"]
    ny_max = dims["element_count_y"] / dims["processors_in_y"]
    nz_max = dims["element_count_z"] / dims["processors_in_z"]

    fname = os.path.join(src_and_binary_folder, "ses3d_modules.f90")
    with open(fname, "r") as fh:
        src_code = fh.read()
    with open(fname, "w") as fh:
        src_code = src_code.replace(
            "integer, parameter :: nx_max=22",
            "integer, parameter :: nx_max=%i" % nx_max)
        src_code = src_code.replace(
            "integer, parameter :: ny_max=27",
            "integer, parameter :: ny_max=%i" % ny_max)

        src_code = src_code.replace(
            "integer, parameter :: nz_max=7",
            "integer, parameter :: nz_max=%i" % nz_max)

        fh.write(src_code)

    # Compile the project_kernels executable.
    source_code_files = ["ses3d_modules.f90", "project_kernel.f90"]
    project_kernel_executable = "project_kernel"
    config.site.compile_fortran_files(
        source_code_files=source_code_files,
        executable=project_kernel_executable, cwd=src_and_binary_folder)

    # Compile the project_model executable.
    source_code_files = ["ses3d_modules.f90", "project_model.f90"]
    project_model_executable = "project_model"
    config.site.compile_fortran_files(
        source_code_files=source_code_files,
        executable=project_model_executable, cwd=src_and_binary_folder)

    _progress("Projecting model ...")

    # Launch it. This will always launch it on the login node/local CPU but
    # that should be ok.
    full_name = os.path.join(os.path.abspath(src_and_binary_folder),
                             os.path.basename(project_model_executable))
    p = subprocess.Popen(full_name,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         stdin=subprocess.PIPE,
                         cwd=os.path.abspath(src_and_binary_folder),
                         bufsize=0)
    out = p.stdout.readline()
    while out:
        line = out
        line = line.strip()

        if verbose:
            print(line)

        if line.startswith("directory for model files"):
            answer = "'%s%s'\n" % (
                os.path.relpath(model_path, src_and_binary_folder),
                os.path.sep)
            p.stdin.write(answer)
            if verbose:
                print(answer.strip())

        elif line.startswith("directory for output:"):
            answer = "'%s%s'\n" % (
                os.path.relpath(os.path.join(output_folder, "MODELS",
                                             "MODELS_3D"),
                                src_and_binary_folder),
                os.path.sep)
            p.stdin.write(answer)
            if verbose:
                print(answer.strip())

        out = p.stdout.readline()

    p.wait()

    kernel_output_dirs = []

    for kernel in kernels:
        _progress("Projecting kernel %s ..." % kernel)
        # Launch it. This will always launch it on the login node/local CPU but
        # that should be ok.
        full_name = os.path.join(os.path.abspath(src_and_binary_folder),
                                 os.path.basename(project_kernel_executable))
        p = subprocess.Popen(full_name,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             stdin=subprocess.PIPE,
                             cwd=os.path.abspath(src_and_binary_folder),
                             bufsize=0)
        out = p.stdout.readline()
        while out:
            line = out
            line = line.strip()

            if verbose:
                print(line)

            if line.startswith("project visco-elastic kernels"):
                answer = "0\n"
                p.stdin.write(answer)
                if verbose:
                    print(answer.strip())

            elif line.startswith("directory for sensitivity densities"):
                answer = "'%s%s'\n" % (
                    os.path.relpath(kernel, src_and_binary_folder),
                    os.path.sep)
                p.stdin.write(answer)
                if verbose:
                    print(answer.strip())

            elif line.startswith("directory for output:"):
                answer = "'%s%s'\n" % (
                    os.path.relpath(output_folder, src_and_binary_folder),
                    os.path.sep)
                p.stdin.write(answer)
                if verbose:
                    print(answer.strip())

            out = p.stdout.readline()

        p.wait()

        final_dir = os.path.join(output_folder, os.path.basename(kernel))
        kernel_output_dirs.append(final_dir)
        os.makedirs(final_dir)
        for filename in glob.glob(os.path.join(output_folder, "gradient_*")):
            shutil.move(filename, os.path.join(
                final_dir, os.path.basename(filename)))
        # The block files are also needed for the later kernel reading.
        shutil.copy(os.path.join(blockfile_folder, "block_x"),
                    os.path.join(final_dir, "block_x"))
        shutil.copy(os.path.join(blockfile_folder, "block_y"),
                    os.path.join(final_dir, "block_y"))
        shutil.copy(os.path.join(blockfile_folder, "block_z"),
                    os.path.join(final_dir, "block_z"))

    _progress("Summing up kernels ...")
    summed_kernel_dir = os.path.join(output_folder, "SUMMED_GRADIENT")
    sum_kernels(kernel_dirs=kernel_output_dirs,
                output_dir=summed_kernel_dir)
    # Also copy the boxfiles to the summed kernels!
    shutil.copy(os.path.join(blockfile_folder, "block_x"),
                os.path.join(summed_kernel_dir, "block_x"))
    shutil.copy(os.path.join(blockfile_folder, "block_y"),
                os.path.join(summed_kernel_dir, "block_y"))
    shutil.copy(os.path.join(blockfile_folder, "block_z"),
                os.path.join(summed_kernel_dir, "block_z"))
