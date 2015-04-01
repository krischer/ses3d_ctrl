#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import click
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile

from .sites import available_sites
from .ses3d_input_files import SES3DInputFiles


CONFIG_FILE_PATH = os.path.expanduser("~/.ses3d_ctrl.json")

DEFAULT_CONFIG = {
    "root_working_dir": "~/ses3d_ctrl_working_directory",
    "adjoint_dir": "/tmp/SES3D_TEMP_ADJOINT",
    "site_name": "local_gcc"
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
    with io.open(CONFIG_FILE_PATH, "w") as fh:
        json.dump(DEFAULT_CONFIG, fh,
                  sort_keys=True,
                  indent=4,
                  separators=(",", ": "))


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


def _progress(msg):
    """
    Consistent and "pretty" progress logs.
    """
    click.echo(click.style(" -> ", fg="red"), nl=False)
    click.echo(click.style(msg, fg="green"))


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
@click.argument("input_files_folders", type=click.Path(), nargs=-1)
@pass_config
def run(config, model, input_files_folders, lpd, fw_lpd, pml_count, pml_limit):
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

    # Get a new working directory.
    cwd = config.site.get_new_working_directory()
    run_name = os.path.basename(cwd)
    _progress("Initializing run '%s' ..." % run_name)

    # Untar SES3D to that working directory.
    _progress("Extracting SES3D ...")
    # Use checksum to assert the file is correct.
    with io.open(SES3D_PATH, "rb") as fh:
        md5 = hashlib.md5(fh.read()).hexdigest()

    if md5 != SES3D_MD5_CHECKSUM:
        raise ValueError("md5 of the SES3D archive is not %s" %
                         SES3D_MD5_CHECKSUM)

    with tarfile.open(SES3D_PATH, "r:gz") as fh:
        for member in fh.getmembers():
            if not member.isreg():
                continue
            member.name = os.path.sep.join(member.name.split(os.path.sep)[1:])
            fh.extract(member, cwd)

    _progress("Compiling SES3D ...")
    s = input_files.setup
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
        adjoint_output_folder=os.path.abspath(config.adjoint_dir))

    _progress("Copying model ...")
    model_folder = os.path.join(cwd, "MODELS", "MODELS")
    if os.path.exists(model_folder):
        shutil.rmtree(model_folder)
    shutil.copytree(config.get_model_path(model), model_folder)

    _progress("Launching SES3D on %i cores ..." % 4)
    config.site.run_ses3d(job_name=run_name, cpu_count=4)


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
    Print the status of any runs.
    """
    for run in config.list_runs():
        status = config.site.get_status(run)
        click.echo("%s\tStatus: %s\t\t updated %s" % (
                   run, status["status"], status["time"].humanize()))

@cli.command()
@pass_config
@click.argument("job-number", type=str)
def cancel(config, job_number):
    if job_number not in config.list_runs():
        raise ValueError("Job not known")
    config.site.cancel_job(job_number)
