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
        if isinstance(value, str):
            data[key] = os.path.expanduser(os.path.expandvars(value))
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

        if self.site_name not in available_sites:
            raise ValueError("Site '%s' is not available. Available sites: %s"
                             % sorted(available_sites.keys()))

        self.site = available_sites[self.site_name](
            working_directory=self.root_working_dir)

    @property
    def model_dir(self):
        return os.path.join(self.root_working_dir, "__MODELS")

    def list_models(self):
        return [_i for _i in os.listdir(self.model_dir)
                if os.path.isdir(os.path.join(self.model_dir, _i))]

    def list_runs(self):
        return [_i for _i in os.listdir(self.root_working_dir) if
                os.path.isdir(os.path.join(self.root_working_dir, _i)) and not
                _i.startswith("_")]

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
@click.option("--lpd", type=int, default=4, show_default=True,
              help="Degree of the Lagrange polynomials for the simulation")
@click.option("--fw-lpd", type=int, default=1, show_default=True,
              help="Polynomial degree for storing the forward field")
@click.option("--pml-count", type=int, default=3, show_default=True,
              help="Number of PMLs at each boundary.")
@click.option("--pml-limit", type=int, default=10000, show_default=True,
              help="Number of time steps for which PMLs are enabled.")
@click.argument("input_files_folder", type=click.Path())
@pass_config
def run(config, input_files_folder, lpd, fw_lpd, pml_count, pml_limit):
    """
    Run a simulation for the chosen input files.
    """
    _progress("Parsing and checking input files ...")
    input_files = SES3DInputFiles(input_files_folder)

    # Get a new working directory.
    cwd = config.site.get_new_working_directory()
    _progress("Initializing run '%s' ..." % os.path.basename(cwd))

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


@cli.command()
@click.argument("model_folder",
                type=click.Path(exists=True, readable=True, resolve_path=True,
                                dir_okay=True))
@pass_config
def add_model(config, model_folder):
    """
    Add a model to SES3D ctrl.
    """
    model_name = os.path.basename(model_folder).lower()
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
