"""
Author: Liran Funaro <liran.funaro@gmail.com>

Copyright (C) 2006-2018 Liran Funaro

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
from configparser import ConfigParser, ExtendedInterpolation

CONFIG_FILE = os.path.abspath(os.path.expanduser("~/.cloudsim"))
DEFAULT_DATA_PATH = os.path.abspath(os.path.expanduser("~/workspace"))

DEFAULT_CONFIG = """
# Cloudsim: data paths of the simulation framework.

[data-path]
# The default data path may be used as a parent for other data paths.
# It is only used by the other paths.
default = %(default)s

# The directory in which we store our simulations' datasets.
%(name)s = ${default}/%(name)s

# The target directory for downloading azure public dataset.
azure_dataset = ${default}/azurepublicdataset

# The location of the framework's interpretation of azure data files.
generated_azure_dataset = ${%(name)s}/azure
""" % dict(name=__name__, default=DEFAULT_DATA_PATH)


def default_config():
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read_string(DEFAULT_CONFIG)
    return config


def init_config_file():
    config = default_config()
    with open(CONFIG_FILE, 'w') as f:
        config.write(f)


def read_config_param(section, key):
    if os.path.isfile(CONFIG_FILE):
        config = ConfigParser(interpolation=ExtendedInterpolation())
        config.read(CONFIG_FILE)
    else:
        config = default_config()

    return config.get(section, key)


def read_config_data_path(key):
    return read_config_param('data-path', key)


framework_data_path = read_config_data_path(__name__)


def get_framework_data_path(*path):
    return os.path.join(framework_data_path, *path)
