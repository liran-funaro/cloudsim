"""
Wrapper for storing client's simulation data.
Uses NestedDictFS as an underlying storage mechanism.

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
import pprint
import numpy as np

from nesteddict import NestedDictFS


#####################################################################
# Simulation Data
#####################################################################

class SimulationData:
    def __init__(self, data_path=None, metadata=None, verbose=False, mode='r'):
        self.verbose = verbose
        self.internal = {}
        self.store = NestedDictFS(data_path, mode=mode)
        self.metadata = None

        if metadata and mode == 'c':
            self.store['metadata'] = metadata
            self.metadata = metadata

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    #####################################################################
    # Accessors
    #####################################################################

    @property
    def data_path(self):
        return self.store.data_path

    @property
    def meta(self):
        if self.metadata:
            return self.metadata

        self.metadata = self.store.get('metadata', {})
        return self.metadata

    @property
    def data(self):
        return self.store.get_child('data')

    @property
    def dist_data(self):
        return self.store.get_child('dist')

    @property
    def init_data(self):
        return self.store.get_child('init')

    def write_metadata(self):
        if not self.metadata:
            return
        self.store['metadata'] = self.metadata

    def save(self):
        self.write_metadata()

    @property
    def n(self):
        return self.meta['n']

    @property
    def ndim(self):
        return self.meta['ndim']

    @property
    def seed(self):
        seed = self.meta.get('seed', None)
        if seed is None:
            seed = int.from_bytes(os.urandom(4), byteorder="big")
            np.random.seed(seed)
            self.meta['seed'] = seed
            self.write_metadata()
        return seed

    def init_seed(self):
        np.random.seed(self.seed)

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, pprint.pformat(self.meta, indent=2))

    def __repr__(self):
        return str(self)

    def print(self):
        pprint.pprint(self.meta, indent=2)
