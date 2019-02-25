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
import numpy as np
from cloudsim import read_config_data_path
import pandas as pd

azure_data_path = read_config_data_path('azure_dataset')
generated_azure_data_path = read_config_data_path('generated_azure_dataset')


def get_azure_data_path(*path):
    return os.path.join(azure_data_path, *path)


def get_generated_azure_data_path(*path):
    return os.path.join(generated_azure_data_path, *path)


azure_csv_data_path = get_azure_data_path("vmtable.csv.gz")
azure_hdf_data_path = get_generated_azure_data_path('vmtable.h5')
azure_vm_ids_path = get_generated_azure_data_path('vm-id-set.txt')
azure_cpu_data_path = get_generated_azure_data_path('cpu-data')

MAIN_CSV_COLS = ('vm id', 'subscription id', 'deployment id', 'timestamp vm created', 'timestamp vm deleted',
                 'max cpu', 'avg cpu', 'p95 max cpu', 'vm category', 'vm virtual core count', 'vm memory (gb)')
MAIN_CSV_FMT = (str, str, str, np.uint32, np.uint32, np.float64, np.float64, np.float64, str, np.uint32, np.float64)
MAIN_CSV_DTYPES = dict(zip(MAIN_CSV_COLS, MAIN_CSV_FMT))

CPU_DATA_COLS = ['timestamp', 'min-cpu', 'max-cpu', 'avg-cpu']
CPU_DATA_FMT = [np.uint64, np.float64, np.float64, np.float64]
CPU_DATA_DTYPES = dict(zip(CPU_DATA_COLS, CPU_DATA_FMT))


def get_azure_cpu_data_path(*path):
    return os.path.join(azure_cpu_data_path, *path)


def read_azure_data():
    with pd.HDFStore(azure_hdf_data_path, mode='r') as store:
        return store['d']


def get_available_vm_ids():
    with open(azure_vm_ids_path, 'r') as f:
        return f.read().splitlines()


def get_vm_id_cpu_data_path(vm_id):
    vm_id = vm_id.replace("/", "_")
    return get_azure_cpu_data_path(vm_id)


def get_vm_id_cpu_data(vm_id):
    vm_id_path = get_vm_id_cpu_data_path(vm_id)
    return pd.read_csv(vm_id_path, names=CPU_DATA_COLS, dtype=CPU_DATA_DTYPES)
