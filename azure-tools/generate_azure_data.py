"""
Generates Azure dataset main HDF file.

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
import argparse
import cloudsim
from cloudsim import azure
import pandas as pd
import numpy as np
import subprocess
import requests
import concurrent.futures

from numbers import Number

CWD = os.path.dirname(__file__)


def _fetch_url(entry):
    uri, path = entry
    if not os.path.exists(path):
        r = requests.get(uri, stream=True)
        if r.status_code == 200:
            with open(path, 'wb') as f:
                for chunk in r:
                    f.write(chunk)
    return path


def download_all():
    with open(os.path.join(CWD, 'azure-files.txt'), 'r') as f:
        dataset_urls = f.read().splitlines()

    dataset_entries = [(url, azure.get_azure_data_path(os.path.split(url)[1])) for url in dataset_urls]

    os.makedirs(azure.azure_data_path, exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=256) as executor:
        for e in dataset_entries:
            executor.submit(_fetch_url, e)


def generate_azure_hdf_db():
    d = pd.read_csv(azure.azure_csv_data_path, names=azure.MAIN_CSV_COLS, dtype=azure.MAIN_CSV_DTYPES,
                    compression='gzip')

    os.makedirs(azure.generated_azure_data_path, exist_ok=True)
    with pd.HDFStore(azure.azure_hdf_data_path) as store:
        store['d'] = d


def generate_random_clients_id_file(n_ids: int = 12*2048, min_alive_days: Number = 1):
    azure_data = azure.read_azure_data()
    days_sec = float(60 * 60 * 24)
    s_day = np.array(azure_data['timestamp vm created']) / days_sec
    e_day = np.array(azure_data['timestamp vm deleted']) / days_sec
    ge_day = np.where((e_day - s_day) > min_alive_days)[0]

    matching_vm_ids = azure_data['vm id'][ge_day]
    choose_vm_ids = np.random.choice(matching_vm_ids, n_ids, replace=False)

    os.makedirs(azure.generated_azure_data_path, exist_ok=True)
    with open(azure.azure_vm_ids_path, 'w') as f:
        f.writelines(f"{vm_id}\n" for vm_id in choose_vm_ids)


def generate_cpu_data():
    os.makedirs(azure.azure_cpu_data_path, exist_ok=True)
    cmd_args = './convert-cpu-data.bash', azure.azure_data_path, azure.azure_vm_ids_path, azure.azure_cpu_data_path
    subprocess.run(cmd_args, cwd=CWD, shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and process Azure Public Dataset.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--init', '-i', action='store_true',
                       help='Initialize the framework configuration file.')
    group.add_argument('--download', '-d', action='store_true',
                       help='Download the entire Azure dataset.')
    group.add_argument('--convert', '-c', action='store_true',
                       help="Convert Azure's CVS file to an HDF one.")
    group.add_argument('--choose-random-ids', '-r', action='store_true',
                       help="Select a group of client's IDs to work with (randomly).")
    group.add_argument('--cpu-data', '-g', action='store_true',
                       help="Generates the CPU data files for each of the selected IDs.")
    args = parser.parse_args()

    if args.init:
        print("Initialize config file...")
        cloudsim.init_config_file()
    if args.download:
        print("Downloading Azure's dataset...")
        download_all()
    if args.convert:
        print("Converting CVS to HDF...")
        generate_azure_hdf_db()
    if args.choose_ids:
        print("Selecting clients...")
        generate_random_clients_id_file()
    if args.cpu_data:
        print("Generating CPU data...")
        generate_cpu_data()
