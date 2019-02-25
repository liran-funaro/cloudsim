"""
A set of generated data items with similar characteristics.

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
import re
import sys
import copy
import pprint
import pandas as pd
from contextlib import suppress
from itertools import chain

from nesteddict import NestedDictFS
from cloudsim import get_framework_data_path
from cloudsim import sim_data
from cloudsim import job


##################################################################
# Data Sets
##################################################################

data_simulation_name_format = "%s[%s]"


def convert_to_nice_filename(fname):
    """ Produce a nice (valid) filename """
    return re.sub(r'[\s"\'()[\]{}]', '', str(fname))


def alter_dataset(parent_metadata, name_fmt, *alter_args, generator_func=None, cache_size=None, **alter_kwargs):
    """ Generate a ney dataset on the basis on another dataset with some alterations """
    if isinstance(parent_metadata, DataSet):
        if generator_func is None:
            generator_func = parent_metadata.generator_func
        if cache_size is None:
            cache_size = parent_metadata.store.cache.get_size()
        parent_metadata = parent_metadata.metadata

    metadata = copy.deepcopy(parent_metadata)
    for k, v in chain(alter_args, alter_kwargs.items()):
        cur_m = metadata
        if type(k) in (list, tuple):
            for kk in k[:-1]:
                cur_m = cur_m[kk]
            k = k[-1]
        if isinstance(v, dict) and isinstance(cur_m[k], dict):
            cur_m[k].update(v)
        else:
            cur_m[k] = v

    prefix = metadata.get('prefix', None)
    if isinstance(prefix, str):
        prefix = prefix.strip('- ')
        name_fmt = "%s-%s" % (prefix, name_fmt)

    metadata['name'] = name_fmt.format(**metadata)
    return DataSet(metadata, generator_func=generator_func, cache_size=cache_size, verbose=True)


class DataSet:
    __slots__ = ('metadata', 'name', 'folder_name', 'generator_func', 'verbose', 'store', 'q')
    DATA_KEY = 'data'
    RESULT_KEY = 'result'
    INFO_KEY = 'info'

    def __init__(self, metadata, generator_func=None, cache_size=None, verbose=None):
        """
        :param metadata:  A dict of metadata or a DataSet object.
        :param generator_func: A function that can generate a new data item given this metadata.
        :param cache_size: The cache size to use.
        :param verbose: If verbose, the class might print information.
        """
        if isinstance(metadata, self.__class__):
            for attr in self.__slots__:
                if attr == 'q':
                    continue
                setattr(self, attr, getattr(metadata, attr))
            if generator_func is not None:
                self.generator_func = generator_func
            if verbose is not None:
                self.verbose = verbose
        else:
            if isinstance(metadata, dict):
                self.metadata = copy.deepcopy(metadata)
                self.folder_name = convert_to_nice_filename(self.metadata['name'])
                folder_path = get_framework_data_path(self.folder_name)
                self.store = NestedDictFS(folder_path, mode='c', cache_size=cache_size)
            elif isinstance(metadata, str):
                self.folder_name = convert_to_nice_filename(metadata)
                folder_path = get_framework_data_path(self.folder_name)
                self.store = NestedDictFS(folder_path, mode='w', cache_size=cache_size)
                self.metadata = self.read_info()
            else:
                raise ValueError("Must supply metadata, folder name or a DataSet. Actual type is %s" % type(metadata))

            self.name = self.folder_name

            self.metadata['name'] = self.folder_name
            self.generator_func = generator_func
            self.verbose = verbose if verbose is not None else True

        if self.verbose:
            self.q = self.__class__(self, verbose=False)
        else:
            self.q = self

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, pprint.pformat(self.metadata, indent=2))

    def __repr__(self):
        return str(self)

    @property
    def folder_path(self):
        return self.store.data_path

    @property
    def meta(self):
        return self.metadata

    @property
    def result_store(self):
        return self.store.get_child(self.RESULT_KEY)

    @property
    def data_store(self):
        return self.store.get_child(self.DATA_KEY)

    def get_data_store(self, i):
        return self.data_store.get_child(i)

    def load_data(self, i, mode='r'):
        return sim_data.SimulationData(self.get_data_store(i), verbose=self.verbose, mode=mode)

    def write_info(self):
        self.store[self.INFO_KEY] = self.metadata

    def read_info(self):
        return self.store[self.INFO_KEY]

    def clear_cache(self):
        self.store.clear_cache()

    def generate_data_item(self, i):
        if self.generator_func is None:
            raise RuntimeError("No generator function supplied.")
        new_metadata = dict(self.metadata, index=i)
        sd = sim_data.SimulationData(self.get_data_store(i), metadata=new_metadata, verbose=self.verbose, mode='c')
        self.generator_func(sd)
        sd.save()
        return sd

    def create_data(self, data_count):
        self.write_info()

        for i in range(data_count):
            self.generate_data_item(i)

    def parallel_create_data(self, data_count, max_workers=12):
        self.write_info()
        sim_name = self.get_job_name('create-data')
        inputs = [(self.folder_name, i, self.generator_func) for i in range(data_count)]
        sim_obj = job.JobElement(sim_name, _create_data_worker, inputs, max_workers=max_workers)
        sim_obj.start()
        return sim_obj

    def verify_data_seeds(self):
        idx = self.get_data_index_list()
        seeds = [self.load_data(i).seed for i in idx]
        assert len(idx) == len(set(seeds))

    @staticmethod
    def _files_to_index_list(file_list):
        # basename: fetch only file name
        # splitext[0]: fetch only file name without ext
        # int: convert to int
        # sorted: returns a sorted list
        return sorted(int(os.path.splitext(os.path.basename(f))[0]) for f in file_list)

    def get_data_index_list(self):
        return self._files_to_index_list(self.data_store.keys())

    #####################################################################################
    # Results
    #####################################################################################

    def get_results_sim_names(self):
        return self.result_store.keys()

    def get_single_result_store(self, name):
        return self.result_store.get_child(name)

    def read_result_file(self, name, i):
        return self.get_single_result_store(name)[i]

    def get_results_index_list(self, name):
        return self._files_to_index_list(self.get_single_result_store(name).keys())

    def write_result_file(self, name, i, result):
        cur_res = self.get_single_result_store(name).get_child(i)
        for k, v in result.items():
            if not isinstance(v, dict):
                cur_res[k] = v
            else:
                for sub_k, sub_v in v.items():
                    cur_res[k, sub_k] = sub_v

    def get_results_names(self):
        results = []
        for d, sub_store in self.result_store.child_items():
            sub_results = [os.path.join(d, dd) for dd in sub_store.child_keys()]
            if sub_results:
                results.extend(sub_results)
            else:
                results.append(d)
        return results

    def is_result_file_exist(self, name, i):
        if not self.store.child_exists(self.RESULT_KEY):
            return False
        res_store = self.store[self.RESULT_KEY]
        if not res_store.child_exists(name):
            return False
        sim_store = res_store[name]
        return sim_store.exists(i)

    def iter_results(self, name, return_index=True, verbose=None):
        res_ids = self.get_results_index_list(name)
        if (verbose is None and self.verbose) or verbose is True:
            print(len(res_ids), 'samples')
        for i in res_ids:
            res = self.read_result_file(name, i)
            if return_index:
                yield i, res
            else:
                yield res

    def read_results(self, name, return_index=True, verbose=None):
        return list(self.iter_results(name, return_index=return_index, verbose=verbose))

    #####################################################################################
    # Simulations
    #####################################################################################

    def get_job_name(self, name):
        if type(name) in (list, tuple):
            name = ":".join(map(str, name))
        return data_simulation_name_format % (name, self.folder_name)

    def create_job(self, name, worker_fn, skip_existing=False, max_workers=12,
                   datasets_list=None, datasets_interval=None, start_job=True, args=None, kwargs=None):
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        sim_name = self.get_job_name(name)
        data_list = list(self.get_data_index_list())
        if datasets_list is not None:
            data_list = datasets_list
        elif datasets_interval is not None:
            data_list = data_list[datasets_interval[0]:datasets_interval[-1]]
        inputs = [(self.metadata, name, worker_fn, i, skip_existing, args, kwargs) for i in data_list]
        cur_job = job.JobElement(sim_name, _dataset_job_worker, inputs, max_workers=max_workers)
        if start_job:
            cur_job.start()
        return cur_job

    def create_job_with_generator(self, name, worker_fn, input_generator_fn, skip_existing=False, max_workers=12,
                                  datasets_list=None, datasets_interval=None, start_job=True, args=None, kwargs=None):
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        sim_name = self.get_job_name(name)
        data_list = list(self.get_data_index_list())
        if datasets_list is not None:
            data_list = datasets_list
        elif datasets_interval is not None:
            data_list = data_list[datasets_interval[0]:datasets_interval[-1]]
        generator_args = data_list, self.metadata, name, worker_fn, input_generator_fn, skip_existing, args, kwargs
        cur_job = job.JobElement(sim_name, _dataset_job_worker, input_generator_fn=_input_generator_worker,
                                 input_generator_args=generator_args, max_workers=max_workers)
        if start_job:
            cur_job.start()
        return cur_job

    def add_batch_job(self, job_func, *job_args, **job_kwargs):
        return job.add_batch_job(_job_starter, self.name, job_func, *job_args, **job_kwargs)

    def plot_job_progress(self, name):
        sim_name = self.get_job_name(name)
        job.plot_job_progress(sim_name)

    def kill_job(self, name):
        sim_name = self.get_job_name(name)
        job.kill_job(sim_name)


#######################################################################################################################
# Job Interface for dataset related jobs
#######################################################################################################################


def get_batch_jobs_list():
    jobs = job.get_batch_jobs_list()
    if jobs is None or len(jobs) < 1:
        print("Batch job queue is empty.")
        return
    print("Total remaining batch jobs:", len(jobs))

    ret = []
    for job_func, job_args, job_kwargs in jobs:
        line = {
            'func name': 'N/A',
            'dataset name': 'N/A',
        }
        with suppress(Exception):
            line['dataset name'] = job_args[0]
        with suppress(Exception):
            line['func name'] = job_args[1].__name__.replace('_', '-')

        if job_kwargs is not None:
            line.update({k.replace('_', '-'): v for k, v in job_kwargs.items()})
        ret.append(line)

    df = pd.DataFrame(ret)
    df.fillna('', inplace=True)

    try:
        cols = ['dataset name', 'func name', 'sim-name', 'sim-param', 'sub-name']
        cols = [*cols, *list(set(df.columns) - set(cols))]
        return df[cols]
    except Exception as e:
        print("Could not order columns: %s" % e, file=sys.stderr)
        return df


#######################################################################################################################
# Job Workers
#######################################################################################################################


def _job_starter(metadata, job_func, *args, **kwargs):
    ds_obj = DataSet(metadata, verbose=False)
    return job_func(ds_obj, *args, **kwargs)


def _dataset_job_worker(sim_input):
    metadata, name, worker_fn, index, skip_existing, args, kwargs = sim_input
    ds_obj = DataSet(metadata, verbose=False)
    if skip_existing and ds_obj.is_result_file_exist(name, index):
        return False
    sd = ds_obj.load_data(index)
    result = worker_fn(ds_obj, index, sd, *args, **kwargs)
    ds_obj.write_result_file(name, index, result)
    return True


def _create_data_worker(sim_input):
    metadata, index, generator_func = sim_input
    ds_obj = DataSet(metadata, generator_func=generator_func, verbose=False)
    ds_obj.generate_data_item(index)


def _input_generator_worker(sim_input):
    data_list, metadata, name, worker_fn, input_generator_fn, skip_existing, args, kwargs = sim_input
    args, kwargs = input_generator_fn(*args, **kwargs)
    return [(metadata, name, worker_fn, i, skip_existing, args, kwargs) for i in data_list]
