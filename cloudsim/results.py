"""
Read results of multiple datasets.

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
import re
import numpy as np
import functools
from cloudsim.dataset import DataSet


def get_result_array(results, *args):
    ret = [functools.reduce(lambda r, d: r[d], args, res) for res in results]
    try:
        return np.array(ret)
    except:
        return ret


def int_or_float(n):
    for number_type in (int, float):
        try:
            return number_type(n)
        except:
            pass
    return n


def get_matching_results(ds_obj: DataSet, name_regexp, param_group=1):
    res_names = ds_obj.get_results_names()
    res_names = [r for r in res_names if len(ds_obj.get_results_index_list(r)) > 0]
    name_re = re.compile(name_regexp)
    matches = [name_re.match(r) for r in res_names]
    matches = [m for m in matches if m is not None]
    match_names = [m.group(0) for m in matches]
    try:
        sort_keys = [m.group(param_group) for m in matches]
    except:
        sort_keys = match_names

    # Try to convert to a number (or a list of numbers)
    try:
        sort_keys = list(map(lambda k: tuple(map(int_or_float, k[-1].split('-'))), sort_keys))
    except:
        pass

    sort_keys = [k if not type(k) in (tuple, list) or len(k) != 1 else k[0] for k in sort_keys]

    return sorted(zip(sort_keys, match_names))


def query_results(ds_obj: DataSet, query):
    res_keys = list(ds_obj.result_store.keys[query])

    res_params = list(map(lambda k: tuple(map(int_or_float, k[-1].split('-'))), res_keys))
    res_params = [k if not type(k) in (tuple, list) or len(k) != 1 else k[0] for k in res_params]

    # num_regexp = re.compile(r'(\d*[.]?\d+(?:[eE][+-]?\d+)?)')
    # res_params = [tuple([int_or_float(n) for n in num_regexp.findall(k)]) for k in res_keys]
    return sorted(zip(res_params, res_keys))


def get_sub_result(ds_obj: DataSet, res_name):
    return query_results(ds_obj, (*res_name, slice(None)))


class Results:
    def __init__(self, ds_obj: DataSet, res_name, verbose=None):
        self.results = ds_obj.read_results(res_name, return_index=False, verbose=verbose)
        self.result_count = len(self.results)
        self.input = {}
        if self.result_count > 0:
            self.input.update(self.results[-1].get('input', {}))

    def get(self, *args):
        return get_result_array(self.results, *args)

    def __getitem__(self, args):
        if type(args) in (tuple, list):
            return self.get(*args)
        else:
            return self.get(args)

    def get_mean(self, *args):
        ret = self.get(*args)
        return np.mean(ret)

    def get_min(self, *args):
        ret = self.get(*args)
        return np.min(ret)


class UnifiedResults:
    def __init__(self, ds_obj: DataSet, result_query, result_count=0):
        self.matching_results = query_results(ds_obj, result_query)
        self.matching_results = self.matching_results[-result_count:]
        self.match_count = len(self.matching_results)
        self.result_count = 0
        self.res_names = []
        self.res_params = []
        self.input = {}
        self.results = []

        for res_param, res_key in self.matching_results:
            cur_res = Results(ds_obj, res_key, verbose=False)
            if cur_res.result_count == 0:
                break
            self.results.append(cur_res)
            self.res_names.append(res_key)
            self.res_params.append(res_param)

        self.result_count = len(self.results)
        self.total_res_count = sum(r.result_count for r in self.results)

        if self.result_count > 0:
            self.input = self.results[-1].input

        if ds_obj.verbose:
            print(ds_obj.name, ":", result_query,
                  "- matching:", self.match_count,
                  "- valid:",    self.result_count,
                  "- total:",    self.total_res_count)

    def get(self, *args):
        ret = [r.get(*args) for r in self.results]
        try:
            return np.array(ret)
        except:
            return ret

    def __getitem__(self, args):
        if type(args) in (tuple, list):
            return self.get(*args)
        else:
            return self.get(args)
