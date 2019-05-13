"""
Job manager.

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
import signal
import inspect
import traceback
import threading
import concurrent.futures
import time
from datetime import datetime
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib import pylab as plt
from matplotlib import colors
from queue import Queue
from collections import OrderedDict
import tempfile
import uuid
from threading import RLock
import pprint
import tabulate
from pygments import highlight
from pygments.lexers import Python3TracebackLexer
from pygments.lexers.data import YamlLexer
from pygments.formatters import Terminal256Formatter

import msgpack
import msgpack_numpy as m

m.patch()


##################################################################
# Default globals
##################################################################

JOB_GLOBAL_KEY = '__cloudsim_job_globals__'
JOB_LOG_FILE_KEY = 'log-file'
JOB_LOG_LOCK_KEY = 'log-file-lock'
JOB_LIST_KEY = 'job-list'
JOB_CONTROL_THREAD_KEY = 'control-threads'
JOB_EXECUTORS_KEY = 'executors'
BATCH_THREAD_NAME = 'batch-jobs'


def get_main_globals():
    return inspect.stack()[-1].frame.f_globals


def get_job_global_dict():
    use_globals = get_main_globals()
    return use_globals.setdefault(JOB_GLOBAL_KEY, {})


def set_job_global_dict(cloudsim_job_globals):
    use_globals = get_main_globals()
    use_globals[JOB_GLOBAL_KEY] = cloudsim_job_globals


def get_global_value(key, default_func):
    cloudsim_job_globals = get_job_global_dict()
    if key not in cloudsim_job_globals:
        cloudsim_job_globals[key] = default_func()
    return cloudsim_job_globals[key]


def get_job_log_file():
    return get_global_value(JOB_LOG_FILE_KEY, lambda: tempfile.mkstemp(prefix='cloudsim-job.log')[1])


def get_job_log_lock():
    return get_global_value(JOB_LOG_LOCK_KEY, lambda: RLock())


def get_job_executors():
    return get_global_value(JOB_EXECUTORS_KEY, lambda: [])


def get_control_threads():
    return get_global_value(JOB_CONTROL_THREAD_KEY, lambda: OrderedDict())


def get_job_list():
    job_list = get_global_value(JOB_LIST_KEY, lambda: OrderedDict())
    for key in list(job_list.keys()):
        try:
            job_element = job_list[key]
            if job_element.start_time is not None and not job_element.running():
                with get_job_log_lock():
                    job_data = job_element.as_dict()
                    del job_list[key]
                    archive_job(job_data)
        except:
            pass
    return job_list


##################################################################
# Jobs Logs API
##################################################################


def archive_job(job_data):
    with get_job_log_lock():
        with open(get_job_log_file(), 'ab') as f:
            f.write(msgpack.packb(job_data, use_bin_type=True))


def handle_job_log_unpacker(handle_func=lambda x: x):
    with get_job_log_lock():
        with open(get_job_log_file(), 'rb') as f:
            unpacker = msgpack.Unpacker(f, raw=False)
            yield from handle_func(unpacker)


def read_job_log():
    return list(handle_job_log_unpacker())


def read_job_errors():
    def yield_if_error(unpacker):
        for j in unpacker:
            if not isinstance(j, dict) or j.get('data_type', None) != 'job' or type(j.get('done', None)) not in (
                    list, tuple):
                yield j
            elif not all(d.get('success', False) for d in j['done']):
                yield j

    return list(handle_job_log_unpacker(yield_if_error))


def print_job_log():
    pprint.pprint(read_job_log())


def print_job_errors():
    for e in read_job_errors():
        if isinstance(e, str):
            print(e)
        elif isinstance(e, dict):
            cur_traceback = e.pop('traceback')
            print(highlight(pprint.pformat(e), YamlLexer(), Terminal256Formatter()))
            if cur_traceback:
                print(highlight(cur_traceback, Python3TracebackLexer(), Terminal256Formatter()))
        else:
            pprint.pprint(e)

        print('=========================================================\n')


##################################################################
# Jobs API
##################################################################


def append_job(job_element):
    job_list = get_job_list()
    job_list[job_element.uuid] = job_element


def get_job(job_uuid=None, job_name=None):
    if job_uuid is None and job_name is None:
        raise ValueError("Must supply uuid or name.")
    job_list = get_job_list()
    if job_uuid is not None:
        return job_list.get(job_uuid, None)
    else:
        for job_element in job_list.values():
            if job_element.name == job_name:
                return job_element


def kill_job(job_uuid=None, job_name=None):
    job_element = get_job(job_uuid, job_name)
    if job_element is not None:
        job_element.kill()


def kill_all_jobs():
    job_list = get_job_list()
    for job_element in job_list.values():
        job_element.kill()


##################################################################
# Control Threads API
##################################################################


def get_control_thread(name):
    control_threads = get_control_threads()
    return control_threads.get(name, None)


def set_control_thread(name, thread_info):
    control_threads = get_control_threads()
    control_threads[name] = thread_info


def is_control_thread_running(name):
    thread_info = get_control_thread(name)
    if thread_info is None:
        return None
    return thread_info['thread'].is_alive()


def terminate_control_thread(name):
    thread_info = get_control_thread(name)
    if thread_info is None:
        return None
    thread_info['event'].set()


def __control_thread_worker(name, target_fn, event, cloudsim_job_globals, args, kwargs):
    set_job_global_dict(cloudsim_job_globals)

    try:
        target_fn(event, *args, **kwargs)
    except:
        job_data = dict(
            data_type='control-thread-error',
            name=name,
            job_func_name=target_fn.__name__,
            job_args=args,
            job_kwargs=kwargs,
            traceback=traceback.format_exc()
        )
        archive_job(job_data)


def start_control_thread(name, thread_fn, args=None, kwargs=None):
    if kwargs is None:
        kwargs = {}
    if args is None:
        args = ()

    event = threading.Event()
    event.clear()

    t = threading.Thread(target=__control_thread_worker,
                         args=(name, thread_fn, event, get_job_global_dict(), args, kwargs))
    t.name = name
    t.daemon = True

    control_thread_data = dict(
        thread=t,
        name=name,
        start_time=time.time(),
        event=event,
        args=args,
        kwargs=kwargs
    )
    set_control_thread(name, control_thread_data)
    t.start()


def __batch_jobs_control_thread(event, queue, cloudsim_job_globals):
    set_job_global_dict(cloudsim_job_globals)

    try:
        while not event.is_set():
            job_func, job_args, job_kwargs = queue.get()
            if event.is_set():
                return
            # noinspection PyBroadException
            try:
                job_element = job_func(*job_args, **job_kwargs)
                job_element.join()
            except Exception:
                job_data = dict(
                    data_type='batch-error',
                    job_func_name=job_func.__name__,
                    job_args=job_args,
                    job_kwargs=job_kwargs,
                    traceback=traceback.format_exc()
                )
                archive_job(job_data)
    finally:
        queue.queue.clear()


def get_batch_control_thread():
    thread_info = get_control_thread(BATCH_THREAD_NAME)
    if thread_info is None or not thread_info['thread'].is_alive():
        return None

    return thread_info


def start_batch_control_thread():
    thread_info = get_batch_control_thread()
    if thread_info is not None:
        return thread_info

    event = threading.Event()
    queue = Queue()
    event.clear()

    t = threading.Thread(name=BATCH_THREAD_NAME, target=__batch_jobs_control_thread,
                         args=(event, queue, get_job_global_dict()), daemon=True)

    thread_info = dict(
        thread=t,
        name=BATCH_THREAD_NAME,
        start_time=time.time(),
        event=event,
        queue=queue
    )
    set_control_thread(BATCH_THREAD_NAME, thread_info)
    t.start()
    return thread_info


def add_batch_job(job_func, *job_args, **job_kwargs):
    thread_info = start_batch_control_thread()
    if thread_info is None:
        raise RuntimeError("Batch control thread could not be started")
    queue = thread_info['queue']
    queue.put((job_func, job_args, job_kwargs))


def get_batch_jobs_list():
    thread_info = get_batch_control_thread()
    if thread_info is None:
        return
    return list(thread_info['queue'].queue)


def terminate_batch_control_thread():
    thread_info = get_control_thread(BATCH_THREAD_NAME)
    if thread_info is None:
        return None
    thread_info['event'].set()
    thread_info['queue'].put((None, None, None))


##################################################################
# Executors API
##################################################################


def _executor_pid_fetcher_worker(_):
    return os.getpid()


def get_executor_pids(executor, max_workers=12):
    ret = []
    while len(ret) != max_workers:
        ret = set(executor.map(_executor_pid_fetcher_worker, range(max_workers*2)))
    return tuple(ret)


def destroy_job_executors():
    job_executors = get_job_executors()
    for cur_executor, w, pids in job_executors:
        # cur_executor.shutdown(False)
        dead_count = 0
        kill_errors = []
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)  # or signal.SIGKILL
            except ProcessLookupError:
                dead_count += 1
            except Exception as e:
                kill_errors.append((pid, str(e)))

        if dead_count == len(pids):
            print("All workers (%d) were already dead" % dead_count)
        elif dead_count > 0:
            print("%d were already dead" % dead_count)
        if len(kill_errors) > 0:
            print("Other errors:", ", ".join(map("[{0}] {1}".format, kill_errors)))

    job_executors.clear()


def get_job_executor(max_workers=12, destroy_old=False):
    if destroy_old:
        destroy_job_executors()

    # Workaround: Don't queue any job so canceling will be quicker
    # concurrent.futures.process.EXTRA_QUEUED_CALLS = 0
    cur_executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

    pids = get_executor_pids(cur_executor, max_workers)

    job_executors = get_job_executors()
    job_executors.append((cur_executor, max_workers, pids))
    return cur_executor


##################################################################
# Create Simulations API
##################################################################


def _job_executor_func(job_worker_fn, index, job_input):
    try:
        return index, True, job_worker_fn(job_input)
    except:
        return index, False, traceback.format_exc()


class JobElement:
    def __init__(self, name, worker_fn, inputs=None,
                 input_generator_fn=None, input_generator_args=None, max_workers=24):
        self.uuid = uuid.uuid4()
        self.name = name
        self.worker_fn = worker_fn
        self.inputs = inputs
        self.input_generator_fn = input_generator_fn
        self.input_generator_args = input_generator_args or ()
        if inputs is None and input_generator_fn is None:
            raise ValueError("Must supply inputs or input generator function.")
        self.max_workers = max_workers

        self.executor = None
        self.futures = []
        self.done = []
        self.start_time = None
        self.status = 'created'

        try:
            self.worker_name = self.worker_fn.__name__
        except:
            self.worker_name = 'unknown function'
        self.job_size = len(self.inputs)
        append_job(self)

    def running(self):
        if self.status == 'started':
            return True
        elif self.status == 'running':
            ret = any(f.running() for f in self.futures)
            if not ret:
                self.status = 'done'
            return ret
        else:
            return False

    def start(self):
        self.status = 'started'
        self.start_time = time.time()
        self.executor = get_job_executor(max_workers=self.max_workers, destroy_old=True)

        if self.inputs is None:
            f = self.executor.submit(self.input_generator_fn, self.input_generator_args)
            try:
                self.inputs = f.result()
            except:
                self.status = 'done'
                self.append_done('input-error', None, False, traceback.format_exc())

        for i, x in enumerate(self.inputs):
            f = self.executor.submit(_job_executor_func, self.worker_fn, i, x)
            self.futures.append(f)
            f.add_done_callback(self.on_job_finish)

        self.status = 'running'

    def run(self):
        self.start_time = time.time()
        for i, x in enumerate(self.inputs):
            result = self.worker_fn(x)
            self.append_done(i, False, True, result)

    def on_job_finish(self, f):
        if f.cancelled():
            index, cancelled, success, result = None, True, False, None
        else:
            index, success, result = f.result()
            cancelled = False
        self.append_done(index, cancelled, success, result)

    def append_done(self, index, cancelled, success, result):
        self.done.append(dict(
            finish_time=time.time(),
            index=index,
            cancelled=cancelled,
            success=success,
            result=result
        ))

    def kill(self):
        not_cancelled = 0
        for f in self.futures:
            if not f.done() and not f.cancel():
                not_cancelled += 1

        if not_cancelled > 0:
            print(not_cancelled, "jobs cannot be cancelled.")

    def join(self):
        _ = [f.result() for f in self.futures]
        return

    def get_failed_idx(self, timeout=0):
        for i, f in enumerate(self.futures):
            if not f.done() or f.cancelled():
                continue
            try:
                f.result(timeout)
            except:
                yield i

    def __str__(self):
        return "%s(%s, %s)" % (self.__class__.__name__, self.name, self.worker_name)

    def __repr__(self):
        return str(self)

    def as_dict(self):
        return dict(
            data_type='job',
            name=self.name,
            worker_name=self.worker_name,
            start_time=self.start_time,
            done=self.done,
            job_size=self.job_size
        )


##################################################################
# Progress API
##################################################################


def plot_job_progress(job_data, is_running=False, subfig=False, short_ticks=False):
    done = job_data['done']
    start = job_data['start_time']
    name = job_data['name']

    x = np.subtract([d['finish_time'] for d in done], start)
    c = ['green' if d['success'] else 'red' for d in done]

    if not subfig:
        fig, ax = plt.subplots(figsize=(10, 1))
    else:
        fig, ax = plt.gcf(), plt.gca()

    ax.scatter([0], [0.5], marker=9, s=200, lw=0, c='#00CC00', alpha=0.5)

    x_max = 0
    if len(x) > 0:
        ax.scatter(x, [0.5]*len(x), c=c, marker='|', lw=1)
        x_max = x[-1]

    if is_running:
        x_max = time.time()-start
        ax.scatter([x_max], [0.5], marker=8, s=200, lw=0, c='#CC0000', alpha=0.5)

    if not subfig:
        fig.autofmt_xdate()

    if short_ticks:
        fmt = lambda t, p: seconds_to_text_short(t)
    else:
        fmt = lambda t, p: seconds_to_text(t)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))

    ax.yaxis.set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # ax.spines['top'].set_visible(False)

    if not is_running:
        ax.spines['bottom'].set_color('#808080')
        ax.spines['top'].set_color('#808080')
        ax.set_facecolor('#f2f2f2')
    else:
        ax.spines['bottom'].set_color('green')
        ax.spines['top'].set_color('green')
        ax.set_facecolor('#fafff9')

    ax.spines['bottom'].set_linestyle(':')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_linestyle(':')
    ax.spines['top'].set_linewidth(2)

    ax.xaxis.set_ticks_position('bottom')
    ax.get_yaxis().set_ticklabels([])
    ax.set_xlim((0, x_max*1.002))
    ax.set_ylim(0, 1)
    plt.title(name, loc='left', fontsize=8)
    plt.tight_layout()


def append_job_line(cur_time, job_lines, name, is_running, start_time, last_update,
                    items_total, items_done, items_running, items_cancelled, items_failed, items_queued):
        end_time = cur_time
        if not is_running and last_update is not None:
            end_time = last_update

        str_start_time = "N/A"
        str_run_time = "N/A"
        str_last_update = "N/A"
        str_throughput = "N/A"
        str_eta = "N/A"

        if start_time is not None:
            str_run_time = seconds_to_text(end_time - start_time)

        if start_time is not None and last_update is not None and items_done > 0:
            work_time = last_update - start_time
            throughput = items_done / work_time

            remaining_items = items_total - items_done - items_cancelled
            eta = remaining_items / throughput
            eta -= (cur_time - last_update)
            eta_time = cur_time + eta

            str_throughput = throughput_to_text(throughput)
            if is_running:
                str_eta = "%s (%s)" % (seconds_to_text(eta),
                                       datetime.fromtimestamp(eta_time).strftime('%H:%M:%S - %d.%m.%Y'))

        if start_time is not None:
            str_start_time = datetime.fromtimestamp(start_time).strftime("%H:%M:%S")

        if last_update is not None:
            str_last_update = datetime.fromtimestamp(last_update).strftime("%H:%M:%S")

        job_lines.append([name, is_running, items_total, items_done, items_failed, items_running,
                          items_queued, items_cancelled, str_start_time, str_run_time, str_last_update, str_throughput,
                          str_eta])


def print_job_status(plot_progress=True, show_only_failed=False, show_finished_count=6, progress_columns=1):
    job_list = get_job_list()
    job_log = read_job_log()
    control_threads = get_control_threads()
    print("Total ran jobs:", len(job_log))

    cur_time = time.time()
    job_log = sorted(filter(lambda e: e.get('data_type', None) == 'job', job_log), key=lambda e: -e['start_time'])
    job_log = job_log[:max(0, show_finished_count)]

    job_lines = []

    for job_element in job_list.values():
        last_update = job_element.done[-1]['finish_time'] if len(job_element.done) > 0 else None

        items_total = len(job_element.futures)
        items_done = len([f for f in job_element.futures if f.done() and not f.cancelled()])
        items_running = len([f for f in job_element.futures if f.running()])
        items_cancelled = len([f for f in job_element.futures if f.cancelled()])
        items_failed = len(list(job_element.get_failed_idx()))
        items_queued = items_total - items_done - items_running - items_cancelled

        append_job_line(cur_time, job_lines,
                        job_element.name, job_element.running(), job_element.start_time, last_update,
                        items_total, items_done, items_running, items_cancelled, items_failed, items_queued)

    for job_data in job_log:
        last_update = job_data['done'][-1]['finish_time'] if len(job_data['done']) > 0 else None

        items_total = job_data['job_size']
        items_done = len([f for f in job_data['done'] if f['success']])
        items_running = 0
        items_cancelled = len([f for f in job_data['done'] if f['cancelled']])
        items_failed = len([f for f in job_data['done'] if not f['success'] and not f['cancelled']])
        items_queued = 0

        if show_only_failed and items_done == items_total:
            continue

        append_job_line(cur_time, job_lines,
                        job_data['name'], False, job_data['start_time'], last_update,
                        items_total, items_done, items_running, items_cancelled, items_failed, items_queued)

    job_headers = ["Name", "Alive", "T", "D", "F", "R", "Q", "C",
                   "Started", "Work Time", "Last Update", "Throughput", "ETA"]
    job_df = pd.DataFrame(job_lines, columns=job_headers)

    def highlight_alive(s):
        if s['Alive']:
            return ['background-color: yellow' for _ in s]
        else:
            return ['' for _ in s]

    def background_gradient(s, s_min, s_max, cmap='PuBu', low=0, high=0):
        rng = s_max - s_min
        norm = colors.Normalize(s_min - (rng * low), s_max + (rng * high))
        normed = norm(s.values)
        c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
        return ['background-color: %s' % color for color in c]

    cm = sns.light_palette("green", as_cmap=True)
    job_style_df = job_df.style.apply(highlight_alive, axis=1)
    if len(job_lines) > 0:
        df_max = job_df[["T", "D", "F", "R", "Q", "C"]].values.max()
        job_style_df.apply(background_gradient, cmap=cm, s_min=0, s_max=df_max, low=0, high=1,
                           subset=["T", "D", "F", "R", "Q", "C"])

    control_thread_order = sorted(
        [(ct['thread'].is_alive(), ct['start_time'], ct['name'], ct['event']) for ct in control_threads.values()],
        key=lambda k: (-k[0], -k[1]))

    cont_lines = []

    for is_running, start_time, name, event in control_thread_order:
        str_start_time = datetime.fromtimestamp(start_time).strftime("%H:%M:%S")
        if is_running:
            str_work_time = seconds_to_text(time.time() - start_time)
        else:
            str_work_time = 'N/A'

        cont_lines.append((name, is_running, str_start_time, str_work_time, event.is_set()))

    cont_headers = ['Name', 'Alive', 'Started', 'Work Time', 'Event State']
    print(tabulate.tabulate(cont_lines, cont_headers, tablefmt='fancy_grid'))

    if plot_progress:
        n_running = len(job_list)
        h = int(np.ceil((len(job_log))/progress_columns)) + n_running
        plt.figure(figsize=(14, h))
        for i, job_element in enumerate(job_list.values()):
            plt.subplot(h, 1, 1 + i)
            plot_job_progress(job_element.as_dict(), is_running=True, subfig=True, short_ticks=False)

        for i, job_data in enumerate(job_log):
            plt.subplot(h, progress_columns, 1 + i + (n_running*progress_columns))
            plot_job_progress(job_data, subfig=True, short_ticks=True)

    return job_style_df


################################################################################
# Text helper functions
################################################################################

def seconds_to_text(time_seconds):
    if time_seconds <= 0:
        return '0'
    elif time_seconds < 60:
        return "%.2f seconds" % time_seconds
    elif time_seconds < (60 * 60):
        seconds, minutes = math.modf(time_seconds / 60)
        return "%d:%02d minutes" % (minutes, seconds*60)
    elif time_seconds < (60 * 60 * 24):
        minutes, hours = math.modf(time_seconds / (60 * 60))
        return "%d:%02d hours" % (hours, minutes*60)
    else:
        hours = time_seconds / (60 * 60)
        days = int(hours / 24)
        hours = hours - (24 * days)
        minutes, hours = math.modf(hours)
        return "%d days, %d:%02d hours" % (days, hours, minutes*60)


def seconds_to_text_short(time_seconds):
    if time_seconds <= 0:
        return '0'
    elif time_seconds < 60:
        return "%.2fs" % time_seconds
    elif time_seconds < (60 * 60):
        seconds, minutes = math.modf(time_seconds / 60)
        return "%d:%02dm" % (minutes, seconds*60)
    elif time_seconds < (60 * 60 * 24):
        minutes, hours = math.modf(time_seconds / (60 * 60))
        return "%d:%02dh" % (hours, minutes*60)
    else:
        hours = time_seconds / (60 * 60)
        days = int(hours / 24)
        hours = hours - (24 * days)
        minutes, hours = math.modf(hours)
        return "%dd, %d:%02dh" % (days, hours, minutes*60)


def throughput_to_text(throughput):
    if throughput > 1:
        return "%.2f items/second" % throughput
    elif throughput * 60 > 1:
        return "%.2f items/minute" % (throughput * 60)
    elif throughput * 60 * 60 > 1:
        return "%.2f items/hour" % (throughput * 60 * 60)
    else:
        return "%.2f items/day" % (throughput * 24 * 60 * 60)


########################################################
# Tests
########################################################

def test_worker(*parameters):
    time.sleep(5)
    return parameters
