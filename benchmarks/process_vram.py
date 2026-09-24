'''Dedicated GPU memory of one process, sampled from Windows performance counters.

This is the figure Task Manager shows per process.  NVML cannot report it on
consumer cards, which run under WDDM, so ``nvidia-smi`` offers only the
whole-card total.  That total carries the desktop compositor and every other
application, and on this class of machine it moves by hundreds of MiB between
arms as Windows evicts desktop surfaces.  Reading the ComfyUI server process
directly removes that baseline instead of subtracting an estimate of it.

The counter includes the CUDA context, driver overhead and allocator
fragmentation, so it reads above torch allocator peaks.  Windows only; import
fails cleanly elsewhere and callers fall back to whole-card sampling.
'''

from __future__ import annotations

import re
import threading
import time

import win32pdh

COUNTER = r'\GPU Process Memory(*)\Dedicated Usage'
INSTANCE = re.compile(r'^pid_(\d+)_luid_([0-9a-fx]+)_([0-9a-fx]+)_phys', re.IGNORECASE)
MIB = 1024.0 * 1024.0


class ProcessVramSampler:
    '''Poll one process's dedicated bytes on its busiest adapter.'''

    def __init__(self, pid, interval_ms=50):
        self.pid = int(pid)
        self.interval = interval_ms / 1000.0
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _read(self, query, counter):
        win32pdh.CollectQueryData(query)
        entries = win32pdh.GetFormattedCounterArray(counter, win32pdh.PDH_FMT_LARGE)
        adapters = {}
        for instance, value in entries.items():
            match = INSTANCE.match(instance)
            if match is None or int(match.group(1)) != self.pid:
                continue
            luid = match.group(2) + '_' + match.group(3)
            adapters[luid] = adapters.get(luid, 0) + int(value or 0)
        return max(adapters.values(), default=0)

    def _run(self):
        query = win32pdh.OpenQuery()
        try:
            counter = win32pdh.AddCounter(query, COUNTER)
            # A wildcard counter needs one collection before values are valid.
            win32pdh.CollectQueryData(query)
            while not self._stop.is_set():
                try:
                    self.samples.append(self._read(query, counter))
                except Exception:
                    pass
                self._stop.wait(self.interval)
        finally:
            win32pdh.CloseQuery(query)

    def read_once(self):
        query = win32pdh.OpenQuery()
        try:
            counter = win32pdh.AddCounter(query, COUNTER)
            win32pdh.CollectQueryData(query)
            time.sleep(0.05)
            return self._read(query, counter)
        finally:
            win32pdh.CloseQuery(query)

    def start(self):
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None

    def peak_mib(self):
        return max(self.samples) / MIB if self.samples else None


def listener_pid(port):
    '''PID owning a local TCP listener, or None.'''
    import psutil

    for connection in psutil.net_connections(kind='tcp'):
        if (connection.status == psutil.CONN_LISTEN and connection.laddr
                and connection.laddr.port == int(port)):
            return connection.pid
    return None
