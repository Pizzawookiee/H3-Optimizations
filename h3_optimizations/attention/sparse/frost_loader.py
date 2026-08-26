'''Load and launch the packaged SM89 FROST cubin through the CUDA Driver API.'''

from __future__ import annotations

import ctypes
import math
import pathlib
import platform
import threading


FROST_ABI = 1
DYNAMIC_SHARED_BYTES = 64 * 1024
THREADS = 128
_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
_PACK_ROOT = pathlib.Path(__file__).resolve().parents[3]
_ARTIFACT = _PACK_ROOT / 'native' / 'frost' / 'h3_frost_bf16_sm89.cubin'
_SYMBOL = _PACK_ROOT / 'native' / 'frost' / 'h3_frost_bf16_sm89.symbol'

_lock = threading.Lock()
_driver = None
_driver_error = None
_functions = {}


class FrostDriverError(RuntimeError):
    pass


def artifact_path():
    return _ARTIFACT


def symbol_path():
    return _SYMBOL


def _library_name():
    return 'nvcuda.dll' if platform.system() == 'Windows' else 'libcuda.so.1'


def _bind(driver):
    p = ctypes.c_void_p
    i = ctypes.c_int
    u = ctypes.c_uint

    driver.cuInit.restype = i
    driver.cuInit.argtypes = [u]
    driver.cuCtxGetCurrent.restype = i
    driver.cuCtxGetCurrent.argtypes = [ctypes.POINTER(p)]
    driver.cuModuleLoadData.restype = i
    driver.cuModuleLoadData.argtypes = [ctypes.POINTER(p), p]
    driver.cuModuleGetFunction.restype = i
    driver.cuModuleGetFunction.argtypes = [ctypes.POINTER(p), p, ctypes.c_char_p]
    driver.cuFuncSetAttribute.restype = i
    driver.cuFuncSetAttribute.argtypes = [p, i, i]
    driver.cuLaunchKernel.restype = i
    driver.cuLaunchKernel.argtypes = [
        p,
        u, u, u,
        u, u, u,
        u,
        p,
        ctypes.POINTER(p),
        p,
    ]
    driver.cuGetErrorName.restype = i
    driver.cuGetErrorName.argtypes = [i, ctypes.POINTER(ctypes.c_char_p)]
    driver.cuGetErrorString.restype = i
    driver.cuGetErrorString.argtypes = [i, ctypes.POINTER(ctypes.c_char_p)]
    return driver


def _error_text(driver, status):
    name = ctypes.c_char_p()
    detail = ctypes.c_char_p()
    driver.cuGetErrorName(int(status), ctypes.byref(name))
    driver.cuGetErrorString(int(status), ctypes.byref(detail))
    name_text = name.value.decode('ascii', 'replace') if name.value else 'CUDA_ERROR_UNKNOWN'
    detail_text = detail.value.decode('utf-8', 'replace') if detail.value else 'no detail'
    return '%s: %s' % (name_text, detail_text)


def _check(driver, status, what):
    if int(status) != 0:
        raise FrostDriverError('%s failed: %s' % (what, _error_text(driver, status)))


def load_driver():
    global _driver, _driver_error
    with _lock:
        if _driver is not None:
            return _driver
        if _driver_error is not None:
            raise FrostDriverError(_driver_error)
        try:
            driver = _bind(ctypes.CDLL(_library_name()))
            _check(driver, driver.cuInit(0), 'cuInit')
        except (OSError, AttributeError, FrostDriverError) as error:
            _driver_error = 'FROST requires the NVIDIA CUDA Driver API: %s' % error
            raise FrostDriverError(_driver_error) from error
        _driver = driver
        return driver


def driver_available():
    try:
        load_driver()
    except FrostDriverError:
        return False
    return True


def unavailable_reason():
    try:
        load_driver()
    except FrostDriverError as error:
        return str(error)
    return None


def _current_context(driver):
    context = ctypes.c_void_p()
    _check(driver, driver.cuCtxGetCurrent(ctypes.byref(context)), 'cuCtxGetCurrent')
    if not context.value:
        raise FrostDriverError('FROST requires an active CUDA context')
    return int(context.value)


def _load_function(driver):
    context = _current_context(driver)
    with _lock:
        cached = _functions.get(context)
        if cached is not None:
            return cached
        if not _ARTIFACT.is_file() or not _SYMBOL.is_file():
            raise FrostDriverError('the packaged FROST SM89 cubin is missing')
        image = ctypes.create_string_buffer(_ARTIFACT.read_bytes())
        module = ctypes.c_void_p()
        _check(
            driver,
            driver.cuModuleLoadData(ctypes.byref(module), ctypes.cast(image, ctypes.c_void_p)),
            'cuModuleLoadData',
        )
        function = ctypes.c_void_p()
        symbol = _SYMBOL.read_text(encoding='ascii').strip().encode('ascii')
        _check(
            driver,
            driver.cuModuleGetFunction(ctypes.byref(function), module, symbol),
            'cuModuleGetFunction',
        )
        _check(
            driver,
            driver.cuFuncSetAttribute(
                function,
                _MAX_DYNAMIC_SHARED_SIZE_BYTES,
                DYNAMIC_SHARED_BYTES,
            ),
            'cuFuncSetAttribute',
        )
        _functions[context] = (module, function)
        return module, function


def _kernel_params(values):
    pointers = [
        ctypes.cast(ctypes.pointer(value), ctypes.c_void_p)
        for value in values
    ]
    return (ctypes.c_void_p * len(pointers))(*pointers)


def launch(q, k, v, output, route, counts, *, scale_log2, stream):
    driver = load_driver()
    _module, function = _load_function(driver)
    sequence_q = int(q.shape[-2])
    sequence_kv = int(k.shape[-2])
    q_tiles = int(route.shape[-2])
    kv_tiles = int(route.shape[-1])
    values = [
        ctypes.c_uint64(int(q.data_ptr())),
        ctypes.c_uint64(int(k.data_ptr())),
        ctypes.c_uint64(int(v.data_ptr())),
        ctypes.c_uint64(int(output.data_ptr())),
        ctypes.c_uint64(int(route.data_ptr())),
        ctypes.c_uint64(int(counts.data_ptr())),
        *(ctypes.c_uint64(0) for _ in range(7)),
        ctypes.c_uint32(q_tiles),
        ctypes.c_uint32(kv_tiles),
        ctypes.c_uint32(kv_tiles),
        ctypes.c_float(float(scale_log2)),
        ctypes.c_uint32(sequence_q),
        ctypes.c_uint32(sequence_kv),
        ctypes.c_uint32(int(q.shape[-1])),
        ctypes.c_uint32(0),
        ctypes.c_float(math.sqrt(int(q.shape[-1]))),
    ]
    params = _kernel_params(values)
    _check(
        driver,
        driver.cuLaunchKernel(
            function,
            q_tiles, int(q.shape[1]), int(q.shape[0]),
            THREADS, 1, 1,
            DYNAMIC_SHARED_BYTES,
            ctypes.c_void_p(int(stream)),
            params,
            None,
        ),
        'cuLaunchKernel',
    )
