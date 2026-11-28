from contextlib import contextmanager
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import os
from typing import Callable, Sequence, TypeVar, Mapping, Any
import numpy as np
from evedesign.types import StatusCallback

T = TypeVar("T")


def ensure_sequence(x: T | Sequence[T]) -> Sequence[T]:
    if isinstance(x, Sequence):
        return x
    else:
        return [x]


@contextmanager
def model_param_context(
    load_func: Callable[[], None],
    delete_func: Callable[[], None],
    keep_model: bool
):
    try:
        load_func()
        yield
    finally:
        if not keep_model:
            delete_func()
        else:
            pass

def status_start(
    status_callback: StatusCallback | None, message: str | None = None
):
    if status_callback is not None:
        status_callback("running", None, message)

def status_done(
    status_callback: StatusCallback | None, message: str | None = None
):
    if status_callback is not None:
        status_callback("done", None, message)

def status_progress(
    status_callback: StatusCallback | None, progress: float
):
    if status_callback is not None:
        status_callback("running", progress, None)

def shorten(text: str, max_len=50):
    return (text[:max_len] + "...") if len(text) > max_len else text

def str_to_np_char_view(x: Sequence[str]):
    """
    Quickly transform a list of strings into a numpy character
    array (much faster than np.array([list(s) for s in x])
    and return a view

    Parameters
    ----------
    x
        List of equal length strings (not checked here)

    Returns
    -------
    2D character array
    """
    x_np = np.array(
        x, dtype=np.str_
    )

    return x_np.view(
        "U1"
    ).reshape(
        (x_np.size, -1)
    )

def map_array(x: np.ndarray, map_: Mapping) -> np.ndarray:
    """
    Efficiently map elements of a numpy array

    Parameters
    ----------
    x
        Array to be mapped
    map_
        Mapping to be applied to individual elements
        (to cover potentially missing values, use a defaultdict)

    Returns
    -------

    """
    return np.vectorize(
        map_.__getitem__
    )(x)

def index_map(options: list[Any], default_option: Any = None, default_value: int | None = None):
    """
    Create mapping from a list of options to integer indices (typically used with map_array above)

    Parameters
    ----------
    options
        Discrete list of hashable options (e.g. strings)
    default_option
        If specified, return the index associated with this option by default if mapping something that is not
        contained in options. Takes precendence over default_value.
    default_value
        If specified, return this number as default if mapping something that is not
        contained in options. Will be overriden by default_option if also specified.

    Returns
    -------

    """
    mapping = {
        symbol: idx for idx, symbol in enumerate(options)
    }

    if default_option is None and default_value is None:
        return mapping
    else:
        if default_option is not None:
            default = mapping[default_option]
        else:
            default = default_value

        return defaultdict(lambda: default, mapping)


def available_cpus() -> int:
    # respects cgroup/CPU-affinity limits (Slurm, Docker, k8s); falls back gracefully
    if hasattr(os, "process_cpu_count"):      # Python 3.13+
        return os.process_cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):      # Linux
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


_POOLS = threading.local()


def score_all(scorers: Sequence[Any], method: str, *args: Any, **kwargs: Any) -> list[Any]:
    """
    [getattr(s, method)(*args, **kwargs) for s in scorers], run concurrently.

    With scorers in other processes, a serial loop costs the sum of their
    latencies per step; concurrently it costs the slowest. Threads suffice:
    proxies wait on I/O, and numpy-heavy local scorers release the GIL. Order
    is preserved and the first failure propagates.

    Each calling thread has its own pool, reused from call to call: concurrent
    samplers do not queue behind each other, and a nested call (made from a
    pool thread) has a pool of its own, so it cannot deadlock. A pool goes
    when its thread ends.
    """
    if len(scorers) <= 1:
        return [getattr(s, method)(*args, **kwargs) for s in scorers]
    if getattr(_POOLS, "size", 0) < len(scorers):
        _POOLS.pool = ThreadPoolExecutor(max_workers=len(scorers), thread_name_prefix="score_all")
        _POOLS.size = len(scorers)
    futures = [_POOLS.pool.submit(getattr(s, method), *args, **kwargs) for s in scorers]
    return [f.result() for f in futures]
