from contextlib import contextmanager
from collections import defaultdict
from typing import Callable, Sequence, TypeVar, Mapping
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
    return (text[:max_len] + "...") if len(text) > 50 else text

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

def index_map(options: list[any], default_option: any = None, default_value: int | None = None):
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
