"""
Version-neutral wire codec.

The two sides run in environments that differ on purpose, so numpy, pandas and
torch versions differ too and pickle is not an option. Everything crosses as
msgpack, with extension types for the few library objects that need them:

    System, SystemInstance   via the library's own serialize()/deserialize()
    pandas.DataFrame/Series  index levels, column labels and one column at a time
                             (numeric/bool as arrays, strings and objects as lists),
                             rebuilt natively on the far side; other dtypes are refused
    numpy.ndarray            dtype + shape + raw bytes
    tuple                    so NamedTuples like Mutation come back as tuples

Anything else that msgpack cannot encode raises CodecError naming the type,
rather than being pickled quietly.
"""

from typing import Any

import msgpack
import numpy as np

_SYSTEM, _INSTANCE, _FRAME, _SERIES, _ARRAY, _TUPLE = 1, 2, 3, 4, 5, 6


class CodecError(TypeError):
    pass


def _values(values) -> Any:
    """One column or index level: a numpy array if numeric/bool, else a list of plain values."""
    array = np.asarray(values)
    if array.dtype.kind in "biuf":
        return array
    if array.dtype.kind in "OUS":
        return array.tolist()
    raise CodecError(f"cannot send a pandas column of dtype {values.dtype}; convert it to numbers or strings")


def _index_to_obj(index) -> list:
    return [list(index.names), [_values(index.get_level_values(i)) for i in range(index.nlevels)]]


def _index_from_obj(obj):
    import pandas as pd

    names, levels = obj
    if len(levels) == 1:
        return pd.Index(levels[0], name=names[0])
    return pd.MultiIndex.from_arrays(levels, names=names)


def _frame_to_bytes(frame) -> bytes:
    columns = [_values(frame.iloc[:, i]) for i in range(frame.shape[1])]
    return dumps([_index_to_obj(frame.index), _index_to_obj(frame.columns), columns])


def _frame_from_bytes(data: bytes):
    import pandas as pd

    index, columns, values = loads(data)
    frame = pd.DataFrame(dict(enumerate(values)), index=_index_from_obj(index))
    frame.columns = _index_from_obj(columns)
    return frame


def _default(obj: Any) -> Any:
    from evedesign.system import System, SystemInstance

    if isinstance(obj, System):
        return msgpack.ExtType(_SYSTEM, dumps(obj.serialize()))
    if isinstance(obj, SystemInstance):
        return msgpack.ExtType(_INSTANCE, dumps(obj.serialize()))
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return obj.tolist()
        return msgpack.ExtType(
            _ARRAY, msgpack.packb([obj.dtype.str, list(obj.shape), obj.tobytes()])
        )
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, tuple):
        return msgpack.ExtType(_TUPLE, dumps(list(obj)))
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    # subclasses of builtins (str enums, OrderedDict, ...) under strict_types
    for base in (bool, int, float, str, bytes, dict, list):
        if isinstance(obj, base):
            return base(obj)

    import pandas as pd

    if isinstance(obj, pd.DataFrame):
        return msgpack.ExtType(_FRAME, _frame_to_bytes(obj))
    if isinstance(obj, pd.Series):
        return msgpack.ExtType(_SERIES, _frame_to_bytes(obj.to_frame(name=obj.name)))

    from .proxy import RemoteModel

    if isinstance(obj, RemoteModel):
        raise CodecError(
            f"cannot pass the remote model {obj!r} into another session: its state lives in its "
            f"own worker. Construct the parent locally with it (e.g. GibbsSampler(scorers=[...])), "
            f"or co-locate parent and children with open_built(target, spec_with_nested_specs, system)."
        )
    raise CodecError(
        f"cannot send a {type(obj).__module__}.{type(obj).__qualname__} across the "
        f"process boundary; only msgpack types, System, SystemInstance, numpy arrays "
        f"and pandas objects are supported"
    )


def _ext_hook(code: int, data: bytes) -> Any:
    if code == _SYSTEM:
        from evedesign.system import System

        return System.deserialize(loads(data))
    if code == _INSTANCE:
        from evedesign.system import SystemInstance

        return SystemInstance.deserialize(loads(data))
    if code == _ARRAY:
        dtype, shape, raw = msgpack.unpackb(data)
        return np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape).copy()
    if code == _TUPLE:
        return tuple(loads(data))
    if code == _FRAME:
        return _frame_from_bytes(data)
    if code == _SERIES:
        frame = _frame_from_bytes(data)
        return frame.iloc[:, 0].rename(frame.columns[0])
    raise CodecError(f"unknown extension type {code}")


def dumps(obj: Any) -> bytes:
    # tuples go through _default (strict_types) so they survive as tuples
    return msgpack.packb(obj, default=_default, strict_types=True, use_bin_type=True)


def loads(data: bytes) -> Any:
    return msgpack.unpackb(data, ext_hook=_ext_hook, raw=False, strict_map_key=False)
