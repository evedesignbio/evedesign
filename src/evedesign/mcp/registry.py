"""
Discovery of evedesign's models, restraints, samplers, analyzers and codon optimizers.
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from evedesign import model as _model
from evedesign.analysis import Analyzer
from evedesign.nucleotides import ProteinToDnaOptimizer

# interfaces a concrete class is checked against
INTERFACES: dict[str, type] = {
    "Generator": _model.Generator,
    "Scorer": _model.Scorer,
    "Transformer": _model.Transformer,
    "MutationScorer": _model.MutationScorer,
    "ConditionalMutationScorer": _model.ConditionalMutationScorer,
    "Analyzer": Analyzer,
    "ProteinToDnaOptimizer": ProteinToDnaOptimizer,
}

# category -> package holding its top level user-facing modules
CATEGORY_PACKAGES: dict[str, str] = {
    "model": "evedesign.models",
    "restraint": "evedesign.restraints",
    "sampler": "evedesign.samplers",
    "analyzer": "evedesign.analyzers",
    "codon_optimizer": "evedesign.codons",
}

# full class path -> pip extra(s) needed for full functionality, beyond the base
# install. Not derivable from the code itself (kept in sync with pyproject.toml
# [project.optional-dependencies])
EXTRAS_BY_CLASS: dict[str, tuple[str, ...]] = {
    "evedesign.models.eve.EVE": ("eve",),
    "evedesign.models.evmutation2.EVmutation2": ("evmutation2",),
    "evedesign.models.evcouplings.EVcouplingsMeanField": ("evcouplings",),
    "evedesign.models.evcouplings.EVcouplingsPLM": ("evcouplings",),
    "evedesign.models.esm2.ESM2": ("esm2",),
    "evedesign.models.mpnn.LigandMPNN": ("mpnn",),
    "evedesign.models.oasis_humanness.OASisHumanness": ("promb",),
    "evedesign.models.boltzfold.BoltzFoldTransformer": ("boltz2fold", "boltz2fold-cuda"),
    "evedesign.models.boltzgen.BoltzGenGenerator": ("boltzgen",),
    "evedesign.models.supervised.GpytorchModel": ("gpytorch",),
    "evedesign.analyzers.sequence_space.SequenceSpaceUMAP": ("umap",),
}

# handwritten fallback for classes whose module cannot be safely imported without an
# optional dependency present
_STATIC_FALLBACK: dict[str, list[dict[str, Any]]] = {
    "evedesign.models.mpnn": [
        {
            "key": "LigandMPNN",
            "category": "model",
            "interfaces": ("Generator", "Scorer", "MutationScorer", "ConditionalMutationScorer"),
            "name": "LigandMPNN",
            "citations": ("doi: 10.1038/s41592-025-02626-1",),
            "requires_gpu": False,
            "supports_gpu": True,
            "supports_cpu_parallel": False,
            "supports_gpu_parallel": False,
            "requires_target": True,
            "requires_fixed_length": True,
            "handles_deletions": False,
            "handles_insertions": False,
            "summary": "LigandMPNN/ProteinMPNN inverse folding.",
        },
    ],
    "evedesign.models.boltzgen": [
        {
            "key": "BoltzGenGenerator",
            "category": "model",
            "interfaces": ("Generator",),
            "name": "BoltzGen",
            "citations": ("doi.org/10.1101/2025.11.20.689494",),
            "requires_gpu": True,
            "supports_gpu": True,
            "supports_cpu_parallel": False,
            "supports_gpu_parallel": True,
            "requires_target": False,
            "requires_fixed_length": False,
            "handles_deletions": False,
            "handles_insertions": True,
            "summary": "BoltzGen de novo structure/sequence generation.",
        },
    ],
}


@dataclass(frozen=True)
class ModelEntry:
    key: str
    class_path: str | None
    category: str
    interfaces: tuple[str, ...]
    name: str | None
    citations: tuple[str, ...]
    available: bool
    requires_gpu: bool | None
    supports_gpu: bool | None
    supports_cpu_parallel: bool | None
    supports_gpu_parallel: bool | None
    requires_target: bool | None
    requires_fixed_length: bool | None
    handles_deletions: bool | None
    handles_insertions: bool | None
    extras: tuple[str, ...]
    summary: str | None
    source_file: str | None = None
    source_line: int | None = None
    introspection_limited: bool = False


def _static(cls: type, attr: str) -> Any | None:
    """
    Read a class attribute without triggering property/descriptor evaluation

    Parameters
    ----------
    cls
        Class to read the attribute from
    attr
        Attribute name

    Returns
    -------
    Attribute value, or None if absent or only computable on an instance (a property or
    method)
    """
    try:
        value = inspect.getattr_static(cls, attr)
    except AttributeError:
        return None
    if isinstance(value, property) or inspect.isfunction(value) or inspect.ismethod(value):
        return None
    return value


def _summary(obj: Any) -> str | None:
    doc = inspect.getdoc(obj)
    if not doc:
        return None
    return doc.strip().splitlines()[0].strip()


def _source_location(cls: type) -> tuple[str | None, int | None]:
    try:
        source_file = inspect.getsourcefile(cls)
        _, line = inspect.getsourcelines(cls)
    except (TypeError, OSError):
        return None, None
    return source_file, line


def _iter_top_level_modules(package_name: str) -> list[str]:
    package = importlib.import_module(package_name)
    return [
        module_info.name
        for module_info in pkgutil.iter_modules(package.__path__, prefix=f"{package_name}.")
        if not module_info.ispkg
    ]


def _entry_for_class(cls: type, category: str) -> ModelEntry:
    interfaces = tuple(sorted(name for name, iface in INTERFACES.items() if issubclass(cls, iface)))
    citations = _static(cls, "citations") or []
    available = _static(cls, "available")
    return ModelEntry(
        key=cls.__qualname__,
        class_path=f"{cls.__module__}.{cls.__qualname__}",
        category=category,
        interfaces=interfaces,
        name=_static(cls, "name"),
        citations=tuple(citations),
        available=bool(available) if available is not None else True,
        requires_gpu=_static(cls, "requires_gpu"),
        supports_gpu=_static(cls, "supports_gpu"),
        supports_cpu_parallel=_static(cls, "supports_cpu_parallel"),
        supports_gpu_parallel=_static(cls, "supports_gpu_parallel"),
        requires_target=_static(cls, "requires_target"),
        requires_fixed_length=_static(cls, "requires_fixed_length"),
        handles_deletions=_static(cls, "handles_deletions"),
        handles_insertions=_static(cls, "handles_insertions"),
        extras=EXTRAS_BY_CLASS.get(f"{cls.__module__}.{cls.__qualname__}", ()),
        summary=_summary(cls),
        source_file=(loc := _source_location(cls))[0],
        source_line=loc[1],
    )


def _entry_for_fallback(module_name: str, data: dict[str, Any]) -> ModelEntry:
    fields = dict(data)
    key = fields.pop("key")
    class_path = f"{module_name}.{key}"
    return ModelEntry(
        key=key,
        class_path=class_path,
        available=False,
        extras=EXTRAS_BY_CLASS.get(class_path, ()),
        introspection_limited=True,
        **fields,
    )


@lru_cache(maxsize=1)
def discover() -> tuple[tuple[ModelEntry, ...], dict[str, str]]:
    """
    Discover every concrete model/restraint/sampler/analyzer/codon optimizer class

    Returns
    -------
    entries
        All discovered classes, sorted by category then key
    import_errors
        Module path mapped to error message, for top-level modules that raised on import
        (which implies an optional dependency is both missing and imported unconditionally
        somewhere in that module's import chain)
    """
    entries: list[ModelEntry] = []
    import_errors: dict[str, str] = {}

    for category, package_name in CATEGORY_PACKAGES.items():
        for module_name in _iter_top_level_modules(package_name):
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:  # defensive: see module notes above
                import_errors[module_name] = f"{type(exc).__name__}: {exc}"
                for fallback in _STATIC_FALLBACK.get(module_name, []):
                    entries.append(_entry_for_fallback(module_name, {**fallback, "category": category}))
                continue

            for _, cls in inspect.getmembers(module, inspect.isclass):
                if cls.__module__ != module.__name__:
                    continue  # re-exported/imported name, not defined in this module
                if inspect.isabstract(cls):
                    continue  # base class / mixin, not directly usable
                if not any(issubclass(cls, iface) for iface in INTERFACES.values()):
                    continue
                entries.append(_entry_for_class(cls, category))

    entries.sort(key=lambda e: (e.category, e.key))
    return tuple(entries), import_errors


def list_all() -> list[ModelEntry]:
    return list(discover()[0])


def import_errors() -> dict[str, str]:
    return dict(discover()[1])


def get(key: str) -> ModelEntry:
    """
    Look up one entry by its short key (class name) or full dotted class path

    Parameters
    ----------
    key
        Class name (e.g. "ESM2") or full dotted class path (e.g.
        "evedesign.models.esm2.ESM2")

    Returns
    -------
    Matching entry

    Raises
    ------
    KeyError
        If no entry matches key, or if a short key matches more than one entry
    """
    matches = [e for e in list_all() if e.key == key or e.class_path == key]
    if not matches:
        raise KeyError(key)
    if len(matches) > 1:
        exact = [e for e in matches if e.class_path == key]
        if len(exact) == 1:
            return exact[0]
        raise KeyError(f"{key!r} is ambiguous: {[e.class_path for e in matches]}")
    return matches[0]


_STOPWORDS = frozenset({"a", "an", "the", "of", "for", "to", "and", "or", "with", "that", "is", "in", "on"})


def search(query: str, limit: int = 10, available_only: bool = False) -> list[tuple[int, ModelEntry]]:
    """
    Score every entry against query by term frequency over its key/name/summary/etc.

    Note: this is a discovery aid over a catalogue of a few dozen entries, not a ranking
    problem worth a synonym table, so scoring is kept deliberately simple.

    Parameters
    ----------
    query
        Free-text search query
    limit
        Maximum number of entries to return
    available_only
        If True, drop entries whose optional dependency isn't currently importable

    Returns
    -------
    Matching entries with their score, best match first
    """
    terms = [t for t in re.split(r"[^a-z0-9]+", query.lower()) if t and t not in _STOPWORDS]
    if not terms:
        return []

    scored = []
    for entry in list_all():
        if available_only and not entry.available:
            continue
        haystack = " ".join(
            filter(
                None,
                [
                    entry.key,
                    entry.name or "",
                    entry.category,
                    " ".join(entry.interfaces),
                    entry.summary or "",
                    " ".join(entry.citations),
                ],
            )
        ).lower()
        score = sum(haystack.count(term) for term in terms)
        if score:
            scored.append((score, entry))

    scored.sort(key=lambda pair: (-pair[0], pair[1].key))
    return scored[:limit]


# utility modules worth making visible to agents
UTILITY_MODULES: tuple[str, ...] = (
    "evedesign.model",
    "evedesign.tools.foldseek",
    "evedesign.tools.mmseqs2",
    "evedesign.tools.structure_mapping",
    "evedesign.analyzers.sequence_clustering",
)


@dataclass(frozen=True)
class UtilityEntry:
    name: str
    module: str
    signature: str | None
    summary: str | None


def list_utilities() -> list[UtilityEntry]:
    """
    List the public top-level functions of UTILITY_MODULES

    Returns
    -------
    Utility function entries, sorted by module then name
    """
    entries: list[UtilityEntry] = []
    for module_name in UTILITY_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for name, fn in inspect.getmembers(module, inspect.isfunction):
            if fn.__module__ != module.__name__ or name.startswith("_"):
                continue
            try:
                signature = str(inspect.signature(fn))
            except (TypeError, ValueError):
                signature = None
            entries.append(
                UtilityEntry(name=name, module=module_name, signature=signature, summary=_summary(fn))
            )
    entries.sort(key=lambda e: (e.module, e.name))
    return entries
