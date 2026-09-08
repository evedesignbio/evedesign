"""
MCP server exposing read-only discovery tools over evedesign model/restraint/sampler/
analyzer catalogue, core interfaces, key concepts and example notebooks.
"""
from __future__ import annotations

import importlib
import inspect
from dataclasses import asdict
from typing import Any

from fastmcp import FastMCP

from evedesign.mcp import concepts, environment, examples, interfaces, registry

INSTRUCTIONS = """Look up evedesign models, interfaces, core concepts and examples.

evedesign is a Python framework. These tools help you write correct evedesign code.
No tool here builds a model, scores a sequence, or touches the filesystem/network.

Typical order of operations:
1. list_interfaces for the abstract contracts (Generator, Scorer, Transformer,
   MutationScorer, ConditionalMutationScorer, Analyzer, ProteinToDnaOptimizer, and the shared
   _Core base every one of them extends).
2. list_models / search_models to find a concrete class for a task, filtered by category or
   interface. Entries carry available (whether the optional dependency it needs is
   importable in this process) and extras (the pip extra to install if not).
3. get_model_info for full class metadata, or get_class_source to read its actual
   source when the metadata isn't enough.
4. explain_concept for the conventions that cut across classes: System/Entity, instances,
   the insertion/deletion coding scheme, and when to use score() vs score_conditional() vs
   score_mutants() vs single_mutation_scan().
5. list_examples / get_example for worked end-to-end notebooks by topic (only available when
   running from a cloned source, not from an installed package).

check_environment reports which optional extras are actually importable in this process and
which torch devices (cpu/cuda/mps) are visible (call it before assuming a model is usable).
"""


def build_server() -> FastMCP:
    mcp: FastMCP = FastMCP(name="evedesign", instructions=INSTRUCTIONS)

    @mcp.tool
    def list_models(
        category: str | None = None,
        interface: str | None = None,
        available_only: bool = False,
    ) -> dict[str, Any]:
        """
        List evedesign's models, restraints, samplers, analyzers and codon optimizers

        Parameters
        ----------
        category
            Narrow to one of "model", "restraint", "sampler", "analyzer", "codon_optimizer"
        interface
            Narrow to classes implementing this interface name
        available_only
            If True, drop classes whose dependency isn't currently importable

        Returns
        -------
        Matching entries
        """
        entries = registry.list_all()
        if category is not None:
            known = sorted(registry.CATEGORY_PACKAGES)
            if category not in known:
                return {"ok": False, "error": f"unknown category {category!r}", "categories": known}
            entries = [e for e in entries if e.category == category]
        if interface is not None:
            known_ifaces = sorted(registry.INTERFACES)
            if interface not in known_ifaces:
                return {"ok": False, "error": f"unknown interface {interface!r}", "interfaces": known_ifaces}
            entries = [e for e in entries if interface in e.interfaces]
        if available_only:
            entries = [e for e in entries if e.available]
        return {"ok": True, "models": [asdict(e) for e in entries], "n_total": len(entries)}

    @mcp.tool
    def search_models(query: str, limit: int = 10, available_only: bool = False) -> dict[str, Any]:
        """
        Find models/restraints/samplers/analyzers by keyword, best match first

        Matches against each entry's key, name, category, interfaces, summary and citations.

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
        scored = registry.search(query, limit=limit, available_only=available_only)
        return {
            "ok": True,
            "models": [{**asdict(entry), "score": score} for score, entry in scored],
            "n_total": len(scored),
        }

    @mcp.tool
    def get_model_info(key: str) -> dict[str, Any]:
        """
        Get full metadata for one model/restraint/sampler/analyzer

        Parameters
        ----------
        key
            Class key (e.g. "ESM2") or full dotted class path (e.g.
            "evedesign.models.esm2.ESM2")

        Returns
        -------
        Matching entry
        """
        try:
            entry = registry.get(key)
        except KeyError:
            return {
                "ok": False,
                "error": f"no model with key {key!r}",
                "hint": "call list_models() or search_models() to find a valid key",
            }
        return {"ok": True, **asdict(entry)}

    @mcp.tool
    def list_interfaces() -> dict[str, Any]:
        """
        Describe evedesign's core interfaces

        Covers the shared _Core base contract (name, citations, requires_gpu, positions(),
        ...) plus what each of Generator, Scorer, Transformer, MutationScorer,
        ConditionalMutationScorer, BaseModel and SupervisedBaseModel adds on top of it

        Returns
        -------
        Interface descriptions, see interfaces.describe_all()
        """
        return {"ok": True, **interfaces.describe_all()}

    @mcp.tool
    def get_class_source(class_path: str, member: str | None = None) -> dict[str, Any]:
        """
        Read the source of an evedesign class, or one of its methods

        Parameters
        ----------
        class_path
            Full dotted path within the evedesign package (ex.
            "evedesign.models.esm2.ESM2"), as returned by list_models/search_models/
            get_model_info
        member
            Method/property name; if given, return just that member's source instead of the
            whole class

        Returns
        -------
        Source, source file and line number for the requested class or member
        """
        if not class_path.startswith("evedesign."):
            return {"ok": False, "error": "class_path must be a dotted path within the evedesign package"}

        module_path, _, class_name = class_path.rpartition(".")
        if not module_path or not class_name:
            return {"ok": False, "error": f"not a valid dotted class path: {class_path!r}"}

        try:
            module = importlib.import_module(module_path)
            target: Any = getattr(module, class_name)
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        if member is not None:
            try:
                target = inspect.getattr_static(target, member)
            except AttributeError:
                return {"ok": False, "error": f"{class_path} has no member {member!r}"}
            if isinstance(target, property):
                target = target.fget
            elif isinstance(target, (classmethod, staticmethod)):
                target = target.__func__

        try:
            source = inspect.getsource(target)
            source_file = inspect.getsourcefile(target)
            _, source_line = inspect.getsourcelines(target)
        except (TypeError, OSError) as exc:
            return {"ok": False, "error": f"could not read source: {exc}"}

        return {
            "ok": True,
            "class_path": class_path,
            "member": member,
            "source_file": source_file,
            "source_line": source_line,
            "source": source,
        }

    @mcp.tool
    def list_utilities() -> dict[str, Any]:
        """
        List standalone utility functions worth calling directly

        Covers structure search (foldseek/mmseqs2), sequence clustering, and
        model-agnostic helpers in evedesign.model (ex. assign_scores_to_instances).

        Returns
        -------
        Utility function entries
        """
        return {"ok": True, "utilities": [asdict(entry) for entry in registry.list_utilities()]}

    @mcp.tool
    def explain_concept(topic: str | None = None) -> dict[str, Any]:
        """
        Explain a core evedesign concept in more depth than the model catalogue captures

        Call with no arguments to list available topics. Topics cover System/Entity,
        instances and representations, the insertion/deletion coding convention, the four
        scoring operations (score/score_conditional/score_mutants/single_mutation_scan) and
        how to implement a new model.

        Parameters
        ----------
        topic
            Topic key, as returned by a call with no arguments

        Returns
        -------
        Requested topic's content, or the list of available topics if topic is None
        """
        if topic is None:
            return {"ok": True, "topics": concepts.list_topics()}
        try:
            return {"ok": True, **concepts.get(topic)}
        except KeyError:
            return {"ok": False, "error": f"no topic {topic!r}", "topics": concepts.list_topics()}

    @mcp.tool
    def list_examples() -> dict[str, Any]:
        """
        List the example notebooks under examples/, by topic

        Only available when running from a source checkout of the evedesign repository.

        Returns
        -------
        Name, topic and title for each notebook found
        """
        return examples.list_examples()

    @mcp.tool
    def get_example(name: str) -> dict[str, Any]:
        """
        Get one example notebook's markdown and code cells as text

        Parameters
        ----------
        name
            Notebook name as returned by list_examples(), e.g.
            "scoring_and_transforming_instances/scoring_and_transforming_instances.ipynb"

        Returns
        -------
        Title and cells for the requested notebook
        """
        return examples.get_example(name)

    @mcp.tool
    def check_environment() -> dict[str, Any]:
        """
        Report which of evedesign's optional extras are importable in this process

        Also reports which torch compute devices (cpu/cuda/mps) are visible and any errors
        encountered while building the model registry (which usually means an optional
        dependency is both missing and imported somewhere in that model import chain).

        Returns
        -------
        Environment/installation status, see environment.check()
        """
        return environment.check()

    return mcp


def main() -> None:
    build_server().run(show_banner=False)


if __name__ == "__main__":
    main()
