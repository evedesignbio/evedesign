# AGENTS.md

Orientation for coding agents working in this repository. Consider 
registering the MCP server in [src/evedesign/mcp](src/evedesign/mcp) 

## What this is

evedesign is a Python framework for biomolecular sequence design. Design problems are described
as Systems, which you can run models against (generation, scoring, embedding, restraints) 
through a set of shared interfaces. See the top-level [README.md](README.md) for a project overview 
and the full model/restraint/sampler/analyzer table.

Code is used by importing evedesign and composing its classes, as shown in `examples/*.ipynb`

## Repository layout

```
src/evedesign/
  system.py       # System, Entity, SystemInstance, EntityInstance, Mutation, Mutant (core data model)
  model.py        # _Core, Generator, Scorer, Transformer, MutationScorer, ConditionalMutationScorer, BaseModel
  analysis.py     # Analyzer interface
  nucleotides.py  # ProteinToDnaOptimizer interface
  models/         # model implementations (ESM-2, EVmutation2, LigandMPNN, Boltz-2, BoltzGen, EVE, ...)
  restraints/     # sequence/structure/physicochemical restraints (implement BaseModel plus Scorer-family interfaces)
  samplers/       # Gibbs sampler (implements Generator)
  analyzers/      # sequence-space projections etc. (implement Analyzer)
  codons/         # protein to DNA codon optimization (implements ProteinToDnaOptimizer)
  tools/          # standalone helpers: foldseek/mmseqs2 structure search, structure-to-entity mapping
  mcp/            # MCP discovery server for this codebase, see mcp/README.md
examples/         # example notebooks, one topic per subdirectory
tests/            # pytest, mirrors src/ layout loosely
```

## Core data model (`evedesign.system`)

- `System` is an ordered list of `Entity` objects (use `Protein`/`DNA`/`RNA`/`Ligand`, not the
  generic `Entity`, unless there is a reason to). It describes the design problem: reference
  sequence, first_index numbering, MSA/homologs, known structures, length/symmetry constraints.
- `SystemInstance` is an ordered list of `EntityInstance`, one concrete realization of a `System`
  (a specific sequence plus embedding plus structural models). `System.rep_to_instance()` builds
  an instance from each entity's own `rep`.
- `Mutation = (entity, pos, ref, to)`; `Mutant = Sequence[Mutation]`. Positions are in the
  entity's `first_index` numbering, not zero-based array indices. Deletions are coded `to = GAP`;
  insertions are coded `ref = ""`, `to` set to lowercase alphabet symbols, positioned directly
  after `pos` (or at `first_index - 1` for an insertion before the first residue), the same scheme
  used by A3M alignment format.
- Both `System`/`Entity` and `SystemInstance`/`EntityInstance` provide `serialize()`/
  `deserialize()` to and from JSON-compatible dicts.

Call the MCP server `explain_concept("system_and_entities")`, `explain_concept("instances_and_representations")`
or `explain_concept("mutations_and_indels")` for more details.

## Core interfaces (`evedesign.model`)

Every model implements `BaseModel` plus one or more of:

- `Generator.generate(...)`: sample new sequences.
- `Scorer.score(instances)`: raw logits/likelihoods per instance, comparable within one call.
- `Transformer.transform(instances)`: map instances to another representation (e.g. embedding).
- `MutationScorer`: `score_mutants()` (relative scores for mutants of one instance) and
  `single_mutation_scan()` (all single substitutions across one instance in one call).
- `ConditionalMutationScorer.score_conditional(...)`: one position's substitutions batched across
  many instances (used by e.g. Gibbs sampling).

`_Core` (the shared base every one of the above extends) declares `name`, `citations`,
`requires_target`, `requires_fixed_length`, `handles_deletions`, `handles_insertions`,
`requires_gpu`, `supports_gpu`, `supports_cpu_parallel`, `supports_gpu_parallel`, and
`positions()`/`valid_positions()`.

Call `list_interfaces()` via the MCP server (or read `evedesign/model.py` directly) for exact
signatures and docstrings before implementing a new model, restraint, sampler or analyzer.

## Conventions used throughout concrete implementations

1. Contract attributes are plain class attributes, not computed properties (ex.
   `requires_gpu: bool = False`, `name: str = "ESM2"`, `citations: list[str] = [...]`)
2. Optional heavy dependencies are guarded at module scope:
   ```python
   try:
       import torch
       ...
       IMPORT_AVAILABLE = True
   except ImportError:
       IMPORT_AVAILABLE = False
   ```
   with `available = IMPORT_AVAILABLE` set on the class, checked before doing real work (see
   `evedesign/models/esm2.py` or `evedesign/codons/dnachisel.py`). Keep any import that needs the
   optional dependency inside that guarded block.
3. Add a new model's class to the appropriate table in the top-level `README.md`, and to
   `evedesign.mcp.registry.EXTRAS_BY_CLASS` if it needs an optional dependency.

## Development

```bash
uv sync                              # base install
uv sync --extra esm2 --extra mpnn    # add specific optional extras as needed - see README.md
uv run pytest                        # run tests
```

Tests requiring an optional extra are marked accordingly (`boltz2fold`, `boltzgen`, `eve`,
`esm2`, `evcouplings`, `mpnn`; see `[tool.pytest.ini_options]` in `pyproject.toml`). To run 
tests without extras:

```bash
uv run pytest -m "not boltz2fold and not boltzgen and not eve and not esm2 and not evcouplings and not mpnn"
```

## MCP server

`src/evedesign/mcp` is a read-only discovery server (list/search models, read interface
contracts and class source, explain core concepts, browse example notebooks, check which extras
are installed) intended for use by coding agents. It does not execute any evedesign code. See
[src/evedesign/mcp/README.md](src/evedesign/mcp/README.md) for setup and the full tool list.
