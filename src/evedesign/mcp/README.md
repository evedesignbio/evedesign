# evedesign MCP server

A read-only [MCP](https://modelcontextprotocol.io) server exposing evedesign's model/restraint/
sampler/analyzer catalogue, core interfaces, key concepts and example notebooks to agents.

This is a discovery/reference server, not an execution server. It helps an agent
write correct evedesign Python code by looking things up. Running evedesign code 
is still done semi-manually (agent writes/executes code using information it looks up).

## Tools

| Tool | Purpose |
|------|---------|
| `list_models` | List models/restraints/samplers/etc, filterable by category or interface. |
| `search_models` | Keyword search over the same catalogue. |
| `get_model_info` | Full metadata for one class (citations, GPU/indel handling flags, required extras, etc). |
| `list_interfaces` | The abstract contracts (`Generator`, `Scorer`, `Transformer`, `MutationScorer`, `ConditionalMutationScorer`, `BaseModel`, `SupervisedBaseModel`, `Analyzer`, `ProteinToDnaOptimizer`) with live signatures and docstrings. |
| `get_class_source` | Read the source of an evedesign class or one of its methods. |
| `list_utilities` | Standalone helper functions (foldseek/mmseqs2 search, sequence clustering, `assign_scores_to_instances`). |
| `explain_concept` | Longer-form explanations: System/Entity, instances, indel coding, scoring operations, how to add a new model. |
| `list_examples` / `get_example` | Browse and read the notebooks under `examples/` (source checkout only). |
| `check_environment` | Which optional extras are importable and which torch devices are visible in this process. |

Every entry a discovery tool returns reports whether its optional dependency is currently
available + which pip extras to install if not (see the main [README](../../../README.md)
for the full extras list). Check this before recommending a model to use.

## Setup

### Step 1: Install server

```bash
pip install "evedesign[mcp]"
```

or from a source checkout with `uv`:

```bash
uv sync --extra mcp
```

This installs an `evedesign-mcp` console script.

### Step 2: Register server w/ your agent

#### Claude Code

```bash
claude mcp add evedesign --scope user -- evedesign-mcp
```

`--scope user` registers for every session. From a source checkout without an installed
console script, register `uv run --project /path/to/evedesign evedesign-mcp` instead.

#### Claude Desktop / other JSON-configured clients

```json
{"mcpServers": {"evedesign": {"command": "evedesign-mcp"}}}
```

## Notes

- `list_examples`/`get_example` read `examples/` relative to the repository root, which is not
  part of the pip-installed package. They only work when `evedesign-mcp` runs from a source
  checkout.
- `get_class_source` only reads source within the `evedesign` package itself.
- The server never imports an optional-dependency-only module speculatively at request time,
  beyond what discovery already does at startup. A model reported as `available: false` means 
  its extra isn't installed in the environment this Python process is running in
