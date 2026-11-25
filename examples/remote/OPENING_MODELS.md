# Running models remotely with `evedesign.remote`

`evedesign.remote` runs a model somewhere else (another Python environment, a
container, or Modal) and gives you a local stand-in that behaves like the model. You
say **where** it runs (the target) and **which** model it is (the spec), and use it like
any other model.

```python
from evedesign.remote import open_model

with open_model("docker:ghcr.io/org/esm2:2026.11", "esm2") as esm:
    esm.build(system)
    scored = esm.score(instances)
```

The examples below assume:

```python
from evedesign.system import Protein, System

system = System([Protein(rep="MKTAYIAKQRQISFVKSHFSRQ", id="A")])
instances = [system.rep_to_instance(require_reps=True)]
```

## The four ways to open a model

| Call | What you get |
|---|---|
| `open_model(target, model)` | the model created but not built; call `build()` yourself |
| `open_built(target, model, system, data=None)` | the model created and built, with any nested models |
| `reload_model(target, blob)` | a model you saved earlier, already built (short for `open_built(target, {"serialized": blob})`) |
| `session(infra, seed)` | several models that are closed together and can be started and built in parallel |

Each call also takes `seed=` and `infra=`. Use them in a `with` block: when it ends, the
container or Modal sandbox stops and its GPU is freed.

```python
from evedesign.remote import open_built, open_model, reload_model, session

# created remotely, built by you
with open_model("docker:esm2:2026.11", "esm2") as esm:
    esm.build(system)
    blob = esm.fetch_serialized()[-1]          # the built model, as bytes

# brought back later, without building again
with reload_model("docker:esm2:2026.11", blob) as esm:
    scored = esm.score(instances)

# a whole sampler with its scorers, built and run remotely in one image
gibbs = {
    "class": "evedesign.samplers.gibbs:GibbsSampler",
    "args": {"num_sweeps": 10, "seed": 7},
    "models": {"scorers": [
        {"class": "evedesign.models.esm2:ESM2", "args": {"device": "cuda"}},
        {"class": "evedesign.restraints.physicochemical:MolecularWeightRestraint",
         "args": {"lower_bound": 2000, "upper_bound": 2600}},
    ]},
}
with open_built("modal:ghcr.io/org/full@sha256:...", gibbs, system) as sampler:
    designs = sampler.generate(num_designs=8)

# several models, closed together
with session("infra.toml", seed=1234) as s:
    esm = s.open_model("esm2-gpu", "esm2")
    mw = s.open_model("molweight-cpu")
    s.build_all(system)                        # builds both at the same time

# several models, also started at the same time
with session("infra.toml", seed=1234) as s:
    esm, mw = s.open_all([("esm2-gpu", "esm2"), ("molweight-cpu", None)])
    s.build_all(system)
```

If one of the models in `open_all` fails to start, the others are stopped again and the
error says which one failed.

A remote model passes the usual checks, so any local sampler can use it:

```python
from evedesign.model import Scorer
from evedesign.samplers.gibbs import GibbsSampler

assert isinstance(esm, Scorer)
designs = GibbsSampler(scorers=[esm, mw], weights=[1.0, 0.5]).generate(16)
```

## Which model: the spec

The second argument says which model to open. It is one of:

```python
open_model(t)                                        # the model the image holds (its card)
open_model(t, "esm2")                                # a key: the image's card, with its settings
open_model(t, "evedesign.models.esm2:ESM2")          # a class path: the class as is, no card
open_model(t, {"key": "esm2", "args": {"device": "cuda", "batch_size": 64}})
```

A plain string carries no arguments. For arguments, or anything else, use a spec
dictionary:

| Field | Meaning |
|---|---|
| `key` or `class` | which model: a key the image's card knows, or a class path |
| `args` | constructor arguments, passed on exactly as given |
| `models` | nested models, built first and passed to the constructor |
| `build_args` | extra arguments for `build()`, such as `entity_subset` |
| `data` | the `data` for `build()` (`open_built`; for nested models, each its own) |
| `wrap` | class wrappers to apply, such as `evedesign.model:system_subset_model` |
| `serialized`, `rebuild` | a saved model (bytes from `fetch_serialized()`) to load instead of creating one; `rebuild: true` builds it again after loading |

**Which arguments win:** your `args` override the model's defaults. Named by key (or not
named at all), the defaults are the image's model card's `hyper_parameters`; the key must
be the card's `key`, or its `name` if it has no `key`, spelled exactly. Named by class
path, the defaults are the constructor's own, whatever card the image has, so a class
path means the same model everywhere.

```python
open_model(t, {"key": "esm2", "args": {"device": "cpu"}})      # card settings, but device="cpu"
open_model(t, {"args": {"device": "cpu"}})                       # the same: the image's own model
```

A model card is the front matter of `/opt/program/README.md` in the image:

```yaml
---
name: esm2
model_class: "evedesign.models.esm2:ESM2"
hyper_parameters:
  model_dir_path: /opt/program/weights/esm2_t33_650M_UR50D
  device: cuda
---
```

An image has one model card, so a key only names the image's own model. In an image
with several models (a sampler and its scorers, say), name the others by class path.

### Nested models

Models that another model takes as arguments, such as a sampler's scorers, go in
`models`. They are built first, in order, and passed to the constructor under their
names, alongside `args`. Each one is a full spec of its own, and may itself be a saved
model.

```python
open_built("modal:ghcr.io/org/full@sha256:...", {
    "class": "evedesign.samplers.gibbs:GibbsSampler",
    "args": {"num_sweeps": 10, "weights": [1.0, 0.5]},
    "models": {"scorers": [
        {"class": "evedesign.models.esm2:ESM2", "args": {"device": "cuda"}},
        {"serialized": blob},                                  # a scorer saved earlier
    ]},
}, system)
```

`args` is never searched for models: whatever you put there reaches the constructor
unchanged. A name in both `args` and `models` is an error.

### Build arguments

`build_args` are used every time the model is built; arguments you pass to `build()`
yourself win over them.

```python
spec = {"key": "esm2", "wrap": ["evedesign.model:system_subset_model"],
        "build_args": {"entity_subset": [1]}}

with open_model("esm2-gpu", spec, infra="infra.toml") as esm:
    esm.build(system)                            # entity_subset=[1]
    esm.build(system, entity_subset=[0, 1])      # your value wins
```

## Where it runs: the target

The first argument says where the model runs:

- **no colon:** a target from your infra file, such as `"esm2-gpu"`;
- **a colon:** a launcher and an image, exactly as written, such as `"docker:esm2:2026.11"`;
- **a dictionary:** a one-off target.

```python
open_model("esm2-gpu", "esm2", infra="infra.toml")                 # from the infra file
open_model("docker:ghcr.io/org/esm2:2026.11", "esm2")              # any tag or digest
open_model("podman:esm2:2026.11", "esm2")
open_model("apptainer:/srv/sif/esm2.sif", "esm2")
open_model("modal:ghcr.io/org/esm2@sha256:...", "esm2")
open_model({"launcher": "docker", "image": "esm2:2026.11", "gpus": "none"}, "esm2")
```

Everything after the first colon is the image, so tags, digests and registry ports work
as usual. Target names in the infra file can't contain a colon.

### The infra file

The infra file lists your targets by name. It can be TOML, YAML, JSON, or a dictionary
in Python:

```toml
[targets.esm2-gpu]
image = "ghcr.io/org/esm2@sha256:..."

[targets.eve-cpu]
image = "ghcr.io/org/eve@sha256:..."
gpus = "none"
cpus = 4

[targets.esm2-modal]
launcher = "modal"
image = "ghcr.io/org/esm2@sha256:..."
gpu_type = "H100"
registry_secret = "ghcr-pull"

[targets.eve-venv]
launcher = "command"
command = ["/opt/envs/eve/bin/python", "-m", "evedesign.remote.worker"]
```

To pin every image to an exact version, pass a lock file made with
`evedesign-remote lock`. Under a lock, only the named targets it lists can be used;
shortcuts and one-off dictionaries are refused:

```python
from evedesign.remote import Infra

infra = Infra.coerce("infra.toml", lock="infra.lock")
```

### Target settings

| Setting | Default | Meaning |
|---|---|---|
| `launcher` | `"docker"` | `docker`, `podman`, `apptainer`, `modal`, `command`, or a plugin |
| `image` | — | the image (a `.sif` path for apptainer) |
| `command` | — | `command` launcher only: the full command that starts the worker |
| `gpus` | `"auto"` | `"auto"`: one of your GPUs per model that can use one; `"none"`; or a list of devices, such as `[0, 1]` |
| `cpus` | no limit | caps the threads numeric libraries use; also a hard limit for docker/podman and a reservation on Modal |
| `env` | — | environment variables for the worker |
| `mounts` | — | folders to make visible: `["/data/weights:/models:ro"]` (docker, podman, apptainer) |
| `start_timeout` | none | seconds to wait for the worker to start; then it is stopped and you get a `TimeoutError` |
| `call_timeout` | none | seconds one call may take; then the worker is stopped and the model can't be used any more |

Launcher-specific settings:

| Launcher | Settings |
|---|---|
| docker, podman | `shm_size` (default `"2g"`), `network_mode` (e.g. `"none"`) |
| modal | `gpu_type` (e.g. `"H100"`), `timeout` (default 24 hours), `app` (default `"evedesign-remote"`), `registry_secret`, `secrets`, `volumes` |
| command | `network = true` when the command reaches another machine (ssh, srun, kubectl) |

Every launcher starts the worker the same way, as `python -m evedesign.remote.worker`
inside the image (whatever the image's entrypoint is), so an image only needs
`evedesign[remote]` installed and its Python on `PATH`.

With `gpus = "auto"`, each model opened on its own gets one GPU, and a whole tree opened
with `open_built` gets one GPU shared by all its models. To give a tree more, list the
devices (`gpus = [0, 1]`), or on Modal ask for several (`gpu_type = "H100:2"`).

Timeouts are useful wherever starting can hang, such as a job waiting in a Slurm queue:

```toml
[targets.cluster]
launcher = "command"
command = ["srun", "--gres=gpu:1", "apptainer", "exec", "--nv", "/srv/sif/esm2.sif",
           "python", "-m", "evedesign.remote.worker"]
network = true
start_timeout = 1800       # give up if the job hasn't started within 30 minutes
```

## Model files

File paths are read where the model runs, so they must exist there.

- **By default,** paths come from the model card, usually files stored in the image.
- **To use other files,** pass the path in `args`, using the model's own argument name.
- **To make outside files visible,** use `mounts`, or `volumes` on Modal.

```python
infra = {"targets": {
    "esm2": {"image": "ghcr.io/org/esm2@sha256:...", "mounts": ["/data/weights:/models:ro"]},
    "esm2-modal": {"launcher": "modal", "image": "ghcr.io/org/esm2@sha256:...",
                   "volumes": {"/models": "esm2-weights"}},
}}

open_model("esm2", {"key": "esm2", "args": {"model_name": "esm2_t36_3B_UR50D", "model_dir_path": "/models/esm2_3B"}}, infra=infra)
```

Files from outside the image are recorded in the provenance, but unlike the image they
aren't pinned to a version.

## Seeds

- **`seed=`** on a call seeds the general random generators where the model runs (Python's
  `random`, numpy, torch).
- **A session seed** gives every model its own seed, worked out from the session seed and
  the order you open them in. A `seed=` on the call wins.
- **Models with their own seed argument,** such as `GibbsSampler`, take it in `args`. The
  seeds above don't reach them.

```python
with session(infra, seed=1234) as s:
    a = s.open_model("esm2", "esm2")                     # gets its own seed from 1234
    b = s.open_model("esm2", "esm2", seed=7)             # 7
    sampler = s.open_built("esm2-modal", {
        "class": "evedesign.samplers.gibbs:GibbsSampler",
        "args": {"seed": 7},                             # the sampler's own seed
        "models": {"scorers": [{"key": "esm2"}]},
    }, system)
```
