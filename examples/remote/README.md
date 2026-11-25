# Running models remotely with `evedesign.remote`

`evedesign.remote` runs a model somewhere else (another Python environment, a
container, an HPC job or Modal) and gives you a local stand-in that behaves like the
model. You say **where** it runs (the target) and **which** model it is (the spec), and
use it like any other model. One use is several mutually incompatible scorers
(different torch builds, say) in one design run. Every model records exactly what ran.

```
pip install "evedesign[remote]"          # Modal targets: "evedesign[modal]"
```

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

# several models, started at the same time, built in parallel, closed together
with session("infra.toml", seed=1234) as s:
    esm, mw = s.open_all([("esm2-gpu", "esm2"), ("molweight-cpu", None)])
    s.build_all(system)
    s.provenance()                             # what ran, per model
```

`s.open_model` and `s.open_built` open one model at a time within a session. If one
of the models in `open_all` fails to start, the others are stopped again and the error
says which one failed.

A remote model passes the usual checks, so any local sampler can use it:

```python
from evedesign.model import Scorer
from evedesign.samplers.gibbs import GibbsSampler

assert isinstance(esm, Scorer)
designs = GibbsSampler(scorers=[esm, mw], weights=[1.0, 0.5]).generate(16)
```

`GibbsSampler` calls its scorers concurrently (`evedesign.utils.score_all`), so remote
scorers cost the slowest per step, not the sum. Other generators can use the same helper.

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
| `data` | the `data` for `build()` (for nested models, each its own) |
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

A model card is the front matter of `/opt/program/README.md` in the image. Its
`model_class` may be written `package.module:Class` or `package.module.Class`:

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
unchanged. A name in both `args` and `models` is an error. A remote model cannot be
passed into another one's `args`; nest its spec in `models` instead, so both run in
the same place.

### Build arguments, data and wrappers

The spec's `build_args` and `data` are used every time the model is built; what you pass
to `build()` yourself wins over them.

A spec may wrap its class with factories, `factory(cls, **args) -> cls`, the first entry
innermost (`evedesign.remote.wrapping`). An entry is `"module:factory"` or
`{"factory": ..., "args": {...}}`. Wrapper arguments configure the wrapper; anything that
depends on the system goes in `build_args`. A wrapped CPU-only model still gets no GPU.

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

The infra file lists your targets by name. It is plain data, so it can be TOML, YAML,
JSON, or a dictionary in Python; anywhere one is accepted, so is a file path. Its only
top-level keys are `targets` and an optional `lock`. Each target lists its own settings;
there are no defaults to inherit. See `infra.example.toml`.

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

Unknown keys and settings raise as soon as the file is read. `Infra.coerce(x)` turns a
mapping or file into an `Infra`, and `Infra.to_dict()` gives the same data back.

To pin every image to an exact version, pass a lock file made with
`evedesign-remote lock` (a run's provenance file works too). Under a lock, only the named
targets it lists can be used; shortcuts and one-off dictionaries are refused:

```python
from evedesign.remote import Infra

infra = Infra.coerce("infra.toml", lock="infra.lock")
```

Which model of a pipeline runs where (placement) is part of evedesign-server
(`evedesign_server.placement`).

### Target settings

| Setting | Default | Meaning |
|---|---|---|
| `launcher` | `"docker"` | `docker`, `podman`, `apptainer`, `modal`, `command`, or a plugin |
| `image` | — | the image (a `.sif` path for apptainer) |
| `command` | — | `command` launcher only: the full command that starts the worker |
| `gpus` | `"auto"` | `"auto"`, `"none"`, or a list of devices such as `[0, 1]` (see below) |
| `cpus` | no limit | caps the threads numeric libraries use; also a hard limit for docker/podman and a reservation on Modal |
| `env` | — | environment variables for the worker |
| `mounts` | — | folders to make visible: `["/data/weights:/models:ro"]` (docker, podman, apptainer) |
| `start_timeout` | none | seconds to wait for the worker to start; then it is stopped and you get a `TimeoutError` |
| `call_timeout` | none | seconds one call may take; then the worker is stopped and the model can't be used any more |

Launcher-specific settings:

| Launcher | Settings |
|---|---|
| docker, podman | `shm_size` (default `"2g"`), `network_mode` (e.g. `"none"`) |
| modal | `gpu_type` (e.g. `"H100"`, `"H100:2"`), `timeout` (default 24 hours), `app` (default `"evedesign-remote"`), `registry_secret`, `secrets`, `volumes` |
| command | `network = true` when the command reaches another machine (ssh, srun, kubectl) |

Every launcher starts the worker the same way, as `python -m evedesign.remote.worker`
inside the image (docker/podman `--entrypoint python`, `apptainer exec`, Modal exec),
whatever the image's entrypoint is.

Timeouts are useful wherever starting can hang, such as a job waiting in a Slurm queue
(see [Venvs, other machines and other providers](#venvs-other-machines-and-other-providers)).

## GPUs and CPUs

With `gpus = "auto"` (the default), every model that can use a GPU gets one of this
process's GPUs, round-robin, and CPU-only models get none. A whole tree opened with
`open_built` counts as one model: all its models share one GPU. Whether a model can use a
GPU comes from its own `supports_gpu`/`requires_gpu`. This process's GPUs are those in
`CUDA_VISIBLE_DEVICES` if a scheduler set it (Snakemake, Slurm), and otherwise every GPU
on the host.

| `gpus` | Effect |
|---|---|
| `"auto"` | one device per GPU model (or tree), round-robin (default) |
| `"none"` | no GPU |
| `[1]`, `[0, 1]` | exactly these devices (host numbering), outside the pool |

To give a tree more than one GPU, list the devices, or on Modal ask for several
(`gpu_type = "H100:2"`). Targets on other machines (Modal, `network = true`) use their own
GPUs, not this host's.

`cpus = 2` sets the thread counts of OpenMP, MKL, OpenBLAS and numexpr (torch follows
`OMP_NUM_THREADS`) with every launcher. It is also a hard limit for docker/podman
(`--cpus`) and a CPU reservation on Modal. For apptainer and `command`, the thread counts
are the only limit. There is no VRAM accounting. The assigned device is recorded in
provenance.

## Model files

File paths are read where the model runs, so they must exist there.

- **By default,** paths come from the model card, usually files stored in the image.
- **To use other files,** pass the path in `args`, using the model's own argument name.
- **To make outside files visible,** use `mounts` (docker, podman, apptainer), or
  `volumes` on Modal. With `command` targets, host paths work directly.

```python
infra = {"targets": {
    "esm2": {"image": "ghcr.io/org/esm2@sha256:...", "mounts": ["/data/weights:/models:ro"]},
    "esm2-modal": {"launcher": "modal", "image": "ghcr.io/org/esm2@sha256:...",
                   "volumes": {"/models": "esm2-weights"}},  # modal volume create / put
}}

open_model("esm2", {"key": "esm2", "args": {"model_name": "esm2_t36_3B_UR50D",
                                            "model_dir_path": "/models/esm2_3B"}}, infra=infra)
```

A mount's source is made absolute and must exist; `ro` makes it read-only. Files from
outside the image are recorded in the provenance (flagged `outside_image`), but unlike
the image they aren't pinned to a version or hashed.

## Seeds

- **`seed=`** on a call seeds the general random generators where the model runs (Python's
  `random`, numpy, torch).
- **A session seed** gives every model its own seed, worked out from the session seed and
  the order you open them in. A `seed=` on the call wins.
- **Models with their own seed argument,** such as `GibbsSampler`, take it in `args`. The
  seeds above don't reach them; without one, Gibbs runs are not reproducible.

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

## Images

An image needs Python 3.12 or newer with `evedesign[remote]` and the model's
dependencies, and that environment's `python` first on `PATH`. No entrypoint, model card
or evedesign-server is required.

- **Minimal:** this repository's `Dockerfile` installs the checkout with the `remote`
  extra, plus model extras (`--build-arg EXTRAS="remote esm2"`), pinned by `uv.lock`.
  Models are then named by class path, and weights mounted or baked in.
- **Model images to share:** one model per image, from the proteingym-models template
  (`model.Dockerfile`): a pinned environment, the model card at `/opt/program/README.md`,
  and the weights baked in. Its `serve` entrypoint is only for running the image by hand.

Worth doing in any image:

- precompile bytecode (`UV_COMPILE_BYTECODE=1` with uv; pip does it by default). Workers
  run as your user and cannot write `.pyc` files, so otherwise every start recompiles
  numpy, torch and so on;
- for Modal, install pip into the environment (`uv pip install pip`): Modal installs its
  client with the first `python` on `PATH`, and uv environments have none;
- bake in or mount the weights, so they are not downloaded at every start.

```
evedesign-remote check docker:ghcr.io/<org>/esm2:2026.11    # starts it, reports what runs
evedesign-remote lock esm2 --infra infra.toml -o infra.lock  # pin targets to content hashes
```

For apptainer: `apptainer build esm2.sif docker://ghcr.io/<org>/esm2:2026.11`.

## Modal

Each session is a Modal Sandbox from the target's registry image, kept for the whole
session. There is nothing to deploy. The image's ENTRYPOINT is reset and the Sandbox's
main process is `sleep infinity`, so any image works whatever its entrypoint; the worker
runs next to it, and the Sandbox ends when the session closes or `timeout` is reached.
The reset is an image layer Modal builds once per image.

1. `pip install "evedesign[modal]"`, then run `modal token new` (or set `MODAL_TOKEN_ID` /
   `MODAL_TOKEN_SECRET`) where the orchestrator runs.
2. Push the image to a registry Modal can pull from, and use its digest.
3. For a private registry: `modal secret create ghcr-pull REGISTRY_USERNAME=<user> REGISTRY_PASSWORD=<token>`.
   For AWS ECR (`<account>.dkr.ecr.<region>.amazonaws.com/...`), use the keys of an IAM user
   with read access instead: `modal secret create aws-ecr AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_REGION=...`.
   ECR images are pulled with Modal's ECR support, which makes its own short-lived tokens.
4. Add a target:

```toml
[targets.esm-modal]
launcher = "modal"
image = "ghcr.io/<org>/esm2@sha256:..."
gpu_type = "L40S"               # or "H100:2" for two; gpus = "none" for no GPU
registry_secret = "ghcr-pull"   # private images
secrets = ["huggingface"]       # named Modal secrets, e.g. HF_TOKEN
volumes = { "/models" = "esm2-weights" }   # optional: weights outside the image
cpus = 4                        # optional: CPU reservation and thread counts
timeout = 14400                 # upper bound on a session's lifetime (s)
start_timeout = 900             # optional: give up if the worker has not started
```

The worker's logs stream to your stderr. A Sandbox holds its GPU while the session is
open, so close sessions when done (`with ...`). `start_timeout` covers the worker
starting, not the creation of the Sandbox.

## Venvs, other machines and other providers

The `command` launcher starts any command whose stdin and stdout reach a worker:

```toml
[targets.esm-venv]                 # a venv (or: uv run --project ... --frozen --no-sync python -m ...)
launcher = "command"
command = ["/opt/envs/esm/bin/python", "-m", "evedesign.remote.worker"]

[targets.gpu-box]                  # another machine
launcher = "command"
command = ["ssh", "-T", "-o", "BatchMode=yes", "gpu-box", "docker", "run", "-i", "--rm", "--entrypoint", "python",
           "ghcr.io/<org>/esm2@sha256:...", "-m", "evedesign.remote.worker"]
network = true                     # this host's GPUs mean nothing there

[targets.cluster]                  # a Slurm job step
launcher = "command"
command = ["srun", "--gres=gpu:1", "--time=02:00:00", "apptainer", "exec", "--nv", "/srv/sif/esm2.sif",
           "python", "-m", "evedesign.remote.worker"]
network = true
start_timeout = 1800               # give up if the job has not started within 30 minutes

[targets.k8s]                      # a Kubernetes pod
launcher = "command"
command = ["kubectl", "run", "-i", "--rm", "--restart=Never", "--quiet", "--image=ghcr.io/<org>/esm2@sha256:...",
           "--command", "esm2-worker", "--", "python", "-m", "evedesign.remote.worker"]
network = true
```

The worker writes a ready marker before its first frame, so output that arrives earlier on
the same stream (a login banner, scheduler messages) is skipped and logged, not taken for
protocol. Nothing may print to stdout after that. These targets are not hermetic:
provenance records the command and the environment the worker reports, not an image
digest.

Another provider is a launcher: a class registered under its name in the entry point
group `evedesign.remote.launchers`, so it can live in its own package. It declares the
target settings it accepts, which are written next to the common ones; Modal is built the
same way (`evedesign/remote/modal_launcher.py`).

```python
from evedesign.remote.launchers import Launcher

class MyCloud(Launcher):
    name = "mycloud"
    options = {"machine": None, "zone": "eu-west"}      # target settings and their defaults

    def is_network(self, target):
        return True

    def pin(self, target):                              # provenance, resolved before start
        return {"pinned_image": target.image, "pinned_by_digest": "@sha256:" in target.image}

    def start(self, target, image):                     # a Transport to a worker there
        return MyCloudTransport(target, image)          # _send(bytes), _receive() -> frame, _terminate(kill)
```

```toml
# the provider's pyproject.toml
[project.entry-points."evedesign.remote.launchers"]
mycloud = "evedesign_mycloud:MyCloud"
```

A provider has to keep one container for the whole session, because the built model
lives in the worker, and has to provide a byte stream to the worker in it.
Request/response-only services (functions without a session) do not fit. If the stream
delivers arbitrary chunks, `protocol.FrameBuffer` reassembles frames, as Modal's
transport does.

## Security notes

- **Infra is code.** A target can run any command (`command`), mount host folders, and
  name images and secrets. Only operators should write infra or placement; never build
  them from user input.
- **Wrap factories are imports and calls.** Pass `allowed=` to
  `wrapping.apply_wraps` wherever specs come from users (evedesign-server checks its
  `wrap_factories` setting for the whole spec tree before anything is built or placed).
- **Containers** run as your uid, with no capabilities (`--cap-drop ALL`,
  `no-new-privileges`). Set `network_mode = "none"` on a docker/podman target for models
  that need no network. Environment variables are passed by name, so values such as
  tokens stay out of command lines.
- **Frames** over `EVEDESIGN_REMOTE_MAX_FRAME_MB` (default 2048) are refused; raise it for
  very large serialized models.
- **Errors** carry the remote type and message; the remote traceback is in
  `RemoteError.remote_traceback`.

## How it works

- **Proxy:** a local stand-in registered with evedesign's ABCs, so `isinstance(x, Scorer)`
  holds without importing the model. The surface comes from the ABCs, and calls bind
  against their signatures. Undeclared attributes raise.
- **Worker:** `python -m evedesign.remote.worker`. It imports `class` (constructor
  defaults), or else uses the image's model card for its key (or no name). Nested models
  come only from the spec's `models`. It exits on stdin EOF, and needs nothing but
  evedesign.
- **Wire:** length-prefixed msgpack. System, SystemInstance and LabeledInstanceDataset
  travel via `serialize()`, DataFrames column by column (numeric as arrays, strings as
  lists; other dtypes are refused), and numpy arrays as raw bytes. There is no pickle, so
  environments can differ (tested: numpy 2.2 / pandas 2.2 ↔ numpy 2.3 / pandas 3.0).
  Session open checks the protocol version and a hash of the interface surface.
- **Launchers:** `docker` / `podman` (pinned by digest), `apptainer`, `modal` (a
  Sandbox), `command` (any argv: venvs, ssh, srun, kubectl; not hermetic), and plugins
  (entry point group `evedesign.remote.launchers`).
- **Timeouts:** on `start_timeout` or `call_timeout` the worker is stopped (docker
  containers by name) and `TimeoutError` raised; the session cannot continue.
- **Provenance:** before launch, the image digest or `.sif` sha256 is pinned. The worker
  reports evedesign, python and package hashes, the model class, resolved arguments,
  the seed, and GPUs. A run's provenance file can be used as a lock.

## Tests

```
pytest tests/remote                                          # in-process tests, a few seconds
tests/remote/docker/build_test_images.sh <envA> <envB>       # two images, different numpy/pandas (Linux)
EVEDESIGN_TEST_ENVS=<envA>,<envB> pytest tests/remote        # plus real environments and containers
```
