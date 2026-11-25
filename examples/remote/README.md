# Models in isolated environments (`evedesign.remote`)

Run models in their own environments, containers or Modal sandboxes, behind local
proxies that behave like the models. One use is several mutually incompatible scorers
(different torch builds, say) in one multi-objective design run. The proxies work
anywhere a model does, and record exactly what ran.

```
pip install "evedesign[remote]"          # Modal targets: "evedesign[modal]"
```

## Direct use

[OPENING_MODELS.md](OPENING_MODELS.md) covers the ways to open a model, how targets
and models are named, which arguments win, and the defaults, with examples.

```python
from evedesign.remote import session
from evedesign.samplers.gibbs import GibbsSampler

with session("infra.toml", seed=1234) as s:        # or a mapping, or none for shortcuts only
    evm, esm = s.open_all([                          # workers start at the same time
        ("evmutation2", "evmutation2"),
        ("docker:ghcr.io/<org>/esm2:2026.11", {"key": "esm2", "args": {"batch_size": 256}}),
    ])
    s.build_all(system, data=None)                   # build() is forwarded; builds run in parallel
    designs = GibbsSampler(scorers=[evm, esm], weights=[1, .5]).generate(16)
    s.provenance()                                   # what ran, per model
```

A model is named by a string (a key the image's model card knows, or `"module:Class"`),
by a spec dictionary (`key` or `class`, `args`, `models`, `build_args`, `data`, `wrap`,
`serialized`, `rebuild`), or by nothing (the image's own model). `args` go to the
constructor unchanged, over the card's defaults.

`open_built(target, spec, system, data)` returns the model already built. Its nested
models (`models` in the spec) are built first on the far side, so a whole sampling loop
can run in one container or on Modal.

A spec may wrap its class with factories, `factory(cls, **args) -> cls` with the first
entry innermost (`evedesign.remote.wrapping`), and give `build()` extra keywords:

```python
{"class": "evedesign.models.esm2:ESM2", "args": {"device": "cuda"},
 "wrap": ["evedesign.model:system_subset_model"],    # or {"factory": ..., "args": {...}}
 "build_args": {"entity_subset": [1]}}              # system-dependent options go here
```

A wrapped class points to the class it wraps (`__wrapped__`), so a wrapped CPU-only
model gets no GPU.

The `seed` of `open_model`/`open_built` seeds the worker's global generators (`random`,
numpy's global state, torch). `GibbsSampler` keeps its own generator, so give it a seed
as a constructor argument, `"args": {"seed": 1234}`; without one, Gibbs runs are not
reproducible.

`GibbsSampler` calls its scorers concurrently (`evedesign.utils.score_all`), so
isolated scorers cost the slowest per step, not the sum. Other generators can use the
same helper.

## Infrastructure is plain data

An infra config is a mapping of plain values, so it can live in Python, YAML, TOML
or JSON (see `infra.example.toml`). Anywhere one is accepted, so is a file path, and
`Infra.to_dict()` round-trips. Each target lists its own settings; there are no
defaults to inherit.

```python
infra = {"targets": {
    "esm2": {"image": "ghcr.io/proteingym/esm2:2026.11"},
    "eve":  {"image": "eve:latest", "gpus": [1]},
}}
with session(infra) as s:
    a = s.open_model("esm2", "esm2")                                  # named (no colon)
    b = s.open_model("docker:esm2:latest", "esm2")                    # <launcher>:<image>
    c = s.open_model({"image": "esm2:latest", "gpus": "none"}, "esm2")  # one-off
```

`Infra.coerce(x, lock=...)` normalises. `lock` takes a lock or provenance file, or a
`{target: pinned_image}` mapping; under a lock only named targets it lists can be used.
Unknown keys and target fields raise immediately.

Which model of a pipeline runs where (placement) is part of evedesign-server
(`evedesign_server.placement`).

## GPUs and CPUs on one host

With `gpus = "auto"` (the default), every model that can use a GPU gets one of this
process's GPUs, round-robin, and CPU-only models get none. A tree opened with
`open_built` counts as one model: all its models share one GPU. Whether a model can use a
GPU comes from its own `supports_gpu`/`requires_gpu`. This process's GPUs are those in
`CUDA_VISIBLE_DEVICES` if a scheduler set it (Snakemake, Slurm), and otherwise every
GPU on the host.

| `gpus` | Effect |
|---|---|
| `"auto"` | one device per GPU model (or tree), round-robin (default) |
| `"none"` | no GPU |
| `[1]`, `[0, 1]` | exactly these devices (host numbering), outside the pool |

`cpus = 2` on a target sets the thread counts of OpenMP, MKL, OpenBLAS and numexpr
(torch follows `OMP_NUM_THREADS`) with every launcher. It is also a hard limit for
docker/podman (`--cpus`) and a CPU reservation on Modal. For apptainer and `command`
(venvs), the thread counts are the only limit. There is no VRAM accounting. The
assigned device is recorded in provenance.

## Model parameters outside the image

Pass the path as the worker sees it, in the spec's `args` or in the card's
`hyper_parameters`, and make it visible:

```toml
[targets.esm2]                     # docker / podman: bind mounts; apptainer: --bind
image = "ghcr.io/<org>/esm2@sha256:..."
mounts = ["/data/weights:/models:ro"]

[targets.esm2-modal]               # Modal: volumes by mount path
launcher = "modal"
image = "ghcr.io/<org>/esm2@sha256:..."
volumes = { "/models" = "esm2-weights" }   # modal volume create esm2-weights; modal volume put ...
```

```python
esm = s.open_model("esm2-modal", {"key": "esm2", "args": {"model_dir_path": "/models/esm2_650M"}})
```

With `command` targets, host paths work directly. Files from outside the image are
not pinned by the image digest: provenance records the mounts and volumes and flags
`outside_image`, but does not hash their contents.

## Images

Use one model per image, built from the proteingym-models template (`model.Dockerfile`,
`entrypoint.sh`):
- a pinned environment with `evedesign[remote]` and the model's dependencies;
- the model card at `/opt/program/README.md` (`model_class`, `hyper_parameters`, optional `key`);
- the weights baked in;
- the environment's python first on `PATH`: every launcher starts the worker as
  `python -m evedesign.remote.worker`, bypassing the image's entrypoint (docker/podman
  `--entrypoint python`, `apptainer exec`, Modal by exec). The template's `serve`
  entrypoint is only for running the image by hand.

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
gpu_type = "L40S"               # or "H100:2" for two
registry_secret = "ghcr-pull"   # private images
secrets = ["huggingface"]       # named Modal secrets, e.g. HF_TOKEN
volumes = { "/models" = "esm2-weights" }   # optional: weights outside the image
cpus = 4                        # optional: CPU reservation and thread counts
timeout = 14400                 # upper bound on a session's lifetime (s)
```

The worker's logs stream to your stderr. A Sandbox holds its GPU while the session is
open, so close sessions when done (`with session(...)`).

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
digest. A queued `srun` waits as long as the queue does unless the target sets
`start_timeout`; `call_timeout` bounds each call. On either timeout the worker is
stopped and `TimeoutError` raised.

Another provider is a launcher: a class registered under its name in the entry point
group `evedesign.remote.launchers`, so it can live in its own package. It declares the
target options it accepts, which are written flat next to the common fields; Modal is
built the same way (`evedesign/remote/modal_launcher.py`).

```python
from evedesign.remote.launchers import Launcher

class MyCloud(Launcher):
    name = "mycloud"
    options = {"machine": None, "zone": "eu-west"}      # target options and their defaults

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
  defaults), or else uses the image's model card for a key it has (or no name). Nested models come only from
  the spec's `models`. It exits on stdin EOF, and needs nothing but evedesign.
- **Wire:** length-prefixed msgpack. System, SystemInstance and LabeledInstanceDataset travel via `serialize()`,
  DataFrames column by column (numeric as arrays, strings as lists; other dtypes
  are refused), and numpy arrays as raw bytes. There is no pickle, so
  environments can differ (tested: numpy 2.2 / pandas 2.2 ↔ numpy 2.3 / pandas 3.0).
  Session open checks the protocol version and a hash of the interface surface.
- **Launchers:** `docker` / `podman` (pinned by digest), `apptainer`, `modal` (a
  Sandbox), `command` (any argv: venvs, ssh, srun, kubectl; not hermetic), and plugins
  (entry point group `evedesign.remote.launchers`).
- **Provenance:** before launch, the image digest or `.sif` sha256 is pinned. The worker
  reports evedesign, python and package hashes, the model class, resolved arguments,
  the seed, and GPUs. A run's provenance file can be used as a lock.

## Tests

```
pytest tests/remote                                          # in-process tests, a few seconds
tests/remote/docker/build_test_images.sh <envA> <envB>       # two images, different numpy/pandas (Linux)
EVEDESIGN_TEST_ENVS=<envA>,<envB> pytest tests/remote        # plus real environments and containers
```
