"""
evedesign.remote without worker processes: codec, surface, config, GPU
accounting, and the proxy against an in-process worker (same protocol and
codec, no process start), so these run in well under a few seconds.
"""

import functools
import pickle

import numpy as np
import pandas as pd
import pytest

from evedesign.model import BaseModel, ConditionalMutationScorer, Generator, MutationScorer, Scorer
from evedesign.remote import Infra, Target, codec, open_model, open_built, session
from evedesign.remote.config import derive_seed
from evedesign.remote.protocol import RemoteError
from evedesign.remote.surface import declared_surface, surface_hash
from evedesign.remote.launchers import Launcher, container_argv, gpu_flags
from evedesign.remote.transport import Transport

from .conftest import MW, P, PI

LOCAL = Target(name="in-process", launcher="command", command=("in-process",))


class InProcessTransport(Transport):
    """A worker Session in this process, behind the real framing and codec."""

    def __init__(self) -> None:
        from evedesign.remote import worker

        super().__init__("in-process")
        self._worker, self._session, self._reply = worker, worker.Session(), None

    def _send(self, data: bytes) -> None:
        message = codec.loads(data[4:])  # one frame: 4-byte length, then the body
        self._reply = codec.loads(codec.dumps(self._worker.handle(self._session, message)))

    def _receive(self):
        return self._reply

    def _terminate(self) -> None:
        pass


@pytest.fixture
def local(monkeypatch):
    """open_model / open_built on LOCAL run their worker in-process."""
    import evedesign.remote.api as api

    monkeypatch.setattr(api, "launch", lambda target, image: InProcessTransport())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")  # the worker sets it; keep it to this test
    return LOCAL


# --- codec and surface --------------------------------------------------------------

def test_codec_roundtrips_library_objects(system):
    inst = system.rep_to_instance(require_reps=True)
    inst.score, inst.metadata = 1.5, {"k": [1, 2]}
    index = pd.MultiIndex.from_tuples([(0, 0, 3), (1, 0, 4)], names=["instance", "entity", "pos"])
    frame = pd.DataFrame({"A": [0.1, 0.2], "C": [np.nan, 1.0]}, index=index)
    value = {"system": system, "inst": inst, "frame": frame, "arr": np.arange(6.0).reshape(2, 3),
             "tuple": (0, 3, "K", "W"), "scalar": np.float32(2.5), 3: "int key"}

    out = codec.loads(codec.dumps(value))

    assert out["system"] == system
    assert str(out["inst"][0].rep) == str(inst[0].rep) and out["inst"].score == 1.5
    pd.testing.assert_frame_equal(out["frame"], frame)
    np.testing.assert_array_equal(out["arr"], value["arr"])
    assert out["tuple"] == (0, 3, "K", "W") and out["scalar"] == 2.5 and out[3] == "int key"
    with pytest.raises(codec.CodecError, match="cannot send"):
        codec.dumps({"x": object()})


def test_codec_frames_keep_index_columns_and_dtypes():
    frames = [
        pd.DataFrame({"wt": ["K", "W"], "ok": [True, False], "n": np.array([1, 2], dtype=np.int32)},
                     index=pd.Index(["a", "b"], name="id")),
        pd.DataFrame(np.zeros((0, 2)), columns=["x", "y"]),
        pd.DataFrame([[1, 2]], columns=pd.MultiIndex.from_tuples([("a", 1), ("b", 2)])),
    ]
    for frame in frames:
        pd.testing.assert_frame_equal(codec.loads(codec.dumps(frame)), frame, check_index_type="equiv")
    for series in (pd.Series([1.0, 2.0], name="s"), pd.Series([3, 4])):
        pd.testing.assert_series_equal(codec.loads(codec.dumps(series)), series, check_index_type="equiv")
    with pytest.raises(codec.CodecError, match="datetime64"):
        codec.dumps(pd.DataFrame({"t": pd.to_datetime(["2020-01-01"])}))


def test_surface_is_derived_from_the_abcs():
    attrs, methods = declared_surface(("ConditionalMutationScorer", "BaseModel"))
    assert {"name", "handles_deletions", "requires_gpu"} <= attrs
    assert {"score_conditional", "positions", "valid_positions", "stats", "can_model"} <= set(methods)
    assert not {"system", "ready", "build"} & (attrs | set(methods))
    assert len(surface_hash()) == 16


def test_score_all_preserves_order_and_overlaps_calls():
    import time

    from evedesign.utils import score_all

    class Slow:
        def __init__(self, v): self.v = v
        def f(self, x): time.sleep(0.1); return self.v * x

    started = time.perf_counter()
    assert score_all([Slow(1), Slow(2), Slow(3)], "f", 10) == [10, 20, 30]
    assert time.perf_counter() - started < 0.25


# --- proxies ---------------------------------------------------------------------

def test_proxy_behaves_like_the_model(local, system):
    from evedesign.restraints.physicochemical import MolecularWeightRestraint

    with open_model(local, MolecularWeightRestraint, seed=3, lower_bound=2000, upper_bound=2600) as m:
        # conformance by registration; named after the remote class
        assert isinstance(m, Scorer) and isinstance(m, ConditionalMutationScorer)
        assert isinstance(m, MutationScorer) and isinstance(m, BaseModel) and not isinstance(m, Generator)
        assert type(m).__name__ == "MolecularWeightRestraint" and m.name == "MolecularWeightRestraint"
        # constructed only: build() is the caller's, as locally
        assert not m.ready and m.system is None
        with pytest.raises(RemoteError) as err:
            m.score([system.rep_to_instance(True)])
        assert "Traceback" in err.value.remote_traceback
        # only the declared surface
        with pytest.raises(AttributeError, match="does not expose"):
            m.lower_bound
        with pytest.raises(TypeError, match="cannot be pickled"):
            pickle.dumps(m)

        m.build(system, None)

        assert m.ready and m.system is system
        assert m.positions(None)[:2] == [(0, 1), (0, 2)]                     # positional call
        assert m.valid_positions([1, 2], system.rep_to_instance(True), 0) == [(0, 1), (0, 2)]
        frame = m.score_conditional([system.rep_to_instance(True)], [0], [3])
        assert isinstance(frame, pd.DataFrame) and frame.index.names == ["instance", "entity", "pos"]
        assert m.provenance["models"][0]["args"] == {"lower_bound": 2000, "upper_bound": 2600}
        assert m.provenance["environment"]["seed"] == 3
        assert m.provenance["launcher"]["cuda_visible_devices"] == ""          # CPU-only: no GPU


def test_open_built_builds_nested_models_and_returns_ready(local, system):
    with pytest.raises(RemoteError, match="use open_built"):
        open_model(local, {"class": "evedesign.samplers.gibbs:GibbsSampler", "args": {"scorers": [MW]}})

    with open_built(local, "evedesign.samplers.gibbs:GibbsSampler", system,
                   scorers=[PI, MW], weights=[1.0, 0.5], num_sweeps=1) as sampler:
        assert isinstance(sampler, Generator) and sampler.ready and sampler.system is system
        designs = sampler.generate(num_designs=2)
        assert [c["name"] for c in designs[0].metadata["scores"]] == [
            "IsoelectricPointRestraint", "MolecularWeightRestraint"]
        assert [r["class"].rpartition(":")[2] for r in sampler.provenance["models"]] == [
            "IsoelectricPointRestraint", "MolecularWeightRestraint", "GibbsSampler"]
        assert len(sampler.fetch_serialized()) == 3


def test_sessions_seed_models_and_refuse_foreign_proxies(local):
    with session(seed=42) as s:
        a = s.open_model(local, P + "MolecularWeightRestraint", args={"lower_bound": 1, "upper_bound": 2})
        b = s.open_model(local, MW)
        seeds = [x.provenance["environment"]["seed"] for x in (a, b)]
        assert None not in seeds and seeds[0] != seeds[1]                    # derived per model
        assert a.provenance["models"][0]["args"]["lower_bound"] == 1
        with pytest.raises(codec.CodecError, match="open_built"):
            s.open_model(local, {"class": "evedesign.samplers.gibbs:GibbsSampler", "args": {"scorers": [a]}})


def test_incompatible_worker_is_refused(local, monkeypatch):
    import evedesign.remote.api as api

    monkeypatch.setattr(api, "surface_hash", lambda: "0" * 16)
    with pytest.raises(RemoteError, match="incompatible worker"):
        open_model(local, MW)


def test_models_are_named_like_locally():
    from evedesign.restraints.physicochemical import MolecularWeightRestraint

    from evedesign.remote.api import as_spec

    assert as_spec("esm2", {"device": "cuda"}) == {"key": "esm2", "args": {"device": "cuda"}}
    assert as_spec("evedesign.models.esm2:ESM2") == {"class": "evedesign.models.esm2:ESM2"}
    assert as_spec(MolecularWeightRestraint)["class"].endswith(":MolecularWeightRestraint")
    assert as_spec({"key": "x", "args": {"a": 1}}, {"b": 2})["args"] == {"a": 1, "b": 2}
    assert as_spec(None) == {}


# --- config ----------------------------------------------------------------------

INFRA = {
    "defaults": {"launcher": "docker", "gpus": "inherit"},
    "targets": {
        "esm2": {"image": "ghcr.io/proteingym/esm2:2026.11"},
        "eve": {"image": "eve:latest", "gpus": "device=1", "env": {"A": "1"}, "mounts": ["/w:/models:ro"]},
        "esmc": {"launcher": "apptainer", "image": "/srv/sif/esmc.sif"},
        "esm2-modal": {"launcher": "modal", "image": "ghcr.io/x/esm2@sha256:ab", "volumes": {"/models": "w"}},
    },
}


def test_convention_targets_need_no_file(tmp_path):
    infra = Infra(defaults={"gpus": "none"}, sif_dir=tmp_path)
    assert infra.resolve("docker:esm2").image == "esm2:latest"
    assert infra.resolve("docker:ghcr.io/x/esm2@sha256:ab").image == "ghcr.io/x/esm2@sha256:ab"
    assert infra.resolve("apptainer:esm2").image == str(tmp_path / "esm2.sif")
    assert infra.resolve("modal:ghcr.io/x/esm2:1").is_network
    with pytest.raises(KeyError, match="unknown target"):
        infra.resolve("esm2")


@pytest.mark.parametrize("suffix", [".toml", ".yaml", ".json"])
def test_infra_is_plain_data_and_round_trips(tmp_path, suffix):
    import json

    import tomli_w
    import yaml

    assert Infra.from_dict(INFRA).to_dict() == INFRA
    write = {".toml": tomli_w.dumps, ".yaml": yaml.safe_dump, ".json": json.dumps}[suffix]
    (tmp_path / f"infra{suffix}").write_text(write(INFRA))
    assert Infra.coerce(tmp_path / f"infra{suffix}").to_dict() == INFRA


def test_inline_targets_locks_and_typos():
    infra = Infra.coerce(INFRA, lock={"esm2": "ghcr.io/proteingym/esm2@sha256:ab"})
    assert infra.resolve("esm2").image.endswith("@sha256:ab") and infra.resolve("esm2").pinned
    with pytest.raises(LookupError, match="not in the lock"):
        infra.resolve("eve")
    inline = Infra.coerce(None).resolve({"launcher": "docker", "image": "eve:big", "gpus": "none"})
    assert inline.image == "eve:big" and inline.gpus == "none"
    with pytest.raises(ValueError, match="unknown fields"):
        Infra.from_dict({"targets": {"x": {"imgae": "typo"}}})
    with pytest.raises(ValueError, match="unknown keys"):
        Infra.from_dict({"target": {}})
    with pytest.raises(ValueError, match=r"unknown fields \['volumes'\] for launcher 'docker'"):
        Infra.from_dict({"targets": {"x": {"image": "i", "volumes": {"/m": "v"}}}})
    # a default only reaches targets whose launcher has the field
    infra = Infra.from_dict({"defaults": {"gpu": "H100"}, "targets": {"d": {"image": "i"},
                                                                     "m": {"launcher": "modal", "image": "i"}}})
    assert infra.resolve("m").options["gpu"] == "H100" and "gpu" not in infra.resolve("d").options
    with pytest.raises(ValueError, match="infra defaults: unknown fields"):
        Infra.from_dict({"defaults": {"gpuu": "H100"}})
    with pytest.raises(ValueError, match="use volumes"):
        Infra.from_dict({"targets": {"x": {"launcher": "modal", "image": "i", "mounts": ["/a:/b"]}}})


def test_seeds_are_derived_deterministically():
    assert derive_seed(None, "x") is None
    assert derive_seed(1, "esm2:default#0") == derive_seed(1, "esm2:default#0") != derive_seed(1, "esm2:default#1")


def test_container_command_line(monkeypatch):
    from evedesign.remote import launchers

    monkeypatch.setattr("evedesign.remote.transport._has_nvidia", lambda: False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    target = Target(name="t", image="img", mounts=("/w:/models:ro",), cpus=2.5, env=(("HF_TOKEN", "secret"),),
                    options={"network_mode": "none"})
    argv = container_argv(target, "img@sha256:1")
    assert argv[:6] == ["docker", "run", "-i", "--rm", "--init", "--label"]
    assert "type=bind,source=/w,target=/models,readonly" in argv and argv[-2:] == ["img@sha256:1", "serve"]
    assert argv[argv.index("--cpus") + 1] == "2.5" and argv[argv.index("--network") + 1] == "none"
    assert {"--cap-drop", "ALL", "no-new-privileges"} <= set(argv) and "--gpus" not in argv
    assert {"HOME=/tmp", "OMP_NUM_THREADS=2"} <= set(argv)
    # target.env by name only: values (tokens) stay out of argv, in the docker CLI's environment
    assert "HF_TOKEN" in argv and not any("secret" in a for a in argv)
    seen = {}
    monkeypatch.setattr(launchers, "StdioTransport", lambda argv, description, env=None: seen.update(env))
    monkeypatch.setenv("HOME", "/home/me")
    launchers.get("docker").start(target, "img")
    assert seen["HF_TOKEN"] == "secret" and seen["HOME"] == "/home/me"  # the CLI keeps its own config


def test_frame_limit_and_short_remote_errors(monkeypatch):
    import struct

    from evedesign.remote import protocol

    with pytest.raises(protocol.FrameTooLarge, match="EVEDESIGN_REMOTE_MAX_FRAME_MB"):
        protocol.FrameBuffer("t").feed(protocol.READY + struct.pack(">I", protocol.MAX_FRAME + 1))
    monkeypatch.setattr(protocol, "MAX_FRAME", 100)
    with pytest.raises(protocol.FrameTooLarge):
        protocol.encode({"blob": b"x" * 200})
    err = protocol.RemoteError("ValueError", "bad", "Traceback ...\n  File \"/secret/path.py\"")
    assert str(err) == "ValueError: bad" and "/secret/path.py" in err.remote_traceback


def test_worker_never_walks_a_pipeline_tree_itself(system):
    from evedesign.remote import worker

    # a pipeline tree may carry user data shaped like specs; only the pipeline's own code reads it
    spec = {"key": "gibbs", "args": {"scorers": [], "kwargs": {"class": "os:system", "wrap": ["x:y"]}}}
    with pytest.raises(ValueError, match="uses_gpu from the orchestrator"):
        worker.Session().op_open_built(spec, system, schema="evedesign_server.schemas:GibbsSamplerSpec")


def test_wrapping_imports_without_the_remote_extra():
    import subprocess
    import sys

    code = ("import sys; sys.modules['msgpack'] = None\n"
            "import evedesign.remote.wrapping, evedesign.remote as r\n"
            "try:\n    r.open_model\nexcept ImportError:\n    print('lazy')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "lazy"


def test_cpus_limit_threads_for_venvs(monkeypatch):
    from evedesign.remote import transport

    from evedesign.remote import launchers

    seen = {}
    monkeypatch.setattr(launchers, "StdioTransport", lambda argv, description, env=None: seen.update(env))
    transport.launch(Target(name="v", launcher="command", command=("python",), cpus=3, env=(("A", "1"),)), None)
    assert seen["OMP_NUM_THREADS"] == seen["OPENBLAS_NUM_THREADS"] == "3" and seen["A"] == "1"


def test_each_gpu_model_gets_its_own_device(monkeypatch):
    from evedesign.remote import transport
    from evedesign.remote.transport import DevicePool, offer_devices, settle_devices

    def assigned(target, uses_gpu):
        """What a worker applies, and what the orchestrator settles, as in api._open."""
        devices = offer_devices(target)
        applied = devices.gpu if uses_gpu else devices.cpu
        settle_devices(devices, applied)
        return applied

    monkeypatch.setattr(transport, "POOL", DevicePool())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")      # what the scheduler gave this job
    docker = Target(name="d", image="img")
    assert gpu_flags(docker) == ["--gpus", '"device=2,3"']
    # CPU models are hidden from the GPUs and take no slot; GPU models alternate
    assert [assigned(docker, g) for g in (False, True, False, True, True)] == ["", "0", "", "1", "0"]
    assert assigned(Target(name="a", launcher="apptainer", image="x.sif"), True) == "3"  # host numbering
    assert assigned(Target(name="s", image="img", gpus="share"), True) is None
    assert assigned(Target(name="p", image="img", gpus="device=1"), True) is None
    assert gpu_flags(Target(name="p", image="img", gpus="device=1")) == ["--gpus", "device=1"]
    assert assigned(Target(name="n", image="img", gpus="none"), True) == ""
    assert assigned(Target(name="m", launcher="modal", image="img"), True) is None

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")         # job got no GPU
    assert gpu_flags(docker) == [] and assigned(docker, True) is None


# --- launchers ---------------------------------------------------------------------

def test_frames_after_the_ready_marker_only():
    from evedesign.remote import protocol

    stream = b"Welcome to gpu-box\nLast login: ...\n" + protocol.READY + protocol.encode({"id": 1, "x": [1, 2]})
    buffer = protocol.FrameBuffer("test")
    frames = [f for i in range(0, len(stream), 5) for f in buffer.feed(stream[i:i + 5])]  # marker split across chunks
    assert frames == [{"id": 1, "x": [1, 2]}]

    import io
    reader = io.BytesIO(stream)
    protocol.read_ready(reader, "test")
    assert protocol.read_frame(reader) == {"id": 1, "x": [1, 2]}
    with pytest.raises(EOFError, match="no worker answered.*command not found"):
        protocol.read_ready(io.BytesIO(b"bash: python: command not found\n"), "test")


def test_a_provider_is_a_small_launcher_plugin(monkeypatch, system):
    """What a package for another provider registers (entry point group evedesign.remote.launchers)."""
    from evedesign.remote import launchers

    class Cloud(Launcher):
        name = "cloud"
        options = {"zone": "eu-west", "machine": None}

        def is_network(self, target):
            return True

        def pin(self, target):
            return {"pinned_image": target.image, "zone": target.options["zone"]}

        def start(self, target, image):
            return InProcessTransport()  # a real one would start a container there and stream to it

    class EntryPoint:
        name = "cloud"

        def load(self):
            return Cloud

    monkeypatch.setattr(launchers, "_plugins", lambda: {"cloud": EntryPoint()})
    monkeypatch.setattr(launchers, "get", functools.cache(launchers.get.__wrapped__))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    infra = Infra.from_dict({"targets": {"c": {"launcher": "cloud", "image": "reg/x@sha256:1", "machine": "a100"}}})
    target = infra.resolve("c")
    assert target.options == {"zone": "eu-west", "machine": "a100"} and target.is_network
    assert infra.resolve("cloud:reg/y:1").image == "reg/y:1"                   # convention names work too
    with pytest.raises(ValueError, match="unknown fields"):
        Infra.from_dict({"targets": {"c": {"launcher": "cloud", "image": "i", "region": "x"}}})

    with open_model("c", MW, infra=infra) as m:
        assert m.provenance["launcher"]["zone"] == "eu-west"
        assert m.provenance["launcher"]["cuda_visible_devices"] is None        # remote: not this host's GPUs
        m.build(system)
        assert m.ready


def test_modal_pulls_ecr_images_with_its_ecr_support():
    from evedesign.remote.modal_launcher import registry_image

    from . import fake_modal

    modal, _ = fake_modal.make(python="python", pythonpath="")
    ecr = registry_image(modal, "123456789012.dkr.ecr.eu-central-1.amazonaws.com/esm2@sha256:ab", "aws-ecr")
    assert ecr.source == "aws_ecr" and ecr.secret.value == "aws-ecr"
    ghcr = registry_image(modal, "ghcr.io/org/esm2@sha256:ab", "ghcr-pull")
    assert ghcr.source == "registry" and ghcr.secret.value == "ghcr-pull"
    assert registry_image(modal, "python:3.12-slim", None).secret is None
    with pytest.raises(ValueError, match="needs registry_secret"):
        registry_image(modal, "123456789012.dkr.ecr.us-east-1.amazonaws.com/x:1", None)
