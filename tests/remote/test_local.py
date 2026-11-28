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
from evedesign.remote import Infra, codec, open_built, open_model, reload_model, session
from evedesign.remote.config import Target, derive_seed
from evedesign.remote.protocol import RemoteError
from evedesign.remote.surface import declared_surface, surface_hash
from evedesign.restraints.physicochemical import MolecularWeightRestraint
from evedesign.remote.launchers import Launcher, container_argv, gpu_flags
from evedesign.remote.transport import Transport

from .conftest import MW, P, PI

LOCAL = Target(name="in-process", launcher="command", command=("in-process",))


class InProcessTransport(Transport):
    """A worker Session in this process, behind the real framing and codec."""

    killed = False

    def __init__(self) -> None:
        from evedesign.remote import worker

        super().__init__("in-process")
        self._worker, self._session, self._reply = worker, worker.Session(), None

    def _send(self, data: bytes) -> None:
        message = codec.loads(data[4:])  # one frame: 4-byte length, then the body
        self._reply = codec.loads(codec.dumps(self._worker.handle(self._session, message)))

    def _receive(self):
        return self._reply

    def _terminate(self, kill: bool = False) -> None:
        self.killed = kill


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


def test_score_all_has_no_shared_limit():
    """40 samplers scoring at once, each nesting score_all: no queueing behind a shared pool, no
    deadlock; and the callers' pools go with them"""
    import gc
    import threading
    import time

    from evedesign.utils import score_all

    class Slow:
        def f(self, x): time.sleep(0.2); return x

    class Composite:
        def f(self, x): return sum(score_all([Slow(), Slow()], "f", x))

    results = []
    before = threading.active_count()
    started = time.perf_counter()
    callers = [threading.Thread(target=lambda: results.append(score_all([Composite(), Slow()], "f", 1)))
               for _ in range(40)]
    for c in callers:
        c.start()
    for c in callers:
        c.join(10)
    assert results == [[2, 1]] * 40 and time.perf_counter() - started < 2
    gc.collect()
    assert _eventually(lambda: threading.active_count() <= before)


# --- proxies ---------------------------------------------------------------------

def test_proxy_behaves_like_the_model(local, system):

    with open_model(local, MW, seed=3) as m:
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
        assert m._label == f"{MW['class']}@in-process"


GIBBS = "evedesign.samplers.gibbs:GibbsSampler"


def test_open_built_builds_nested_models_and_returns_ready(local, system):
    with pytest.raises(ValueError, match="opened with open_built"):
        open_model(local, {"class": GIBBS, "models": {"scorers": [MW]}})

    spec = {"class": GIBBS, "args": {"weights": [1.0, 0.5], "num_sweeps": 1}, "models": {"scorers": [PI, MW]}}
    with open_built(local, spec, system) as sampler:
        assert isinstance(sampler, Generator) and sampler.ready and sampler.system is system
        designs = sampler.generate(num_designs=2)
        assert [c["name"] for c in designs[0].metadata["scores"]] == [
            "IsoelectricPointRestraint", "MolecularWeightRestraint"]
        records = sampler.provenance["models"]
        assert [r["class"].rpartition(":")[2] for r in records] == [
            "IsoelectricPointRestraint", "MolecularWeightRestraint", "GibbsSampler"]  # children first
        assert all("stats" in r and "citations" in r for r in records)
        assert len(sampler.fetch_serialized()) == 3
    with pytest.raises(ValueError, match="needs a system"):
        open_built(local, spec)


def test_nested_models_are_explicit():
    from evedesign.remote.spec import as_spec

    with pytest.raises(ValueError, match=r"\['scorers'\] given in both"):
        as_spec({"class": GIBBS, "args": {"scorers": []}, "models": {"scorers": [MW]}})
    with pytest.raises(ValueError, match=r"spec.models.scorers\[1\]: unknown fields \['variant'\]"):
        as_spec({"class": GIBBS, "models": {"scorers": [MW, {"key": "esm2", "variant": "650M"}]}})
    with pytest.raises(TypeError, match=r"spec.models.scorers\[0\]: a spec is a dictionary"):
        as_spec({"class": GIBBS, "models": {"scorers": ["esm2"]}})
    with pytest.raises(ValueError, match="either 'key' or 'class'"):
        as_spec({"key": "a", "class": "b.c:D"})
    with pytest.raises(ValueError, match="'rebuild' only applies"):
        as_spec({"key": "a", "rebuild": True})
    with pytest.raises(ValueError, match="takes only"):
        as_spec({"serialized": b"x", "args": {}})


class Holder(MolecularWeightRestraint):
    """Reports a free-form argument through its declared `name`."""

    def __init__(self, payload, **kwargs):
        super().__init__(**kwargs)
        self.payload = payload

    @property
    def name(self):
        return repr(self.payload)


def test_args_are_data_and_never_searched_for_models(local):
    # user data shaped like a spec stays data: only `models` holds models
    payload = {"class": "os:system", "wrap": ["x:y"], "args": {"cmd": "id"}}
    spec = {"class": f"{__name__}:Holder", "args": {"payload": payload, "lower_bound": 0, "upper_bound": 1}}
    with open_model(local, spec) as m:
        assert m.name == repr(payload)


def test_stored_models_load_at_the_root_and_nested(local, system):
    with open_built(local, MW, system) as m:
        (blob,) = m.fetch_serialized()

    with reload_model(local, blob) as again:                     # short for {"serialized": blob}
        assert again.ready and again.system == system
        assert again.provenance["models"][0]["loaded"] and not again.provenance["models"][0]["rebuilt"]
        assert again.provenance["spec"]["serialized"] == f"<{len(blob)} bytes>"

    other = system.__class__([type(system[0])(rep="MKTAYIAKQRQISFVKSHFSRQAA", id="A")])
    with open_built(local, {"serialized": blob, "rebuild": True}, other) as rebuilt:
        assert rebuilt.system is other and rebuilt.provenance["models"][0]["rebuilt"]
    with pytest.raises(RemoteError, match="does not agree"):
        open_built(local, {"serialized": blob}, other)

    spec = {"class": GIBBS, "args": {"num_sweeps": 1}, "models": {"scorers": [{"serialized": blob}, PI]}}
    with open_built(local, spec, system) as sampler:
        assert [r["loaded"] for r in sampler.provenance["models"]] == [True, False, False]
        sampler.generate(num_designs=1)


CARD = """---
key: mw
name: Molecular weight
model_class: evedesign.restraints.physicochemical:MolecularWeightRestraint
hyper_parameters: {lower_bound: 2000, upper_bound: 2600}
---
"""


def test_the_card_applies_to_its_key_only(local, tmp_path, monkeypatch):
    from evedesign.remote import build

    card = tmp_path / "README.md"
    card.write_text(CARD)
    monkeypatch.setattr(build, "CARD_PATH", card)

    def opened(model):
        with open_model(local, model) as m:
            return m.provenance["models"][0]

    for model in (None, "mw", {"key": "mw"}):
        record = opened(model)
        assert record["via"] == "model_card" and record["args"] == {"lower_bound": 2000, "upper_bound": 2600}
    # a class path means the class as is, even the card's own: the card adds nothing
    record = opened({"class": P + "MolecularWeightRestraint", "args": {"lower_bound": 1, "upper_bound": 2}})
    assert record["via"] == "class" and record["args"] == {"lower_bound": 1, "upper_bound": 2}
    assert opened({"key": "mw", "args": {"upper_bound": 2700}})["args"]["upper_bound"] == 2700   # spec args win
    with pytest.raises(RemoteError, match="the image's model card is for 'mw'"):
        opened("MW")                                                           # exact match only
    with pytest.raises(RemoteError, match="card is for 'mw'"):
        opened("Molecular weight")                                            # the key, not the name
    record = opened({"class": P + "IsoelectricPointRestraint", "args": {"lower_bound": 5, "upper_bound": 7}})
    assert record["via"] == "class"                                            # another class: no card

    card.write_text(CARD.replace("key: mw\n", ""))
    assert opened("Molecular weight")["via"] == "model_card"                  # without a key: the name

    # a card may write its class path dotted, as proteingym-models cards allow
    card.write_text(CARD.replace("physicochemical:Molecular", "physicochemical.Molecular"))
    assert opened(None)["class"] == "evedesign.restraints.physicochemical.MolecularWeightRestraint"


def test_sessions_seed_models_and_refuse_foreign_proxies(local):
    with session(seed=42) as s:
        a = s.open_model(local, {"class": P + "MolecularWeightRestraint", "args": {"lower_bound": 1, "upper_bound": 2}})
        b = s.open_model(local, MW)
        seeds = [x.provenance["environment"]["seed"] for x in (a, b)]
        assert None not in seeds and seeds[0] != seeds[1]                    # derived per model
        assert a.provenance["models"][0]["args"]["lower_bound"] == 1
        with pytest.raises(codec.CodecError, match="open_built"):
            s.open_model(local, {"class": P + "MolecularWeightRestraint", "args": {"other": a}})


def test_open_all_starts_workers_together_and_cleans_up(local, monkeypatch):
    import threading
    import time

    import evedesign.remote.api as api

    started, transports = [], []

    def slow_launch(target, image):
        started.append(time.perf_counter())
        time.sleep(0.2)
        transports.append(InProcessTransport())
        return transports[-1]

    monkeypatch.setattr(api, "launch", slow_launch)
    with session(seed=1) as s:
        pi, mw = s.open_all([(local, PI), (local, MW, {"seed": 5})])
        assert max(started) - min(started) < 0.15 and type(pi).__name__ == "IsoelectricPointRestraint"
        assert mw.provenance["environment"]["seed"] == 5 and not mw.ready
        with pytest.raises(ValueError, match="only 'seed'"):
            s.open_all([(local, MW, {"sed": 1})])

        transports.clear()
        with pytest.raises(RemoteError) as err:
            s.open_all([(local, MW), (local, "no.such.module:Model")])
        assert "entry 1 (no.such.module:Model" in "".join(err.value.__notes__)
        # the good one is closed, once opened: open_all does not wait for it
        assert len(transports) == 2 and _eventually(lambda: transports[0]._closed) and len(s.models) == 2
    assert threading.active_count() >= 1


def test_incompatible_worker_is_refused(local, monkeypatch):
    import evedesign.remote.api as api

    monkeypatch.setattr(api, "surface_hash", lambda: "0" * 16)
    with pytest.raises(RemoteError, match="incompatible worker"):
        open_model(local, MW)


def test_models_are_named_by_a_string_a_spec_or_nothing():
    from evedesign.remote.spec import as_spec, describe

    assert as_spec("esm2") == {"key": "esm2"}
    assert as_spec("esm2:650M") == {"key": "esm2:650M"}                       # no dotted module: a key
    assert as_spec("evedesign.models.esm2:ESM2") == {"class": "evedesign.models.esm2:ESM2"}
    assert as_spec(None) == {} and describe({}) == "card"
    with pytest.raises(TypeError, match="a string, a spec dictionary or nothing"):
        as_spec(MolecularWeightRestraint)


# --- config ----------------------------------------------------------------------

INFRA = {
    "targets": {
        "esm2": {"image": "ghcr.io/proteingym/esm2:2026.11"},
        "eve": {"image": "eve:latest", "gpus": [1], "env": {"A": "1"}, "mounts": ["/w:/models:ro"],
                "start_timeout": 600, "call_timeout": 3600},
        "esmc": {"launcher": "apptainer", "image": "/srv/sif/esmc.sif", "gpus": "none"},
        "esm2-modal": {"launcher": "modal", "image": "ghcr.io/x/esm2@sha256:ab", "gpu_type": "L40S",
                       "memory": 16384, "volumes": {"/models": "w"}},
        "venv": {"launcher": "command", "command": ["/opt/venv/bin/python", "-m", "evedesign.remote.worker"]},
    },
}


def test_shortcuts_name_a_launcher_and_an_image():
    infra = Infra()
    assert infra.resolve("docker:esm2").image == "esm2"                       # as written: no :latest added
    assert infra.resolve("docker:ghcr.io/x/esm2:2026.11").image == "ghcr.io/x/esm2:2026.11"   # first colon only
    assert infra.resolve("docker:localhost:5000/esm2@sha256:ab").image == "localhost:5000/esm2@sha256:ab"
    assert infra.resolve("apptainer:/srv/sif/esm2.sif").image == "/srv/sif/esm2.sif"
    assert infra.resolve("modal:ghcr.io/x/esm2:1").is_network
    assert infra.resolve("docker:esm2").gpus == "auto"
    with pytest.raises(ValueError, match="unknown launcher 'esm2'"):
        Infra.from_dict(INFRA).resolve("esm2:650M")                           # never an infra lookup
    with pytest.raises(ValueError, match="has no shortcut"):
        infra.resolve("command:python")
    with pytest.raises(KeyError, match="unknown target"):
        infra.resolve("esm2")
    with pytest.raises(ValueError, match="may not contain a colon"):
        Infra.from_dict({"targets": {"esm2:gpu": {"image": "i"}}})


@pytest.mark.parametrize("suffix", [".toml", ".yaml", ".json"])
def test_infra_is_plain_data_and_round_trips(tmp_path, suffix):
    import json

    import tomli_w
    import yaml

    assert Infra.from_dict(INFRA).to_dict() == INFRA
    write = {".toml": tomli_w.dumps, ".yaml": yaml.safe_dump, ".json": json.dumps}[suffix]
    (tmp_path / f"infra{suffix}").write_text(write(INFRA))
    assert Infra.coerce(tmp_path / f"infra{suffix}").to_dict() == INFRA
    eve = Infra.from_dict(INFRA).resolve("eve")
    assert eve.gpus == ("1",) and eve.start_timeout == 600 and eve.call_timeout == 3600


def test_one_off_targets_settings_and_typos():
    inline = Infra().resolve({"launcher": "docker", "image": "eve:big", "gpus": "none"})
    assert inline.image == "eve:big" and inline.gpus == "none" and inline.name == "docker:eve:big"
    with pytest.raises(ValueError, match="unknown fields"):
        Infra.from_dict({"targets": {"x": {"imgae": "typo"}}})
    with pytest.raises(ValueError, match="unknown keys"):
        Infra.from_dict({"defaults": {"launcher": "docker"}})                 # no defaults: each target says
    with pytest.raises(ValueError, match=r"unknown fields \['volumes'\] for launcher 'docker'"):
        Infra.from_dict({"targets": {"x": {"image": "i", "volumes": {"/m": "v"}}}})
    with pytest.raises(ValueError, match="use volumes"):
        Infra.from_dict({"targets": {"x": {"launcher": "modal", "image": "i", "mounts": ["/a:/b"]}}})
    with pytest.raises(ValueError, match="only for the command launcher"):
        Infra().resolve({"image": "i", "command": ["serve"]})
    with pytest.raises(ValueError, match="needs a command"):
        Infra().resolve({"launcher": "command"})
    for gpus, match in (("share", "gpus is"), ("device=1", "gpus is"), (1, "gpus is"), ([], "device numbers"),
                        (["a"], "device")):
        with pytest.raises(ValueError, match=match):
            Infra().resolve({"image": "i", "gpus": gpus})
    with pytest.raises(ValueError, match="gpu_type"):
        Infra().resolve({"launcher": "modal", "image": "i", "gpus": [0]})
    for bad in (0, True, "10"):
        with pytest.raises(ValueError, match="positive number"):
            Infra().resolve({"image": "i", "call_timeout": bad})


def test_locks_pin_named_targets_only(tmp_path):
    from evedesign.remote.provenance import write_lock

    infra = Infra.coerce(INFRA, lock={"esm2": "ghcr.io/proteingym/esm2@sha256:ab"})
    assert infra.resolve("esm2").image.endswith("@sha256:ab") and infra.resolve("esm2").pinned
    with pytest.raises(LookupError, match="not in the lock"):
        infra.resolve("eve")
    for target in ("docker:esm2", {"image": "esm2"}):
        with pytest.raises(ValueError, match="only named targets"):
            infra.resolve(target)
    with pytest.raises(ValueError, match="named targets of the infra file only"):
        write_lock(Infra.from_dict(INFRA), ["docker:esm2"], tmp_path / "x.lock")


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
    assert "type=bind,source=/w,target=/models,readonly" in argv
    # the worker by name, never through the image's entrypoint
    assert argv[-5:] == ["--entrypoint", "python", "img@sha256:1", "-m", "evedesign.remote.worker"]
    assert argv[argv.index("--cpus") + 1] == "2.5" and argv[argv.index("--network") + 1] == "none"
    assert {"--cap-drop", "ALL", "no-new-privileges"} <= set(argv) and "--gpus" not in argv
    assert {"HOME=/tmp", "OMP_NUM_THREADS=2"} <= set(argv)
    # target.env by name only: values (tokens) stay out of argv, in the docker CLI's environment
    assert "HF_TOKEN" in argv and not any("secret" in a for a in argv)
    seen = {}
    monkeypatch.setattr(launchers, "StdioTransport",
                        lambda argv, description, env=None, kill_argv=None: seen.update(env, argv=argv, kill=kill_argv))
    monkeypatch.setenv("HOME", "/home/me")
    launchers.get("docker").start(target, "img")
    assert seen["HF_TOKEN"] == "secret" and seen["HOME"] == "/home/me"  # the CLI keeps its own config
    name = seen["argv"][seen["argv"].index("--name") + 1]
    assert seen["kill"] == ["docker", "kill", name]                     # a timeout stops the container


def test_apptainer_execs_the_worker(monkeypatch):
    from evedesign.remote import launchers

    monkeypatch.setattr("evedesign.remote.transport._has_nvidia", lambda: True)
    seen = {}
    monkeypatch.setattr(launchers, "StdioTransport",
                        lambda argv, description, env=None: seen.update(argv=argv, env=env))
    launchers.get("apptainer").start(Target(name="a", launcher="apptainer", image="x.sif", gpus=("2",)), "x.sif")
    assert seen["argv"] == ["apptainer", "exec", "--containall", "--nv", "x.sif",
                            "python", "-m", "evedesign.remote.worker"]
    assert seen["env"]["APPTAINERENV_CUDA_VISIBLE_DEVICES"] == "2"


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

    import types

    seen = {}
    monkeypatch.setattr(launchers, "StdioTransport",
                        lambda argv, description, env=None: seen.update(env) or types.SimpleNamespace())
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
    pinned = Target(name="p", image="img", gpus=(1, 3))
    assert pinned.gpus == ("1", "3") and assigned(pinned, True) is None         # as launched
    assert gpu_flags(pinned) == ["--gpus", '"device=1,3"']
    assert assigned(Target(name="n", image="img", gpus="none"), True) == ""
    assert gpu_flags(Target(name="n", image="img", gpus="none")) == []
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
    assert infra.resolve("cloud:reg/y:1").image == "reg/y:1"                   # shortcuts work too
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
    assert ecr.source == "aws_ecr" and ecr.secret.value == "aws-ecr" and ecr.entrypoint is None
    ghcr = registry_image(modal, "ghcr.io/org/esm2@sha256:ab", "ghcr-pull")
    assert ghcr.source == "registry" and ghcr.secret.value == "ghcr-pull"
    assert registry_image(modal, "python:3.12-slim", None).secret is None
    with pytest.raises(ValueError, match="needs registry_secret"):
        registry_image(modal, "123456789012.dkr.ecr.us-east-1.amazonaws.com/x:1", None)


# --- reproducibility, build_args, wrapped classes -----------------------------------

def test_gibbs_seed_makes_remote_samplers_reproducible(local, system):
    from evedesign.restraints.physicochemical import MolecularWeightRestraint
    from evedesign.samplers.gibbs import GibbsSampler

    with pytest.raises(ValueError, match="either rng or seed"):
        GibbsSampler(scorers=[MolecularWeightRestraint(2000, 2600).build(system, None)],
                     rng=np.random.default_rng(0), seed=0)

    def run(seed):
        # the sampler's own seed is a constructor argument; open_built's `seed` seeds the
        # worker's global generators, which GibbsSampler does not use
        spec = {"class": GIBBS, "args": {"num_sweeps": 1, "seed": seed}, "models": {"scorers": [MW]}}
        with open_built(local, spec, system) as sampler:
            return [str(d[0].rep) for d in sampler.generate(num_designs=4)]

    assert run(7) == run(7) != run(8)


class Recorder(MolecularWeightRestraint):
    """Reports the build arguments it got, through its declared `name`."""

    def build(self, system, data=None, status_callback=None, flag=None):
        self.flag, self.data = flag, data
        return super().build(system, None, status_callback)

    @property
    def name(self):
        return f"flag={getattr(self, 'flag', None)} data={getattr(self, 'data', None)}"


def test_open_model_applies_the_specs_build_args(local, system):
    spec = {"class": f"{__name__}:Recorder", "args": {"lower_bound": 0, "upper_bound": 1},
            "build_args": {"flag": "from-spec"}, "data": "spec-data"}
    with open_model(local, spec) as m:
        assert m.build(system).name == "flag=from-spec data=spec-data"   # the spec's, by default
        assert m.build(system, "mine", flag="here").name == "flag=here data=mine"   # a call overrides them


def subset_like(cls):
    """A wrapper that, like system_subset_model, answers GPU questions through properties."""
    class Wrapper(cls):
        requires_gpu = property(lambda self: True)
        supports_gpu = property(lambda self: True)

    return Wrapper


def test_wrapped_cpu_models_take_no_gpu():
    from evedesign.model import system_subset_model
    from evedesign.remote.worker import _uses_gpu
    from evedesign.remote.wrapping import apply_wraps
    from evedesign.restraints.physicochemical import MolecularWeightRestraint

    assert not _uses_gpu(system_subset_model(MolecularWeightRestraint))          # direct use
    assert not _uses_gpu(apply_wraps(MolecularWeightRestraint, [f"{__name__}:subset_like"]))
    assert _uses_gpu(subset_like(MolecularWeightRestraint))                     # unknown: may use one


# --- timeouts, datasets, workers without the server --------------------------------

def test_start_timeout_stops_a_worker_that_never_answers():
    import sys
    import time

    from evedesign.remote.api import probe

    target = {"launcher": "command", "command": [sys.executable, "-c", "import time; time.sleep(60)"],
              "start_timeout": 1}
    started = time.perf_counter()
    with pytest.raises(TimeoutError, match="command: hello took longer than 1s"):
        probe(target)
    assert time.perf_counter() - started < 10


def test_call_timeout_stops_the_worker(local, monkeypatch, system):
    import time

    import evedesign.remote.api as api

    class Slow(InProcessTransport):
        def _receive(self):
            if self._reply.get("type") == "result" and self.slow:
                time.sleep(5)
            return self._reply

    transport = Slow()
    transport.slow = False
    monkeypatch.setattr(api, "launch", lambda target, image: (setattr(transport, "call_timeout", 0.5), transport)[1])
    m = open_model(local, MW)
    m.build(system)
    transport.slow = True
    with pytest.raises(TimeoutError, match="score took longer than 0.5s; the worker was stopped"):
        m.score([system.rep_to_instance(True)])
    assert transport.killed
    from evedesign.remote.protocol import SessionLost

    with pytest.raises(SessionLost, match="already closed"):
        m.score([system.rep_to_instance(True)])
    m.close()


def test_datasets_cross_the_wire(system):
    from evedesign.dataset import LabeledInstanceDataset

    instances = [system.rep_to_instance(True) for _ in range(6)]
    data = LabeledInstanceDataset(instances, {"y": [0.1, None, 2, 3.5, np.float64(4), 5]}, splits=("cv", 3),
                                  final_train_val_split={"train": [0, 1, 2, 3], "val": [4, 5]})
    out = codec.loads(codec.dumps({"data": data}))["data"]
    assert isinstance(out, LabeledInstanceDataset) and out.serialize() == data.serialize()
    assert out.labels["y"] == [0.1, None, 2.0, 3.5, 4.0, 5.0]
    assert sorted(out.serialize()["splits"]) == ["cv0", "cv1", "cv2"]        # expanded, as plain lists
    assert out.select("y")[2] == data.select("y")[2]

    serialized = data.serialize()
    assert serialized["schema_version"] == "0.1"
    with pytest.raises(ValueError, match="Unable to handle LabeledInstanceDataset version 9.9"):
        LabeledInstanceDataset.deserialize({**serialized, "schema_version": "9.9"})


def test_workers_run_without_evedesign_server(system):
    import sys

    # what a model image holds: evedesign alone
    code = ("import sys; sys.modules['evedesign_server'] = None\n"
            "from evedesign.remote.worker import main; main()")
    target = {"launcher": "command", "command": [sys.executable, "-c", code], "call_timeout": 120}
    spec = {"class": GIBBS, "args": {"num_sweeps": 1, "seed": 1}, "models": {"scorers": [PI, MW]}}
    with open_built(target, spec, system) as sampler:
        assert len(sampler.generate(num_designs=2)) == 2


def test_build_local_uses_prebuilt_children(local, system):
    """A tree built here, with one nested model opened on a target: the sampler runs locally"""
    from evedesign.remote import build_local
    from evedesign.samplers.gibbs import GibbsSampler

    mw, pi = dict(MW), dict(PI)
    gibbs = {"class": "evedesign.samplers.gibbs:GibbsSampler", "args": {"num_sweeps": 2, "seed": 1},
             "models": {"scorers": [mw, pi]}}
    with open_built(local, mw, system) as remote_mw:
        sampler, collection, records = build_local(gibbs, system, prebuilt={id(mw): remote_mw})
        assert isinstance(sampler, GibbsSampler)
        assert sampler.scorers[0] is remote_mw and not hasattr(sampler.scorers[1], "_transport")
        assert [r["via"] for r in records] == ["prebuilt", "class", "class"]
        assert [obj for _, obj in collection][-1] is sampler
        assert len(sampler.generate(num_designs=2)) == 2

    # without prebuilt nodes: the whole tree here, as a worker would build it
    sampler, _, records = build_local(gibbs, system)
    assert [r["via"] for r in records] == ["class", "class", "class"] and sampler.scorers[0].ready


def test_build_local_prebuilt_records_and_checks(local, system):
    import json

    from evedesign.remote import build_local

    mw = dict(MW)
    gibbs = {"class": GIBBS, "args": {"num_sweeps": 1, "seed": 1}, "models": {"scorers": [mw, dict(PI)]}}
    other = system.__class__([type(system[0])(rep="MKTAYIAKQRQISFVKSHFSRQAA", id="A")])
    with open_built(local, mw, system) as remote_mw:
        # the record describes the remote model, as its worker reported it
        _, _, records = build_local(gibbs, system, prebuilt={id(mw): remote_mw})
        assert records[0]["class"] == MW["class"] and records[0]["citations"] == remote_mw.citations
        assert records[0]["loaded"] and not records[0]["rebuilt"]

        # a copy of the tree would miss the entry and build the model here: refused before building
        with pytest.raises(ValueError, match="1 prebuilt entry is not a spec this tree builds"):
            build_local(json.loads(json.dumps(gibbs)), system, prebuilt={id(mw): remote_mw})
        # nor is a node inside another prebuilt one (the outer one is used as given)
        with pytest.raises(ValueError, match="1 prebuilt entry"):
            build_local(gibbs, system, prebuilt={id(gibbs): object(), id(mw): remote_mw})

    with open_built(local, mw, other) as wrong:
        with pytest.raises(ValueError, match="built for a different system"):
            build_local(gibbs, system, prebuilt={id(mw): wrong})


def test_records_follow_a_build(local, system):
    """A model opened unbuilt reports its citations and stats once built, as open_built does"""
    with open_model(local, MW) as m:
        assert m.provenance["models"][0]["stats"] is None
        m.build(system)
        record = m.provenance["models"][0]
        assert record["citations"] == MolecularWeightRestraint.citations and "stats" in record


# --- lifetime, status, Modal transport ------------------------------------------------

def _eventually(check, seconds=5.0):
    import time

    end = time.monotonic() + seconds
    while not check():
        if time.monotonic() > end:
            return False
        time.sleep(0.05)
    return True


# A scorer that takes a while and reports progress; run as the worker's __main__
SLOW = """
import time
from evedesign.restraints.physicochemical import MolecularWeightRestraint

class Slow(MolecularWeightRestraint):
    def score(self, instances, status_callback=None):
        for i in range(5):
            if status_callback is not None:
                status_callback("running", i / 5, None)
            time.sleep(0.3)
        if status_callback is not None:
            status_callback("done", None, "scored")
        return super().score(instances)

from evedesign.remote.worker import main; main()
"""
SLOW_SPEC = {"class": "__main__:Slow", "args": {"lower_bound": 2000, "upper_bound": 2600}}


def _slow_target():
    import sys

    return {"launcher": "command", "command": [sys.executable, "-c", SLOW], "call_timeout": 60}


def test_status_callbacks_cross_the_wire(system):
    seen = []
    with open_model(_slow_target(), SLOW_SPEC) as m:
        m.build(system, status_callback=lambda *a: seen.append(("build", *a)))   # accepted, not required
        scored = m.score([system.rep_to_instance(True)], status_callback=lambda *a: seen.append(a))
        assert len(scored) == 1
        assert [p for s, p, _ in seen if s == "running"] == [0.0, 0.2, 0.4, 0.6, 0.8]
        assert seen[-1] == ("done", None, "scored")
        assert len(m.score([system.rep_to_instance(True)])) == 1                  # without: no messages


def test_close_does_not_wait_for_a_running_call(system):
    import threading
    import time

    from evedesign.remote.protocol import SessionLost

    errors = []
    with open_model(_slow_target(), SLOW_SPEC) as m:
        m.build(system)

        def score():
            try:
                m.score([system.rep_to_instance(True)])
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        caller = threading.Thread(target=score)
        caller.start()
        time.sleep(0.3)
        started = time.monotonic()
        m.close()                                                    # from another thread, mid-call
        assert time.monotonic() - started < 1.0                      # the call needs 1.5 s
        caller.join(5)
        assert isinstance(errors[0], SessionLost)


def _worker(lease: float):
    import sys

    from evedesign.remote.surface import PROTOCOL_VERSION, surface_hash
    from evedesign.remote.transport import StdioTransport

    transport = StdioTransport([sys.executable, "-m", "evedesign.remote.worker"], "worker")
    transport.request("hello", protocol_version=PROTOCOL_VERSION, surface=surface_hash(), lease=lease)
    return transport


def test_a_worker_whose_lease_runs_out_exits(monkeypatch):
    import sys
    import time

    from evedesign.remote.transport import StdioTransport

    monkeypatch.delenv("EVEDESIGN_WORKER_END_CONTAINER", raising=False)
    monkeypatch.setenv("EVEDESIGN_WORKER_LEASE", "1.5")                       # as the Modal launcher sets it
    unheard = StdioTransport([sys.executable, "-m", "evedesign.remote.worker"], "worker")
    monkeypatch.delenv("EVEDESIGN_WORKER_LEASE")
    assert unheard._proc.wait(timeout=10) == 0                      # no hello ever came: gone all the same
    idle, pinged = _worker(1.5), _worker(1.5)
    pinged.keep_alive(1.5)
    try:
        assert idle._proc.wait(timeout=10) == 0                     # no message for 1.5 s: gone
        time.sleep(1)
        assert pinged._proc.poll() is None and pinged.request("ping") is None   # pings keep it
    finally:
        idle.close()
        pinged.close()


def test_an_expired_worker_ends_its_container_only_when_asked(monkeypatch):
    import os
    import signal

    from evedesign.remote import worker

    killed = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
    monkeypatch.delenv("EVEDESIGN_WORKER_END_CONTAINER", raising=False)
    with pytest.raises(SystemExit):
        worker._expire(1)
    assert killed == []                                              # never PID 1 of a plain host
    monkeypatch.setenv("EVEDESIGN_WORKER_END_CONTAINER", "1")
    with pytest.raises(SystemExit):
        worker._expire(1)
    assert killed == [(1, signal.SIGTERM)]


def test_cuda_facts_are_collected_after_devices_are_chosen(local, monkeypatch):
    import os

    from evedesign.remote import provenance

    visible = []
    monkeypatch.setattr(provenance, "collect_cuda",
                        lambda: visible.append(os.environ.get("CUDA_VISIBLE_DEVICES")) or {"cuda": "12.4"})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    hello = provenance.collect_environment()
    assert "cuda" not in hello and "cuda_available" not in hello  # the handshake leaves CUDA alone
    with open_model(local, MW) as m:
        assert visible == [""] and m.provenance["environment"]["cuda"] == "12.4"   # set first, then asked


def test_modal_targets_validate_memory_and_want_digests():
    from evedesign.remote.provenance import pin

    modal = {"launcher": "modal", "image": "ghcr.io/x/esm2@sha256:ab"}
    for memory in (16384, [16384, 32768]):
        Infra().resolve({**modal, "memory": memory})
    for memory in ([1, 2, 3], "16Gi", True, [32768, 16384], 0):
        with pytest.raises(ValueError, match="memory is MiB"):
            Infra().resolve({**modal, "memory": memory})
    assert pin(Infra().resolve(modal))["pinned_by_digest"]
    tagged = {**modal, "image": "ghcr.io/x/esm2:1"}
    with pytest.raises(ValueError, match="pinned by digest"):
        pin(Infra().resolve(tagged))
    assert not pin(Infra().resolve({**tagged, "allow_tags": True}))["pinned_by_digest"]
    for lease in (-1, True, "5"):
        with pytest.raises(ValueError, match="lease"):
            Infra().resolve({**modal, "lease": lease})


def _modal(monkeypatch, **options):
    import sys

    from evedesign.remote.modal_launcher import ModalTransport

    from . import fake_modal
    from .conftest import SRC

    module, calls = fake_modal.make(python=sys.executable, pythonpath=SRC)
    monkeypatch.setitem(sys.modules, "modal", module)
    target = Infra().resolve({"launcher": "modal", "image": "i@sha256:ab", **options})
    return ModalTransport, target, module, calls


def test_modal_sandbox_settings_and_cleanup(monkeypatch):
    ModalTransport, target, module, calls = _modal(monkeypatch, memory=[512, 1024])
    transport = ModalTransport(target, target.image)
    kwargs = calls["sandboxes"][0].kwargs
    transport.close()
    assert kwargs["memory"] == (512, 1024) and kwargs["cpu"] == 1          # default: one core, not 1/8
    flags = kwargs["secrets"][-1].value
    assert flags["EVEDESIGN_WORKER_END_CONTAINER"] == "1" and flags["OMP_NUM_THREADS"] == "1"
    assert flags["EVEDESIGN_WORKER_LEASE"] == "300"

    def broken_exec(self, *command, text=True):
        raise RuntimeError("exec failed")

    monkeypatch.setattr(module.Sandbox, "exec", broken_exec)
    with pytest.raises(RuntimeError, match="exec failed"):
        ModalTransport(target, target.image)
    assert calls["sandboxes"][-1].terminated                                 # not left idling


def test_modal_sends_in_chunks_and_ends_cleanly_on_modal_errors(monkeypatch):
    ModalTransport, target, _, calls = _modal(monkeypatch, memory=[512, 1024])
    transport = ModalTransport(target, target.image)
    writes = []

    class Stdin:
        def write(self, data):
            writes.append(len(data))

        def drain(self):
            writes.append("drain")

    real = transport._proc.stdin
    transport._proc.stdin = Stdin()
    transport._send(b"\0" * (9 << 20))
    assert writes == [4 << 20, "drain", 4 << 20, "drain", 1 << 20, "drain"]   # Modal refuses big writes

    class Gone:
        def write(self, data):
            raise BufferError("Buffer size exceed limit")

    transport._proc.stdin = Gone()
    from evedesign.remote.protocol import SessionLost

    with pytest.raises(SessionLost, match="memory limit of 1024 MiB"):
        transport.request("ping")
    calls["sandboxes"][0].terminate = lambda wait=False: (_ for _ in ()).throw(RuntimeError("not found"))
    transport._proc.stdin = real
    transport.close()                                                         # an ended Sandbox: no error
