"""
evedesign.remote across real process boundaries: two environments with
different numpy/pandas, a Modal sandbox (stand-in running the real worker),
and containers. Kept to a few worker starts; proxy behaviour is in test_local.
"""

import os
import sys

import numpy as np
import pytest

from evedesign.remote import open_model, session

from . import fake_modal
from .conftest import ENVS, MW, P, PI, SRC

pytestmark = pytest.mark.remote


def _gibbs(scorers, seed=0):
    from evedesign.samplers.gibbs import GibbsSampler

    designs = GibbsSampler(scorers=scorers, weights=[1.0, 0.5], num_sweeps=1,
                           rng=np.random.default_rng(seed)).generate(num_designs=4)
    return [(str(d[0].rep), round(d.score, 8), [c["name"] for c in d.metadata["scores"]]) for d in designs]


def _local_scorers(system):
    from evedesign.restraints.physicochemical import IsoelectricPointRestraint, MolecularWeightRestraint

    return [IsoelectricPointRestraint(5.0, 7.0).build(system, None),
            MolecularWeightRestraint(2000, 2600).build(system, None)]


def _numpy(record):
    return next(p for p in record["environment"]["packages"] if p.lower().startswith("numpy=="))


def test_scorers_in_two_environments_match_local(envs, system):
    local = _gibbs(_local_scorers(system))
    with session() as s:
        scorers = [s.open_model(envs[0], PI), s.open_model(envs[1], MW)]
        s.build_all(system)
        assert _gibbs(scorers) == local
        assert _numpy(scorers[0].provenance) != _numpy(scorers[1].provenance)   # really isolated


def test_modal_sandbox(monkeypatch, system):
    if len(ENVS) < 2:
        pytest.skip("set EVEDESIGN_TEST_ENVS to two environments with evedesign[remote]")
    module, calls = fake_modal.make(python=f"{ENVS[1]}/bin/python", pythonpath=SRC)
    monkeypatch.setitem(sys.modules, "modal", module)
    infra = {"targets": {"mw-modal": {
        "launcher": "modal", "image": "ghcr.io/org/molweight@sha256:abc", "gpu": "L40S",
        "registry_secret": "ghcr-pull", "secrets": ["huggingface"], "env": {"LOG_LEVEL": "x"},
        "volumes": {"/models": "esm2-weights"}, "cpus": 4}}}

    pi, _ = _local_scorers(system)
    local = _gibbs(_local_scorers(system))
    with session(infra, seed=3) as s:
        mw = s.open_model("mw-modal", P + "MolecularWeightRestraint", lower_bound=2000, upper_bound=2600)
        mw.build(system)
        assert _gibbs([pi, mw]) == local                    # chunked stream reassembled exactly
        assert mw.provenance["launcher"]["pinned_by_digest"] is True
        assert mw.provenance["launcher"]["volumes"] == {"/models": "esm2-weights"}   # outside the image
        sandbox = calls["sandboxes"][0]

    kwargs = sandbox.kwargs
    assert kwargs["image"].ref == "ghcr.io/org/molweight@sha256:abc"
    assert kwargs["image"].secret.value == "ghcr-pull"      # private registry credentials
    assert kwargs["gpu"] == "L40S" and kwargs["app"] == "evedesign-remote"
    assert [x.value for x in kwargs["secrets"]][:2] == ["huggingface", {"LOG_LEVEL": "x"}]
    assert kwargs["secrets"][2].value["OMP_NUM_THREADS"] == "4" and kwargs["cpu"] == 4
    assert {path: v.name for path, v in kwargs["volumes"].items()} == {"/models": "esm2-weights"}
    assert sandbox.terminated                              # session end stops the container


def test_containers_resolve_cards_and_record_provenance(infra, system, tmp_path):
    from evedesign.remote.config import read_lock
    from evedesign.remote.protocol import RemoteError
    from evedesign.remote.provenance import write_lock

    local = _gibbs(_local_scorers(system))
    with session(infra) as s:
        iso = s.open_model("iso", {"key": "isoelectric"})  # constructor args from the image's card
        mw = s.open_model("mw")                             # no key: the image's own model
        s.build_all(system)
        assert _gibbs([iso, mw]) == local
        iso_rec, mw_rec = s.provenance()
    assert iso_rec["models"][0]["via"] == "model_card" and iso_rec["models"][0]["args"]["lower_bound"] == 5.0
    assert "sha256:" in iso_rec["launcher"]["pinned_image"]
    assert _numpy(iso_rec) != _numpy(mw_rec)

    # a card does not claim other keys
    with pytest.raises(RemoteError, match="no 'class' given"):
        open_model("iso", {"key": "esm2"}, infra=infra)

    # locks pin by content and refuse what they do not list
    lock = tmp_path / "infra.lock"
    pinned = write_lock(infra, ["iso"], lock)["models"][0]["launcher"]["pinned_image"]
    locked = infra.with_lock(read_lock(lock))
    assert locked.resolve("iso").image == pinned
    with pytest.raises(LookupError, match="not in the lock"):
        locked.resolve("mw")


SSH = """#!/bin/sh
# stand-in for ssh: a login banner on stdout, then the remote command, run here
while [ $# -gt 0 ]; do case "$1" in -T|-q) shift;; -o|-p|-i) shift 2;; *) break;; esac; done
echo "Welcome to $1"; echo "Last login: Sun Nov 22 09:00:00 2026"; shift
exec "$@"
"""
SRUN = """#!/bin/sh
# stand-in for srun: allocation messages on stderr, then the job step, run here
while [ $# -gt 0 ]; do case "$1" in --*) shift;; -N|-n|-p|-t) shift 2;; *) break;; esac; done
echo "srun: job 4711 queued and waiting for resources" >&2
exec "$@"
"""


def test_workers_through_ssh_and_srun(tmp_path, monkeypatch, system):
    """The command launcher reaches workers elsewhere; output before the worker starts is skipped."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, script in (("ssh", SSH), ("srun", SRUN)):
        (bin_dir / name).write_text(script)
        (bin_dir / name).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    worker = [sys.executable, "-m", "evedesign.remote.worker"]
    infra = {"targets": {
        "gpu-box": {"launcher": "command", "command": ["ssh", "-T", "gpu-box", *worker], "network": True},
        "cluster": {"launcher": "command", "command": ["srun", "--gres=gpu:1", "-t", "60", *worker],
                    "network": True},
    }}
    with session(infra) as s:
        models = [s.open_model(name, MW) for name in ("gpu-box", "cluster")]
        s.build_all(system)
        scores = [m.score([system.rep_to_instance(True)])[0].score for m in models]
    assert scores[0] == scores[1]
    assert all(m.provenance["launcher"]["not_hermetic"] for m in models)
    assert all(m.provenance["launcher"]["cuda_visible_devices"] is None for m in models)  # not this host's GPUs
