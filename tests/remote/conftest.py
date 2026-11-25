"""
Fixtures for evedesign.remote.

Most tests need no workers. Those that do are marked `remote` and need either

  EVEDESIGN_TEST_ENVS=/path/envA,/path/envB   two environments with this evedesign
      installed (evedesign[remote]); different numpy/pandas versions make the
      isolation meaningful. Workers run there via the "command" launcher.
  the test images from tests/remote/docker/build_test_images.sh, for containers.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from loguru import logger

SRC = str(Path(__file__).resolve().parents[2] / "src")
ENVS = [e for e in os.environ.get("EVEDESIGN_TEST_ENVS", "").split(",") if e]
IMAGES = ("evedesign-test-isoelectric", "evedesign-test-molweight")

P = "evedesign.restraints.physicochemical:"
MW = {"class": P + "MolecularWeightRestraint", "args": {"lower_bound": 2000, "upper_bound": 2600}}
PI = {"class": P + "IsoelectricPointRestraint", "args": {"lower_bound": 5.0, "upper_bound": 7.0}}


@pytest.fixture(autouse=True, scope="session")
def _quiet():
    logger.remove()
    yield


def env_target(env: str):
    from evedesign.remote import Target

    return Target(name=f"env:{Path(env).name}", launcher="command",
                  command=(f"{env}/bin/python", "-m", "evedesign.remote.worker"))


@pytest.fixture
def system():
    from evedesign.system import Protein, System

    return System([Protein(rep="MKTAYIAKQRQISFVKSHFSRQ", id="A")])


@pytest.fixture
def envs():
    if len(ENVS) < 2:
        pytest.skip("set EVEDESIGN_TEST_ENVS to two environments with evedesign[remote]")
    return [env_target(e) for e in ENVS]


def _image(name: str) -> bool:
    return shutil.which("docker") is not None and subprocess.run(
        ["docker", "image", "inspect", name], capture_output=True).returncode == 0


@pytest.fixture
def infra(tmp_path):
    """Docker targets `iso` and `mw` for the test images, as a TOML file."""
    from evedesign.remote import Infra

    if not all(_image(f"{n}:latest") for n in IMAGES):
        pytest.skip("test images not built (tests/remote/docker/build_test_images.sh)")
    # EVEDESIGN_TEST_MOUNT_SRC=1 runs this checkout's code in the images, without rebuilding them
    mount = os.environ.get("EVEDESIGN_TEST_MOUNT_SRC") == "1"
    extra = f'mounts = ["{SRC}:/opt/src:ro"]\nenv = {{ PYTHONPATH = "/opt/src" }}\n' if mount else ""
    path = tmp_path / "infra.toml"
    path.write_text("[defaults]\nlauncher = 'docker'\n" + "".join(
        f"[targets.{name}]\nimage = '{image}:latest'\n{extra}" for name, image in zip(("iso", "mw"), IMAGES)))
    return Infra.load(path)
