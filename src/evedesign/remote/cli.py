"""
evedesign-remote: set up and verify a site, and pin it.

    evedesign-remote build <zoo-dir> [model ...] [--apptainer --sif-dir DIR]
    evedesign-remote check [target ...] [--infra infra.toml] [--lock infra.lock]
    evedesign-remote lock  [target ...] --infra infra.toml -o infra.lock

Pipeline steps (execute-step, resources) are evedesign-server's `evedesign-pipeline`.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from .config import Infra


def _infra(args) -> Infra:
    return Infra.load(getattr(args, "infra", None), getattr(args, "lock", None))


def cmd_build(args) -> int:
    zoo = Path(args.zoo)
    models = args.models or sorted(p.name for p in zoo.iterdir() if (p / "Dockerfile").exists())
    failed = []
    for model in models:
        folder = zoo / model
        if args.apptainer:
            sif = Path(args.sif_dir) / f"{model}.sif"
            cmd = ["apptainer", "build", "--force", str(sif), str(folder / "Singularity.def")]
        else:
            cmd = [args.runtime, "build", "-t", f"{model}:{args.tag}", str(folder)]
        print("+", " ".join(cmd), flush=True)
        if subprocess.run(cmd).returncode != 0:
            failed.append(model)
    if failed:
        print(f"failed: {failed}", file=sys.stderr)
    return 1 if failed else 0


def cmd_check(args) -> int:
    from .api import probe

    infra = _infra(args)
    names = args.targets or infra.names()
    if not names:
        print("no targets: name some, or define them in infra.toml", file=sys.stderr)
        return 2
    status = 0
    for name in names:
        try:
            result = probe(name, infra)
            env, launcher = result["environment"], result["launcher"]
            flags = [f for f in ("local_only", "not_hermetic") if launcher.get(f)]
            print(f"ok    {name:24} {launcher.get('pinned_image', launcher.get('argv'))} "
                  f"evedesign={env['evedesign']} python={env['python']} "
                  f"gpus={env.get('gpus', [])} cuda={env.get('cuda_available')} {' '.join(flags)}")
        except Exception as exc:  # noqa: BLE001 -- report every target
            status = 1
            print(f"FAIL  {name:24} {type(exc).__name__}: {str(exc).splitlines()[0]}")
    return status


def cmd_lock(args) -> int:
    from .provenance import write_lock

    infra = Infra.load(args.infra)
    data = write_lock(infra, args.targets or infra.names(), args.output)
    for model in data["models"]:
        print(f"{model['launcher']['target']:24} {model['launcher']['pinned_image']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evedesign-remote")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build", help="build images for proteingym-models folders")
    p.add_argument("zoo")
    p.add_argument("models", nargs="*")
    p.add_argument("--runtime", default="docker")
    p.add_argument("--tag", default="latest")
    p.add_argument("--apptainer", action="store_true")
    p.add_argument("--sif-dir", default=".")
    p.set_defaults(func=cmd_build)

    for name, func, help_ in (("check", cmd_check, "start each target and report what runs"),
                              ("lock", cmd_lock, "pin targets to content hashes")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("targets", nargs="*")
        p.add_argument("--infra")
        if name == "check":
            p.add_argument("--lock")
        else:
            p.add_argument("-o", "--output", default="infra.lock")
        p.set_defaults(func=func)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
