"""
A stand-in for the `modal` package, enough to run ModalTransport for real: the
Sandbox's exec starts the actual worker as a local subprocess, and its stdout is
handed back in small, odd-sized chunks, as a network stream would deliver it.
Records what was requested so tests can check it.
"""

import os
import subprocess
import types


def make(python: str, pythonpath: str):
    calls: dict = {"sandboxes": []}

    class Secret:
        def __init__(self, kind, value):
            self.kind, self.value = kind, value

        @classmethod
        def from_name(cls, name):
            return cls("name", name)

        @classmethod
        def from_dict(cls, values):
            return cls("dict", dict(values))

    class Image:
        def __init__(self, ref, secret, source="registry"):
            self.ref, self.secret, self.source = ref, secret, source

        @classmethod
        def from_registry(cls, ref, secret=None):
            return cls(ref, secret)

        @classmethod
        def from_aws_ecr(cls, ref, secret):
            return cls(ref, secret, "aws_ecr")

    class Volume:
        def __init__(self, name):
            self.name = name

        @classmethod
        def from_name(cls, name, create_if_missing=False):
            return cls(name)

    class App:
        @staticmethod
        def lookup(name, create_if_missing=False):
            calls["app"] = name
            return name

    class _Writer:
        def __init__(self, stream):
            self._stream = stream

        def write(self, data):
            self._stream.write(data)

        def drain(self):
            self._stream.flush()

    class _Reader:
        def __init__(self, stream, sizes=(7, 1, 300, 4096)):
            self._stream, self._sizes = stream, sizes

        def __iter__(self):
            i = 0
            while chunk := os.read(self._stream.fileno(), self._sizes[i % len(self._sizes)]):
                i += 1
                yield chunk

    class _Process:
        def __init__(self, proc):
            self._proc = proc
            self.stdin = _Writer(proc.stdin)
            self.stdout = _Reader(proc.stdout)
            self.stderr = _Reader(proc.stderr, (4096,))

    class Sandbox:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.procs = []
            self.terminated = False

        @classmethod
        def create(cls, **kwargs):
            sandbox = cls(**kwargs)
            calls["sandboxes"].append(sandbox)
            return sandbox

        def exec(self, *command, text=True):
            assert text is False, "the protocol is binary"
            command = [python if c == "python" else c for c in command]
            env = {**os.environ, "PYTHONPATH": pythonpath}
            for secret in self.kwargs.get("secrets", []):
                if secret.kind == "dict":
                    env.update(secret.value)
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, env=env, bufsize=0)
            self.procs.append(proc)
            return _Process(proc)

        def terminate(self, wait=False):
            self.terminated = True
            for proc in self.procs:
                if proc.poll() is None:
                    proc.kill()

    module = types.ModuleType("modal")
    module.App, module.Image, module.Secret, module.Sandbox, module.Volume = App, Image, Secret, Sandbox, Volume
    return module, calls
