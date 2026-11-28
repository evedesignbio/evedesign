# Test images only (build_test_images.sh): a prebuilt environment on a locally
# imported base, because the test machine cannot pull from registries. Same
# conventions as the proteingym-models template: the environment's python first
# on PATH, the model card at /opt/program/README.md. The entrypoint is deliberately
# not the worker: launchers start `python -m evedesign.remote.worker` themselves.
FROM evedesign-test-base:py312
COPY env /opt/env
COPY README.md /opt/program/README.md
ENV PATH=/opt/env/bin:/usr/bin:/bin PYTHONUNBUFFERED=1 HOME=/tmp
# Containers run as the caller's uid and cannot write bytecode into /opt/env:
# without this, every start recompiles numpy, pandas, sklearn, ... (tens of seconds)
RUN python -m compileall -q -j 0 /opt/env/lib
ENTRYPOINT ["sh", "-c", "echo 'not the worker' >&2; exit 3"]
