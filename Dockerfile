# Reproducible CPU environment for the FR3 HOCBF gain-learning pipeline.
# Also used by .devcontainer/devcontainer.json (VS Code "Reopen in Container").
#
#   docker build -t fr3-hocbf .
#   docker run --rm -it -v "$PWD/out:/out" fr3-hocbf                                  # shell
#   docker run --rm -v "$PWD/out:/out" fr3-hocbf /repo/docker/reproduce_tables.sh      # Tables I-III + Fig. 4
#
# Layout: code at /repo (baked in here; bind-mounted live in the dev container),
# evaluation scenarios at /data (outside /repo so the dev-container mount does
# not hide them).
#
# Python 3.10.12 is the validated environment (see requirements.txt).
FROM python:3.10.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

# git for source control inside the dev container;
# libquadmath0 is required by cmeel-boost's charconv/locale libraries
RUN apt-get update \
 && apt-get install -y --no-install-recommends git libquadmath0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /repo

# The base image's pip (23.x) rejects PyTorch-index wheels whose metadata name
# differs in case/separators (Jinja2, typing_extensions) and then fails building
# them from source, so upgrade pip first.
# CPU-only torch next, so the pinned torch==2.11.0 in requirements.txt is
# already satisfied by the +cpu wheel and pip does not pull the CUDA stack.
COPY requirements.txt .
RUN python -m pip install --upgrade "pip>=24.2" \
 && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0 \
 && pip install -r requirements.txt

# Unpack the 1000 evaluation scenarios -> /data/generated_scenarios/
# and rewrite the 460-scenario benchmark list -> /data/pd_failed_460.txt.
COPY artifacts/scenarios/generated_scenarios_eval.zip artifacts/results/fr3_pd_only_all1000_failed_scenarios.txt /tmp/
RUN python -m zipfile -e /tmp/generated_scenarios_eval.zip /data/ \
 && python -c "from pathlib import Path; \
root = Path('/data/generated_scenarios'); \
names = [Path(l.strip()).name for l in Path('/tmp/fr3_pd_only_all1000_failed_scenarios.txt').read_text().splitlines() if l.strip()]; \
missing = [n for n in names if not (root / n).is_file()]; \
assert not missing, f'missing scenarios: {missing[:5]}'; \
Path('/data/pd_failed_460.txt').write_text('\n'.join(str(root / n) for n in names) + '\n'); \
print(len(names), 'scenarios')" \
 && rm /tmp/generated_scenarios_eval.zip /tmp/fr3_pd_only_all1000_failed_scenarios.txt

COPY . .
RUN sed -i 's/\r$//' docker/reproduce_tables.sh && chmod +x docker/reproduce_tables.sh

WORKDIR /repo/src
CMD ["bash"]
