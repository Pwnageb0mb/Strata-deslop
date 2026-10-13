# syntax=docker/dockerfile:1
#
# Strata: Qwen3.8-Flash-Next on NVIDIA GPUs (RTX 30/40/50, 12+ GB VRAM; two or
# three cards can share one model, 8 GB each - docs/MULTI_GPU.md).
#
# The engine is compiled during docker build, so the first container start only
# downloads the model (~70 GB) and starts the server.  The build never downloads llama.cpp:
# it must be in the build context at third_party/llama.cpp (the pinned commit). docker build has no GPU,
# so the CUDA architectures are fixed here instead of read from nvidia-smi: the
# engine is a fat binary with a cubin per listed arch, and the runtime picks the
# one matching your card. Narrow CUDA_ARCHITECTURES to your card for a faster
# build; a card outside the set needs a rebuild with its own arch.
#
# Build:
#   docker build -t strata .
#   docker build -t strata --build-arg CUDA_ARCHITECTURES=89 .        # RTX 40 only
#
# Volta / Pascal (Tesla V100, Titan V, P100, P40 - compute capability below 7.5): CUDA 13 dropped them, so
# the image must be built on a CUDA 12 base with the experimental arch, and the engine goes to engine-cuda12/
# (docs/NVIDIA_V100.md, docs/OLDER_GPUS.md).  STRATA_CUDA_TOOLKIT=12 also adds -DSTRATA_EXPERIMENTAL_SM60=ON,
# which lowers the compute-capability floor, and defaults CUDA_ARCHITECTURES to 70:
#   docker build -t strata:v100 \
#     --build-arg CUDA_VERSION=12.9.1 --build-arg STRATA_CUDA_TOOLKIT=12 --build-arg CUDA_ARCHITECTURES=70 .
# A card below 7.5 is the oldest card the model runs on, so a mixed PC builds for all of them (e.g. "70;86").
#
# Run (host needs an NVIDIA driver >= 580 and nvidia-container-toolkit):
#   docker run --rm --gpus all \
#     -p 8080:8080 \
#     --ulimit memlock=-1 \
#     -v strata-data:/data \
#     -e MODEL=IQ2_XS \
#     strata
#
# The image is built in two stages: the CUDA toolkit compiles the engine in the first, and the second
# (the CUDA *runtime* image) keeps only what runs - no nvcc, no Nsight Compute, no build tools. The server,
# the engine and the image encoder run as the unprivileged user "strata" (uid 1000). Rebuild the image when a
# card outside CUDA_ARCHITECTURES has to run it: the runtime image has no compiler of its own.
#
# Setup choices are env vars, read by docker-entrypoint.sh: FAMILY, MODEL, CONTEXT,
# VISION (no | yes | cpu), KV (int8 | q4_0 | k8v4), GPU (one card) or GPUS ("0,2"
# or "all", with LAYER_SPLIT), LOW_RAM (auto | on | off), HOST, PORT, API_KEY,
# GGUF_DIR (GGUF files you already have), RESIDENT_BUDGET_GIB, KV_STREAMING.
#
# Only the model files, the prepared pack, the MTP layer and the install config
# live in the /data volume; the engine is part of the image. Strata loads 32-62 GB
# into RAM, so a capped container needs -e LOW_RAM=on: setup.py reads the RAM from
# /proc/meminfo, which here is the host's total, not the container's limit. Add an
# API key before exposing the port to a network: -e API_KEY=<secret>. Pass
# -e REINSTALL=1 to change the model settings later.
#
# --gpus all on a host with two usable cards: setup takes both (the layer split is
# its recommended default). Pin one card with -e GPU=0, or name them with
# -e GPUS=0,2. A volume set up for one card switches to the pair on its first start
# on a two-card host unless GPU or GPUS pins it. LOW_RAM=on runs on one card.

ARG CUDA_VERSION=13.0.0

# --- build stage: only this one has the CUDA toolkit.  It compiles the engine and the image encoder and
# makes the venv; the runtime image below gets the results and none of the toolchain.
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build

# STRATA_EXECV=1: setup.py replaces itself with the server, so the server is PID 1
# and docker stop's SIGTERM reaches it (see setup.start). Normal Linux starts, which
# don't set it, keep spawning the server as a child.
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 LANG=C.UTF-8 STRATA_EXECV=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git libatomic1 libgomp1 \
        python3 python3-pip python3-venv unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/strata
COPY . .

# RTX 20 (75), RTX 30 (86), RTX 40 (89), RTX 50 (120), plus 80 for A-series. CMakeLists
# refuses anything below 75 unless STRATA_CUDA_TOOLKIT=12 lowers the floor (Volta 70, Pascal 60/61).
# BUILD_VISION=0 skips the image encoder build.  An empty CUDA_ARCHITECTURES: 70 for the CUDA 12
# toolkit, else the list above.
ARG CUDA_ARCHITECTURES=
ARG STRATA_CUDA_TOOLKIT=13
ARG STRATA_BUILD_JOBS=all
ARG BUILD_VISION=1

RUN python3 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir --upgrade pip \
    && .venv/bin/pip install --no-cache-dir -r requirements.txt \
    && chmod +x setup.sh docker-entrypoint.sh

# llama.cpp is not fetched here: the build context carries it at third_party/llama.cpp (the pinned commit).
# The engine, the server's tokenizer library (libstrata_text.so) and the image encoder are built
# exactly the way setup.py builds them: native code for the CPU that builds the
# image, so build it on the PC it runs on. BUILD.json is what setup.py reads to
# decide whether an engine is current: source=local with a matching src hash
# means the first start reuses it instead of recompiling.
RUN .venv/bin/python - <<'PYEOF'
import json, os, pathlib, shutil
import setup

llama = setup.get_llama_cpp()
nvcc, _ = setup.find_nvcc()
toolkit = int(os.environ.get("STRATA_CUDA_TOOLKIT", "13"))
arch = (os.environ.get("CUDA_ARCHITECTURES", "").strip().strip('"').replace(",", ";")
        or ("70" if toolkit == 12 else "75;80;86;89;120"))
archs = [int(a.split("-")[0]) for a in arch.split(";") if a.split("-")[0].isdigit()]
vision = "gpu" if os.environ.get("BUILD_VISION", "1") == "1" else "none"
eng = setup.engine_dir(toolkit)                       # engine/, or engine-cuda12/ for the experimental build

setup.cmake_build(setup.ROOT, setup.ROOT / "build", "strata",
    ["-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_BUILD_TESTS=OFF",
     f"-DCMAKE_CUDA_ARCHITECTURES={arch}", f"-DCMAKE_CUDA_COMPILER={nvcc}",
     *setup.engine_defs(archs, toolkit),              # -DSTRATA_EXPERIMENTAL_SM60=ON under sm_75 / CUDA 12
     f"-DSTRATA_GGML_DIR={llama}"])
# the server's tokenizer (C++, CPU only), next to the engine; same build dir and options, so nothing is configured twice
setup.cmake_build(setup.ROOT, setup.ROOT / "build", "strata_text",
    ["-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_BUILD_TESTS=OFF",
     f"-DCMAKE_CUDA_ARCHITECTURES={arch}", f"-DCMAKE_CUDA_COMPILER={nvcc}",
     *setup.engine_defs(archs, toolkit),
     f"-DSTRATA_GGML_DIR={llama}"])
if vision != "none":
    setup.cmake_build(setup.ROOT / "tools" / "vision", setup.ROOT / "build-vision", "strata-vision",
        [f"-DLLAMA_DIR={llama}", "-DSTRATA_VISION_CUDA=ON", "-DSTRATA_PORTABLE=OFF",
         f"-DCMAKE_CUDA_ARCHITECTURES={arch}", f"-DCMAKE_CUDA_COMPILER={nvcc}"])

eng.mkdir(exist_ok=True)
shutil.copy2(setup.ROOT / "build" / setup.EXE, eng / setup.EXE)
shutil.copy2(setup.ROOT / "build" / "libstrata_text.so", eng / "libstrata_text.so")
if vision != "none":
    shutil.copy2(setup.ROOT / "build-vision" / "bin" / setup.VEXE, eng / setup.VEXE)
bindir = pathlib.Path(nvcc).parent
meta = {"source": "local", "version": setup.source_version(),
        "archs": archs, "vision": vision,
        **({"toolkit": 12} if toolkit == 12 else {}),
        "cuda_dirs": [str(d) for d in (bindir, bindir / "x64", bindir.parent / "lib64") if d.is_dir()],
        "src": setup.source_hash(setup.ENGINE_SOURCES),
        "vision_src": setup.source_hash(setup.VISION_SOURCES) if vision != "none" else None}
(eng / "BUILD.json").write_text(json.dumps(meta, indent=1))
PYEOF

# the cmake trees are build-time only; the engine itself is what the container needs.  The cmake and ninja pip
# packages were only for the build, and the runtime copy of the venv does not need them.  Tests, bench and
# review state are not read at run time (the engine/include/src/cmake/third_party trees ARE: setup.py hashes
# them to decide the engine is current).  The code is byte-compiled here because the service user cannot write
# __pycache__ into it.
RUN rm -rf build build-vision \
    && .venv/bin/pip uninstall -y cmake ninja \
    && rm -rf tests bench sec-reports ref \
    && .venv/bin/python -m compileall -q setup.py serve tools

# ------------------------------------------------------------------------------------------------ runtime
# The CUDA *runtime* image: the libraries the engine loads, no nvcc, no Nsight Compute, no build tools and
# no system pip.  What is left over from the build stage is the source, the engine, the venv and llama.cpp.
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 LANG=C.UTF-8 STRATA_EXECV=1

# python3: the venv's interpreter is a link to it.  curl: the healthcheck.  libatomic1/libgomp1: the engine
# and ggml's CPU kernels.  No python3-pip: the venv has its own, and the system one is only needed to make it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl libatomic1 libgomp1 python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/strata
COPY --from=build /opt/strata /opt/strata

# A non-root user runs the server, the engine and the image encoder (trivy DS-0002).  The code, the engine and the
# venv stay root-owned, so the service user cannot rewrite them.  /opt/strata itself is root:strata 1775 (sticky): the
# user can create the files setup.py and the server write next to the program (run-<model>.sh, strata-<model>.json
# and its .log / .shared-settings.json / .tmp, tmp/) and replace its own, but not rename or delete a root-owned one.
# The model files go to /data, which is this user's; a named volume takes the ownership of that directory when it is
# first mounted.  No --uid: the CUDA images already ship uid 1000 (the "ubuntu" user), and a fixed number would collide with it.
RUN useradd --create-home --shell /usr/sbin/nologin strata \
    && mkdir -p /data \
    && chown root:strata /opt/strata && chmod 1775 /opt/strata \
    && chown strata:strata /data
USER strata

VOLUME ["/data"]
EXPOSE 8080

# /health is answered before the API key gate, so it works with or without one.
# The port only opens after the model loads (1-3 minutes, longer on a first run),
# so the start period is generous: a too short one marks a still-loading container
# unhealthy and a restart policy would kill it mid-download.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD curl -fs "http://127.0.0.1:${PORT:-8080}/health" || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]
