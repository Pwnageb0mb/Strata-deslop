# Security audit (local fork, October 2026)

Scope: the whole checkout at commit `7695859a` plus the changes in this fork, audited on this machine (2x Tesla
PG500-216, Ubuntu, Docker 29.7.2) because the goal is to run it here in a container. This is a **first pass with
automated tools and a manual read of the network and request-handling paths**, not a replacement for the outside
audit `SECURITY.md` says has not happened yet.

## What was run

| Tool | Version | What it covers | Result |
| --- | --- | --- | --- |
| bandit | 1.6.2 | Python, static | 271 issues (3 High, 120 Medium, 148 Low) over 46,770 lines |
| semgrep | 1.180.0 | Python + Dockerfile, `--config auto` | 123 findings (2 ERROR, 121 WARNING) |
| gitleaks | 8.28.0 | secrets in the working tree and history | 44 hits, **1 real credential** |
| pip-audit | 2.10.1 | `requirements.txt` against the advisory databases | none |
| cppcheck | 2.13.0 | C++/CUDA, the engine and the vision encoder | 2 errors, 2 warnings, 1297 style/portability notes (5,376 lines of output) |
| trivy | 0.75.0 | the built container image (CVEs, misconfig) | 621 HIGH/CRITICAL, 473 of them with a fix; 1 misconfiguration in this repo's Dockerfile |

Raw output is in `sec-reports/` (`bandit.txt`, `semgrep.json`, `gitleaks.json`, `cppcheck.txt`). Rerun with the
same commands; nothing here needs a GPU.

```
bandit -r . -x ./third_party,./bench,./.git -f txt -o sec-reports/bandit.txt
semgrep scan --config auto --metrics=on --json -o sec-reports/semgrep.json --exclude third_party --exclude bench
gitleaks detect --source . --report-format json --report-path sec-reports/gitleaks.json
pip-audit -r requirements.txt
cppcheck --enable=warning,style,performance,portability --inconclusive --force --std=c++17 -I include -I src src include tools/vision
```

## Findings

### 1. A live third-party bearer token is committed (High, real)

`CRAWL4AI_BEARER_TOKEN` is stored in plain text in the benchmark results that are tracked in git:

```
bench/results/2026-10-04-community-rtx4090-iq3xxs-200k-{code,ru}/{config,env,runs}.json
bench/results/2026-10-08-community-rtx4090-iq3xxs-200k-{code,ru}/...
```

**Scrubbed.** The value is replaced with `<redacted: leaked credential>` in all 16 tracked files — the fourteen
bench reports above, plus the two `sec-reports/gitleaks.json` copies that this scan itself produced — and
gitleaks now reports no bearer token anywhere in the tree.

It was never this repository's credential, and it is not a cloud key. The config around it reads
a `CRAWL4AI_ENDPOINT` on a private (RFC1918) address, so it is a bearer token for a **self-hosted crawl4ai on
the reporter's private LAN**. Nothing to rotate here — the endpoint is
RFC1918 and not routable from the internet, so the token is close to useless off that network — but it should
not have been committed, and it was: the copies in git history stay unless the history is rewritten.

It is also **not a dependency.** crawl4ai appears nowhere in `setup.py`, `pyproject.toml`, `requirements*.txt`,
the `Dockerfile` or `src/`. It sits inside the reporter's own `mcp_servers` block, and Strata's MCP support is a
*client*: `serve/mcp.py` takes the server list from the run config. Strata ships no search tool of any kind.

**The rest of what the bench reports carry is not a secret, and is left alone.** `env.json` files record the
machine a benchmark ran on, which is their entire purpose: hundreds of home-directory paths and a handful of
private LAN addresses. Masking those would falsify upstream's measurements to hide information that
is not exploitable. The distinction the scrub draws: **a credential is never data; a home directory in an
environment snapshot is.**

**And the mechanism is fixed, not just the instance.** `tools/strata_audit.py` ran gitleaks straight into a
tracked file, so every future scan would have re-leaked whatever it found. It now redacts the `Secret` and
`Match` fields before writing: the report stops being a copy of the secret without stopping being a finding
(`RuleID`, `File`, `StartLine` and the counts are kept). gitleaks is the only report that needs this — it is the
only scanner whose output is a secret by definition; semgrep quotes source lines already in the tree, and trivy
reports packages.

A test key `KEY="z840-strata-key"` in
`bench/results/2026-10-03-community-z840-rtx3090ti/strata-numa-ab.sh` is benchmark-only and harmless.

The other 42 gitleaks hits are false positives: SHA-256 digests in manifests and GGUF tensor names
(`blk.1.ple_key.weight`), which merely look like `key = "..."`.

### 2. SSRF through an image URL (Medium, fixed)

`Vision.download` fetched any `http(s)://` URL an OpenAI request named, from the server. A client could point it
at this PC's own services, at another container on the Docker network, or at a cloud metadata endpoint
(`169.254.169.254`) and read the answer back as image bytes.

Fixed by removing the fetch: `serve/server.py` no longer fetches image URLs and no longer reads local file paths as images. Only `data:` URLs are read, and the `image_url_private` switch is gone.

### 3. The container ran as root (Medium, fixed)

semgrep: `dockerfile.security.missing-user-entrypoint` at `Dockerfile:117` (and `sycl/tools/Dockerfile:9`), and
trivy's `DS-0002`. The server, the engine and the image encoder all ran as root inside the container, with
`--gpus all`, `-p 8080:8080` and a mounted `/data` volume - more privilege than the process needs.

Fixed by making the image two-stage: the first stage keeps the CUDA toolkit and compiles the engine, the second
is the CUDA **runtime** image with a non-root user `strata`, which owns `/opt/strata` and `/data`. See "The
container image", below, for what that removed. `sycl/tools/Dockerfile` is the Intel Arc port's own file and was
left alone; its `USER` is a separate change.

### 4. `before_load` runs a configured command with `shell=True` (Low, by design)

bandit `B602` at `serve/server.py:3009`: the run config's `before_load` command (e.g. to unload another program
from the GPU) is run through the shell. It is the user's own config file, and it is deliberately not editable
from the web page, but anything that can write that file runs code as this user. Worth a note in `SECURITY.md`;
not changed.

### 5. Downloads that are not hash-checked (Low, supply chain) - narrowed

Setup no longer downloads anything: the model files, llama.cpp and the ready-made engine are not fetched, so the unhashed llama.cpp source zip is gone. The build uses the tree in `third_party/llama.cpp` at the pinned commit (`LLAMA_CPP_COMMIT` in `setup.py`). What is left is the Docker build, which installs apt and pip packages (PyPI) while the image is built: the pip packages are pinned by version in `requirements.txt` but installed without `--require-hashes`. That is ordinary practice, and it is supply-chain exposure. `STRATA_WHEELS` (below) lets pip install from a local, reviewable wheel folder instead.

### 6. bandit's remaining noise

`B310` (120 hits, `urllib.request.urlopen` with a variable URL) was the download code; setup has none now (see the phone-home inventory). `B404`/`B603`/`B607` (subprocess) are the build and
GPU probes. `B101` (`assert`) and `B110` (bare `except: pass`) are in the test suite and in cleanup paths.

### 7. The C++ side (cppcheck)

Over `src/`, `include/` and `tools/vision`: 2 errors, 2 warnings, and 1,281 `style` plus 38 `portability` notes.
None of the errors or warnings is a memory-safety bug in shipped code:

- `src/kernels/cpu/q8k_quant_parity.cpp:116` - a parity test file sets a file-scope format pointer to a
  function-local object (`danglingLifetime`). Test-only.
- `src/program/generate.cpp:8980` - `STRATA_VERSION` is an unknown macro to cppcheck; it is a `-D` from CMake.
  False positive.
- `src/core/verify.cpp:2467` - `std::log(sum)` where `sum` can be 0, which yields `-inf` rather than stopping.
  A robustness edge in a verification helper, not a safety bug.
- `src/core/conversation_cache_test.cpp:295` - "inconclusive" use of a moved variable. Test-only.

cppcheck is a partial, static view of CUDA code and does not check the kernels' correctness or memory safety on
the device. With `-DSTRATA_BUILD_TESTS=ON` the repo's own parity and unit tests (`ctest`) are the stronger check;
they run on the GPU and were not run here (they need a full build).

## The container image (trivy)

`trivy image --scanners vuln,misconfig --severity HIGH,CRITICAL strata:v100`
(`sec-reports/trivy-image.json` single-stage, `sec-reports/trivy-image-2stage.json` after).

| | single-stage, root | two-stage, non-root |
| --- | ---: | ---: |
| HIGH/CRITICAL vulnerabilities | 621 | **18** |
| misconfigurations | 56 | 54 (none of them ours) |
| image size | 17.5 GB | 7.92 GB |
| runs as | root | `strata` (uid 1001, group `video`) |

The 603 that went away were the CUDA **devel** toolchain's: 592 packages from the Ubuntu base (dev headers,
build tools, `git`, `python3-pip` and its tree) and the 25 in `nvidia/cuda`'s bundled Nsight Compute (a Go
binary built with Go 1.23.4). The runtime base carries neither, and `/opt/nvidia` there is 40 KB of entrypoint
scripts.

The 18 that remain:

- **14 Ubuntu 24.04 packages** - `gnupg` (`CVE-2025-68973`) and `openssl`/`libssl3t64`
  (`CVE-2026-45447`, `CVE-2026-84782`). All three have a fix published, so an `apt-get upgrade` in the runtime
  stage clears them; they are in the base image the CUDA images are built from.
- **4 Python packages** - `msgpack` 1.1.2, `urllib3` 2.7.0 (twice), `setuptools` 70.3.0. These are **pip's own
  vendored copies** in `.venv/lib/python3.12/site-packages/pip/_vendor/` (`vendor.txt` names exactly those
  versions), not the project's dependencies: `pip list` in the image reports `urllib3==2.8.0`, and `pip-audit`
  against `requirements.txt` found nothing. pip is an install-time tool; it does not run while serving. Removing
  it from the image would also remove the ability setup has to install the CUDA wheels for a model that needs
  them, so it stays - noted, not fixed.

Neither scan found a vulnerability in Strata's own code, and trivy now reports **no misconfiguration in this
repository's `Dockerfile`**.

## Deployment: what was verified

The container image was built for these cards and checked on this machine (2x Tesla PG500-216, driver 580,
Docker 29.7.2, nvidia-container-toolkit with CDI):

- `docker build --build-arg CUDA_VERSION=12.9.1 --build-arg STRATA_CUDA_TOOLKIT=12 --build-arg
  CUDA_ARCHITECTURES=70` succeeds; the engine compiles for sm_70 (`nvcc` warns that offline compilation below
  7.5 is deprecated - it still builds and links).
- The image carries `engine-cuda12/{strata,strata-vision}` with `BUILD.json` `archs: [70]`, `toolkit: 12`,
  `source: local` - which is what setup uses to skip recompiling at the first start.
- `docker run --gpus all` sees both cards (`nvidia-smi -L`), `engine-cuda12/strata --version` prints
  `strata 0.1.43`, and an `-arch=sm_70` CUDA kernel compiled in the image runs on **both** cards, returning the
  expected result - as the non-root user `strata` (uid 1001), in the two-stage runtime image.
- The venv's interpreter works in the runtime image (only the `python3` package is installed there) and `ldd`
  on both engine binaries reports no missing library.
- **Not verified:** a full request path. That needs the model (68.0 GB for IQ2_XS, 83.6 GB for IQ3_S) and the
  disk for it is not free yet.

## Phone-home inventory

Every outbound endpoint in the repository, and when it is used. There is **no analytics, telemetry, beacon or
version-ping** anywhere: `serve/telemetry.py` reads local hardware through NVML and `psutil`.

Install and update (`setup.py`, `setup.sh`, `docker-entrypoint.sh`, `update.sh`) reach no endpoint: setup downloads nothing, and the model files, llama.cpp and the engine are all on disk before it runs. The Docker build (`Dockerfile`) reaches apt and PyPI while the image is built, and nothing else.

Runtime (the server):

| Where | Endpoint | What |
| --- | --- | --- |
| `serve/server.py` | nothing: image URLs are not fetched and local paths are not read (finding 2) | removed |
| `serve/mcp.py` | none: the HTTP transport was removed; stdio servers are programs the run config starts | removed |
| `tools/*.py` | `127.0.0.1` | benchmarks hitting this server |

A plain start on an installed model does **not** reach the network: with a local engine (`BUILD.json`
`source: local`) `update_installed_engine` returns before any request, and the packages are already installed.

### Local-only mode

Setup has no network code, so there is no offline switch to turn on: every model file, the llama.cpp tree and the Python wheels (`STRATA_WHEELS=<dir>`, installed with `--no-index --find-links`) must be on this PC. `--offline` and `STRATA_OFFLINE=1` are accepted and do nothing. The model files come from `--gguf-dir`, and llama.cpp from `third_party/llama.cpp` (or `STRATA_LLAMA_DIR`).

## Not covered

- No dynamic analysis of a running server, no fuzzing of the HTTP surface, no review of the CUDA kernels for
  correctness or memory safety (cppcheck is static and partial).
- The model files themselves (GGUF) are trusted by hash but their contents were not inspected.
- `third_party/` (llama.cpp and ggml) is vendored source and was not audited.
- trivy's image scan of the built container is reported separately once the image builds.

## Keeping this current

Upstream changes almost every day, and the audit above is one commit. `tools/strata_audit.py` is the loop that
keeps it true without doing the whole thing again by hand:

```sh
git fetch origin                        # yourself: the audit makes no network calls
python tools/strata_audit.py            # report the delta, run the offline scanners (1-2 minutes)
python tools/strata_audit.py --cppcheck # also the C++ static pass (about 10 minutes)
python tools/strata_audit.py --image    # also rebuild and scan the container image (about 15 minutes)
python tools/strata_audit.py --accept   # make upstream's tip the new baseline, after reading the report
```

What one run does:

1. `git fetch origin` - read-only, it does not touch the working tree.
2. Diffs the last audited commit against `origin/main` and writes the file list, the commits, the **non-test
   Python added** (the port candidates, against `docs/CXX_PORT.md`) and the **added lines that look like an
   outbound request** - so a new address or download is visible as one line, not as a needle in a 300-line
   report.
3. Makes a throwaway `git worktree` of the target and runs bandit, gitleaks, semgrep and pip-audit **over
   upstream's tree**, so the checkout being worked in stays where it is. The scanners' counts are written next to
   the previous run's, so a moved number is the finding.
4. Writes `sec-reports/audit-<sha>/REPORT.md` and `delta.json`, and records the counts in `sec-reports/state.json`.

A scheduled run uses `--auto-accept`: if nothing was added outbound and no count moved, it moves the baseline on
by itself; if something needs reading, it leaves the baseline where it is and says so. That way an unattended job
neither buries a real finding nor keeps re-reporting a quiet one.

To run it weekly on this machine:

```sh
# crontab -e
0 6 * * 1  cd $HOME/Projects/Strata && python3 tools/strata_audit.py --auto-accept >> sec-reports/cron.log 2>&1
```

`sec-reports/state.json` holds the audited commit and the last run's counts; delete it to start the baseline over.
A scheduled run needs the scanners on `PATH` (bandit, gitleaks, semgrep, pip-audit; cppcheck only for
`--cppcheck`); one that is missing is reported and skipped rather than guessed at.
