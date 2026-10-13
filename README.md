> **Yes, I see the irony.** One of the main criticisms of the original Strata is that it's AI slop, and every fix in
> this fork was also written by AI. There's a healthy dose of hypocrisy in that. If you don't like it, you can suck it.

# strata-deslop

A fork of [Niko1221/Strata](https://github.com/Niko1221/Strata) — a local runner for the Qwen3.8-Flash-Next
mixture-of-experts model — with the slop scraped off. It runs the model on one machine, inside a Docker container
that cannot reach the internet, behind an OpenAI- and Anthropic-compatible API.

> **Who wrote this.** 100% of the changes and code additions in this fork were made by two AI models:
> **DeepSeek 4.1 Flash** (the first pass: pruning, the first Python → C++ ports, the deployment scripts) and
> **Claude Opus 5.5** (the code review, removing the outbound calls DeepSeek claimed to have removed and hadn't,
> the C++ server pipeline, the startup and CPU-placement optimizations, the hardening, and this README). A human
> chose what to do, ran the prompts, and owns the result. No line here was typed by a person. Read it with that in
> mind.

> **This is tuned for one machine.** Every default below — thread counts, core placement, cache sizes, slot count,
> context length, the CUDA toolkit and GPU architecture — was picked and measured on the host described in
> [The host](#the-host). On different hardware it will still start, but the numbers in this README will not be yours.
> Hand this repo and your own hardware details to Claude or another capable LLM and have it re-tune
> `deploy/strata-container.conf` and the Docker build arguments for you.

---

## Contents

- [The host](#the-host)
- [Running it](#running-it)
- [How the Docker deployment is isolated from the internet](#how-the-docker-deployment-is-isolated-from-the-internet)
- [What was wrong with upstream Strata](#what-was-wrong-with-upstream-strata)
- [Changelog](#changelog)
- [Optimizations and results](#optimizations-and-results)
- [Layout](#layout)
- [Licence](#licence)

---

## The host

Every measurement in this README was taken here.

| Part | Spec |
| --- | --- |
| CPU | 2× Intel Xeon E5-2680 v4 @ 2.40 GHz (3.3 GHz turbo), 14 cores each = **28 cores, SMT off**, 35 MiB L3 per socket (70 MiB total), 2 NUMA nodes (even CPUs on node 0, odd on node 1), `schedutil` governor |
| RAM | **251 GiB** usable |
| GPUs | **2× NVIDIA Tesla PG500-216** — V100-class, 32 GiB HBM2 each, compute capability **7.0 (sm_70)**, PCIe Gen3 ×16. GPU0 hangs off socket 0, GPU1 off socket 1; GPU↔GPU traffic crosses the socket interconnect (`SYS`) |
| Measured links | host→device 9.6 GB/s (GPU0) and 10.5 GB/s (GPU1) pageable probe; 9.2 GB/s from pinned memory |
| Storage | 1 TB NVMe (SK hynix Platinum P41, `SHGP31-1000GM`), ext4 on LVM; 2.8 GB/s direct sequential read |
| OS | Ubuntu 24.04.4 LTS, Linux 6.8.0-146 |
| Driver / CUDA | NVIDIA 580.178.04; the engine is built with the **CUDA 12.9** toolkit, because CUDA 13 dropped Volta |
| Docker | 29.7.2 with the NVIDIA container toolkit |

What that means for you: the build is `sm_70` only, the thread layout assumes 28 non-SMT cores on two sockets, and the
expert cache and context sizes assume 64 GiB of VRAM and ~250 GiB of RAM. A single consumer GPU, an SMT CPU, a
different core count or less RAM all need different settings. See the warning at the top.

---

## Running it

```
deploy/strata-start.sh      # applies deploy/strata-container.conf, recreates the container, waits for healthy
deploy/strata-stop.sh       # stops it; the /data volume is untouched
```

`deploy/strata-container.conf` holds every knob the container takes in one file: the `docker run` parameters, the
entrypoint's environment, and the engine flags. `strata-start.sh` always recreates the container, so that file is
what runs.

What it serves today: Qwen3.8-Flash-Next, Unsloth `UD-Q4_K_XL`, **2 slots of 262,144 tokens** with int8 KV, MTP
speculative decoding, an adaptive expert cache across both GPUs (layers 0–22 on GPU0, 23–47 on GPU1), and
`--pool-workers 22 --aux-cpus auto` (see [Optimizations](#optimizations-and-results)).

- API: `http://127.0.0.1:8081/v1/chat/completions` (and `/v1/messages`, `/v1/responses`), plus any LAN address you
  add to `PUBLISH` in `deploy/strata-container.local.conf`.
- API key: the contents of `~/.strata-api-key` on the host (a bearer token; make it random). Health check:
  `curl 127.0.0.1:8081/health`, the only route without a key.
- Benchmark: `tools/strata_bench.py --url http://127.0.0.1:8081 --api-key "$(cat ~/.strata-api-key)"`.

---

## How the Docker deployment is isolated from the internet

The goal: the container that runs the model has **no route to the internet**, and nothing in the code tries to
reach it anyway. Two independent layers.

### Layer 1: the code makes no outbound calls

Upstream downloads things all over the place (see [below](#what-was-wrong-with-upstream-strata)). In this fork:

- `setup.py` has no network code. Models, the vision encoder and the MTP draft layer must already be on disk.
- CMake never fetches llama.cpp; it must be in `third_party/llama.cpp`.
- The server accepts images only as `data:` URLs. It fetches no URLs and reads no local files as images.
- MCP tool servers can only be local processes (stdio). URL-based MCP servers are refused at startup.
- `update.sh`, `tools/mtp_fetch.py`, `tools/quantscope.py`, `tools/strata_inspect.py` and `tools/strata_audit.py`
  only touch local files and local git refs.

### Layer 2: the network can't get out even if something tries

1. **A dedicated Docker network.** The container runs on `strata-net` (set as `NETWORK` in
   `deploy/strata-container.conf`). `strata-start.sh` creates it on first use.
2. **An egress allowlist on that network's bridge.** `deploy/strata-netfilter.sh` adds two rules to Docker's
   `DOCKER-USER` chain, scoped to the `strata-net` bridge interface:
   - allow `ESTABLISHED,RELATED` traffic, so replies to clients that connected *in* still flow;
   - **drop every new packet from the container to anything outside `192.168.0.0/16`**.

   The rules are tagged `strata-netfilter`, and stale rules for bridges that no longer exist are removed on each
   run. Install it as a boot-time unit (`deploy/strata-netfilter.service`, which runs
   `/usr/local/lib/strata/strata-netfilter.sh`) so it survives reboots. **Re-run it after the network is recreated**:
   the rules follow the bridge name, and `strata-start.sh` warns when it had to create the network.
   - Want *zero* egress, not even the LAN? Run it with `STRATA_ALLOW=` set to a range nothing lives in, e.g.
     `STRATA_ALLOW=192.0.2.0/32` (a documentation-only address).
   - Check the live rules: `sudo deploy/strata-netfilter.sh --check`.
3. **Inbound is published explicitly.** `PUBLISH="127.0.0.1:8081:8080"` exposes the API on loopback only. Add
   `<your-LAN-IP>:8081:8080` in the gitignored `deploy/strata-container.local.conf` to reach it from other machines. Note: Docker publishes ports with its own DNAT and `FORWARD` rules, which
   run before ufw, so **ufw does not gate the published port**. The real boundary is who can route to
   the published address, plus the API key. There is no TLS. Keep it on the LAN.

### What still needs the internet, once, and how to do it offline

| Need | When | Offline answer |
| --- | --- | --- |
| Base images (`nvidia/cuda`), apt packages, Python wheels from PyPI | `docker build` only | Build on a connected machine and `docker save` / `docker load` the image, or point Docker at a local mirror |
| llama.cpp source at commit `3cf03257f219afbe7334045ff7c6a06ac68c627d` | `docker build` | Put it in `third_party/llama.cpp` yourself (a tarball from anywhere). The build never downloads it |
| Model GGUFs, MTP folder | first start | Copy them into the `strata-data` volume, or mount a folder as `GGUF_DIR` |

At runtime the container reaches nothing outside the allowlist, and nothing in it tries to.

### First-time setup, end to end

```
# 1. llama.cpp source at the pinned commit, placed by hand
#    third_party/llama.cpp/ggml/CMakeLists.txt must exist

# 2. build the image (CUDA 12 + sm_70 for Volta; change for your GPUs)
docker build -t strata:v102 --build-arg CUDA_VERSION=12.9.1 \
  --build-arg STRATA_CUDA_TOOLKIT=12 --build-arg CUDA_ARCHITECTURES=70 .

# 3. the API key file
head -c 24 /dev/urandom | base64 > ~/.strata-api-key && chmod 600 ~/.strata-api-key

# 3b. optional: machine-specific settings (gitignored), e.g. a LAN binding
echo 'PUBLISH="127.0.0.1:8081:8080 <your-LAN-IP>:8081:8080"' > deploy/strata-container.local.conf

# 4. model files into the volume (or set GGUF_DIR in the conf)

# 5. egress allowlist, now and at boot
sudo install -D deploy/strata-netfilter.sh /usr/local/lib/strata/strata-netfilter.sh
sudo cp deploy/strata-netfilter.service /etc/systemd/system/ && sudo systemctl enable --now strata-netfilter

# 6. start
deploy/strata-start.sh
```

- **How the API key reaches the server:** `strata-start.sh` pipes the key into `deploy/strata-config.py` over
  stdin, inside a throwaway container, and writes it into the install config on the volume. It also writes the
  engine flags from `ENGINE_ARGS` there.
- **The one exception:** on the very first start the config doesn't exist yet, so the key is passed as an
  environment variable for that one start; the script says so when it happens.
- **Inside the image:** the code is root-owned and read-only to the `strata` service user, and the container runs as
  that user.

---

## What was wrong with upstream Strata

Upstream bills itself as a one-click way to run a huge model on a normal PC. What it actually is: a
6,500-line Python web server, a 5,900-line installer, and a C++ engine whose `main()` is one function longer than
most novels, all glued together by a project that moves faster than it reads its own diffs. In detail:

**It phones home, constantly, and trusts whatever comes back.**
- The installer downloaded models from Hugging Face or ModelScope, llama.cpp as a zip from GitHub, ready-made engine
  binaries from GitHub "latest" releases, NVIDIA's CUDA apt keyring, and ROCm wheels from nightly indexes. It also
  installed packages with pip at setup time.
- Hugging Face files and the llama.cpp zip were never hashed.
- If the pinned model revision disappeared, it silently downloaded whatever the repository held that day.
- When a prebuilt engine binary had no published checksum, the code's own comment says "no digest is never a reason
  to refuse" — and then it ran the binary.
- CMake would `git clone` llama.cpp at configure time if you didn't stop it.
- The server fetched any image URL a client sent. The SSRF guard was bolted on after the fact.
- The same image path read **any local file** an API client named, as long as the client wasn't a browser on another
  origin. curl qualified.

**Its "measured" claims cite measurements it deleted.**
The docs insist on "no claims without a measurement". Then comments throughout `src/` cite `bench/results/...` files
as their provenance, the archive got pruned, and the citations now point at nothing.

**A leaked credential, leaked twice.**
A crawl4ai bearer token sat in fourteen committed bench reports in the public repo. Then the project's own
secret-scanning reports were committed, quoting the token they had found. (DeepSeek's first pass redacted it; it is
still in upstream's history.)

**The C++ engine is a monolith with hidden knobs.**
- `src/program/generate.cpp` is 13,437 lines, and **11,700 of them are `main()`**.
- It reads **327 distinct `STRATA_*` environment variables** through 439 `getenv` calls, on top of ~170 command-line
  flags. Undocumented behaviour switches, everywhere.

**It wastes the hardware it claims to make the most of.**
- **Startup pegged one core for two minutes** (118 s to ready on this host):
  - The 39 GiB of page-locked RAM came from one `cudaHostAlloc` call. The kernel zeroes and locks every page on a
    single thread, at 1.75 GB/s.
  - The GPU expert cache was filled one 3 MB expert at a time, with blocking copies from memory-mapped files, at
    1.26 GB/s. The link does 9.2 GB/s and the disk 2.8 GB/s.
  - Worst of all: the engine pins its host thread to core 0, and Linux gives new threads their creator's CPU mask.
    So the "parallel" startup helpers — up to 32 of them — **all ran on core 0, taking turns**.
- **The same bug at runtime.**
  - The host thread, the per-request threads and ~20 CUDA/helper threads all shared core 0.
  - The engine has a setting to move them (`--aux-cpus`), but it's off by default.
  - Even if you turned it on, the CPU worker pool had already claimed every other core, so it had nowhere to put
    them.
  - Result: core 0 at 100%, 27 cores idle, and decode ~29% slower than it needed to be.
- **The prompt tokenizer for a 262,144-token context was pure-Python BPE.** 2 s for a 10 MB prompt.

**Its installer is a museum.**
- `setup.py` is 5,889 lines in one file: Windows registry and page-file code, AMD ROCm/HIP install logic, Strix Halo
  and gfx1103 special cases, Intel SYCL hooks, and download/retry/mirror logic for all of it.
- Every platform's quirks live in the same file as the network code, and none of it is optional. A Linux + NVIDIA
  box still loads, parses and maintains it all.
- When the first pass pruned those platforms, the docs and comments kept linking to files that no longer exist.

**Its Docker image trusted itself too much.**
- The service user owned the entire install, so it could rewrite its own engine and code.
- `cmake` and `ninja` shipped in the runtime image.
- The API key went in as a `-e` environment variable, readable by anyone with `docker inspect`.

**Smaller things a review shouldn't have to find:**
- The safetensors reader allocated whatever length the file's first 8 bytes asked for.
- The JSON reader had no nesting limit.
- The GGUF reader's bounds check could overflow.
- The expert-profile reader sized its buffer from an unchecked header field.
- The netfilter script piled up stale rules every time the network was recreated.

**And the first-pass fork wasn't clean either.**
DeepSeek 4.1 Flash was asked to remove the outbound calls and port the Python to C++, and claimed both.
- The installer still downloaded everything.
- The fourteen "ported" tools were only built under the test flag. Nothing called them, and the Python originals
  kept running.
- Its C++ tokenizer port had a merge bug on long runs, such as long whitespace, that its own golden tests never
  exercised.

The second pass (Claude Opus 5.5) found and fixed these. It's in the changelog.

---

## Changelog

All changes since upstream commit `7695859a` (2026-10-11). Authors: DeepSeek 4.1 Flash (first pass) and Claude Opus
5.5 (everything after the code review).

### Outbound calls removed
- `setup.py`:
  - No Hugging Face, ModelScope, GitHub-release, llama.cpp-zip, CUDA-keyring, ROCm or pip downloads.
  - `download()` only copies local files.
  - `get_llama_cpp()` only uses `third_party/llama.cpp` or `$STRATA_LLAMA_DIR`.
  - The engine is always compiled locally.
- CMake: the `FetchContent` clone of llama.cpp is gone; configuring without it fails with a clear message.
- `.dockerignore`: lets a locally placed `third_party/llama.cpp` into the build.
- `update.sh`: no `git pull`, no zip download.
- `tools/mtp_fetch.py`: verify-only.
- `tools/quantscope.py` and `tools/strata_inspect.py`: local files only.
- `tools/strata_audit.py`: compares a ref you fetched yourself; semgrep needs local rules, pip-audit no longer runs,
  and trivy runs offline.
- Server: image URL fetching removed, along with reading local files as images; `data:` URLs only.
- MCP: the HTTP transport was removed; stdio servers only.

### Server processing moved to C++ (`libstrata_text.so`)
- **Tokenizer:**
  - The existing C++ port is now actually used by the server.
  - Fixed its long-run BPE merge bug.
  - Added a per-thread piece cache and a fast special-token scan.
- **Chat templates:** rendered by llama.cpp's own C++ Jinja engine. It is vendored in `third_party/llama_jinja` with
  nlohmann/json, both MIT.
  - As first vendored, the engine was 5× slower than Python's Jinja2; it was profiled and rewritten in place, with
    every change listed in its README.
  - A direct JSON reader replaced nlohmann on the hot path.
- **Per-token output pipeline:** the detokenizer, stop strings, the reasoning/tool-call stream parser and the
  repeated-reasoning check, fused into one C++ call per token.
- **Request translation (OpenAI/Anthropic → messages):** ported, but measured 3–4× slower end to end because the JSON
  round trip costs more than the Python. It was off by default, and a follow-up commit removed the port
  (`STRATA_NATIVE_REQUEST`); the Python translation is the live path.
- **Kept in Python:**
  - The original Python tokenizer, detokenizer, stop matcher, output parser and repeat check were first kept as the
    reference and fallback (`STRATA_PY_TOKENIZER=1`); a follow-up commit removed them. The server now requires
    `libstrata_text`. The Jinja2 chat-template fallback stays, for templates the C++ engine cannot parse.
  - HTTP handling and process control, which is glue.
  - JSON-schema checks, which use the `jsonschema` package.

### Engine startup
- Helper threads no longer inherit the host thread's single core during startup phases (`WidenGuard`;
  `STRATA_WIDEN_STARTUP=0` reverts).
- Pinned RAM: parallel first-touch of an anonymous mapping plus `cudaHostRegister`, replacing the single-threaded
  `cudaHostAlloc` (`STRATA_FAST_PIN=0` reverts).
- GPU cache fill: up to 16 reader threads feeding pinned staging buffers and async copies, for both GPUs
  (`STRATA_FILL_THREADS=0` reverts).
- RAM expert copy: large `pread` reads instead of page faults (`STRATA_COMPLEMENT_PREAD=0` reverts).
- A shared read counter made atomic for the new parallel paths.

### Deployment
- `deploy/strata-container.conf`: image `strata:v102`; `ENGINE_ARGS` adds `--pool-workers 22 --aux-cpus auto`.
- `deploy/strata-start.sh`: the API key is no longer passed as an environment variable (except on the first start).
- `deploy/strata-netfilter.sh`: tagged rules, stale-bridge cleanup.

### Hardening
- Dockerfile:
  - `/opt/strata` is root-owned with the sticky bit, so the service user can create its files but cannot replace the
    code.
  - `cmake`/`ninja`, `tests/`, `bench/`, `sec-reports/` and `ref/` are removed from the runtime image.
  - Python is byte-compiled at build.
- Readers: safetensors headers capped at 100 MiB, JSON nesting capped at 256, the GGUF bounds check made
  overflow-safe, the tensor count and alignment validated, and expert-profile sizes validated before allocating.

### Removed
- **AMD / HIP / Windows / Intel code in `setup.py`:** 5,889 → 3,584 lines in total, together with the network
  removal.
- **Their tests:** the setup tests for every removed feature.
- **The rest of the platform pruning:** done by DeepSeek in the first pass.

---

## Optimizations and results

All measured on [the host](#the-host), same model and settings unless stated.

### Startup (engine start → ready)

| Phase | Before (v100) | After (v102) | Change |
| --- | --- | --- | --- |
| **Total** | **118 s** | **27.6 s** | **4.3× faster** |
| GPU cache fill, GPU0 (21.6 GiB) | 18.4 s, 1.26 GB/s | 2.9 s, 7.9 GB/s | 6.3× |
| GPU cache fill, GPU1 (18 GiB) | not logged | 2.1 s, 9.3 GB/s | — |
| Pinned RAM allocation (39 GiB) | ~29 s | 7.2 s | 4× |
| Expert copy into RAM (39 GiB) | 32 s | 3 s | 10.7× |
| CPU use during startup | 1 core at 100% | all cores | — |

The biggest single factor was the core-0 inheritance bug. The same three changes without it were 74 s.

### Runtime: core placement

`tools/strata_bench.py`, 3 runs per config, medians. Same image (v102).

| Metric | Upstream default (A) | `--pool-workers 22 --aux-cpus auto` (C, deployed) | Change |
| --- | --- | --- | --- |
| Chat decode | 39.2 tok/s | **50.7 tok/s** | +29% |
| Agent-session decode | 53.1 tok/s | **60.7 tok/s** | +14% |
| Chat time to first token | 0.61 s | 0.51 s | −16% |
| 4K prompt reading | 906 tok/s | 945 tok/s | +4% |
| 16K prompt reading | 1,636 tok/s | 1,696 tok/s | +4% |

- **Why it works:** freeing five cores from the expert pool gives the engine somewhere to put the threads that were
  crowding core 0. The pool only computes the 2–6% of experts not resident on the GPUs, so it loses nothing
  measurable.
- **Noise:** decode is noisy run to run (chat ranged 38–57 tok/s within one config); prompt reading varied 1–2%.
- **The other configs:** also tested were 24 pool workers, which came close, and per-slot MTP drafting
  (`--batch-groups 1 --batch-mtp`), which gained nothing.

### Server text processing (Python → C++)

| Work | Python | C++ | Change |
| --- | --- | --- | --- |
| Tokenize a 10 MB prompt (2.77M tokens) | 2.01 s | 0.68 s | 3.0× |
| Chat template, 201-message conversation | 5.1 ms | 3.1 ms | 1.6× |
| Per-token output pipeline | 4.0 µs | 1.7 µs | 2.4× |
| Repeated-reasoning check | 47 ms | 9 ms | 5.2× |
| Request translation (200 messages) | ~0.5 ms | ~1.7 ms | **slower, off by default** |

Every C++ path was checked against the Python reference (since removed, see above):
- **Tokenizer:** identical ids on the deployed 248,320-token vocabulary.
- **Chat templates:** 3,960 renders identical to Jinja2.
- **Output pipeline:** 20,000 randomly chunked streams identical.
- **Request translation:** 12,000+ generated requests and 277 test-suite requests identical.

### Image and code size

| | Before | After |
| --- | --- | --- |
| Docker image | 7.92 GB | 6.97 GB |
| `setup.py` | 5,889 lines | 3,584 lines |
| Outbound network calls in code | many | 0 |

### Known gaps

- **Prompt reading leaves half the GPU time idle.** With the layer split, the two GPUs take turns and never run at
  once. Overlapping prompt chunks across the GPUs could roughly double prompt speed. Not done; it's an engine change.
- **Python still matches pins only.** It doesn't verify package hashes, because computing them needs the network.
- **`main()` is still 11,700 lines.** Nobody volunteered.

---

## Layout

| | |
| --- | --- |
| `src/`, `include/` | the engine (CUDA), plus `src/text/` and `include/strata/text/`: the server's C++ text library |
| `serve/` | the HTTP API (Python) and its ctypes wrappers: `native_text.py`, `native_stream.py` |
| `tools/` | packer, tokenizer, benchmarks, audit |
| `deploy/` | the container config, start/stop scripts, config reconciler, egress allowlist |
| `third_party/` | `ggml` headers, vendored `llama_jinja` and `nlohmann` (MIT); `llama.cpp` is placed here by hand, gitignored |
| `docs/` | `port-ledger.tsv`, `CXX_PORT.md`, `SECURITY_AUDIT.md`, and upstream's engine docs |
| `data/` | expert profile and draft vocabularies |

`AGENTS.md` is upstream's agent guidance and still talks about Windows and AMD. Ignore that part. [TODO.md](TODO.md)
lists what's open.

## Licence

Upstream's MIT; see [LICENSE](LICENSE).
- **The model:** [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team, quantised by
  Unsloth.
- **Vendored code:** llama.cpp's Jinja engine and nlohmann/json, both MIT, with their licences kept.
