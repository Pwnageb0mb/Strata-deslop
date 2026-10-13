# TODO

Open work on this fork. Only what is actually outstanding — what the checkout *is* and what it can do is in
[README.md](README.md); what is ported and why is in [docs/port-ledger.tsv](docs/port-ledger.tsv).

## 1. The recurring upstream audit is not scheduled

`tools/strata_audit.py` (local ref → delta → scans → port verdict; `git fetch` is run by hand) is meant to run on a
schedule. Nothing schedules it yet: a cron entry or a systemd timer that runs `git fetch origin` and then the audit
would do.

## 2. The container has no restart policy

`RESTART` is commented out in `deploy/strata-container.conf`, so a host reboot leaves Strata down until
`deploy/strata-start.sh` runs again. Uncommenting it is a one-line change; it was left off deliberately.

## 3. The remaining engine knobs

Set, and measured, in `ENGINE_ARGS` in `deploy/strata-container.conf`:

- **`--batch 2`** — 2 slots, up from 1. Two clients: 52.8 tok/s aggregate against 38.1 for one (+39%), per-request
  32.1 against 46.5.
- **`--max-context 262144`** — 256k per slot, up from 32,768, at the model's own ceiling.

Both cost expert cache, which is the trade to keep in mind if either moves again: 1 slot of 32k held 8,740 + 8,153
cache slots (25.53 + 23.77 GiB, ~99.0% of the routed mass), and 2 slots of 256k hold 7,085 + 6,469 (20.69 +
18.91 GiB, ~98.9%), taking 3.73 GiB of VRAM on CUDA0 for the slot sessions. The decode hit rate over a 240k-token
request was 86.6%.

Still open:

- **`--fit-max-tokens`** (a server flag, not an engine one) — off, so a request whose prompt + `max_tokens` exceeds
  the context is rejected with 400 rather than clamped. A client that sizes its own request cannot reach the last
  ~2% of the 262,144-token slot: a 231,736-token prompt works, but budgeting 256k of prompt *plus* output is
  refused.
- **`--batch-mtp`** — the per-slot MTP drafters. With the layer split and `--batch 2`, the engine pipelines the
  slots in groups of one by default (it logs `--batch-groups auto: 2 groups of 1 slot`) and the pipelined path does
  not run per-slot MTP drafts. `--batch-mtp` needs `--batch-groups 1`; giving `--batch-groups` keeps the pipeline
  and turns `--batch-mtp` off. Unmeasured here — docs/BATCHING.md measured +15.4% aggregate for it on a 2× RTX 3080
  layer split, at 35-110 expert-cache slots and 50 MiB of private state per slot.

## 4. References to what the pruning removed

Upstream-tracked files were removed; other upstream files still name them. **None of this breaks the build or the
server** — verified by reconfiguring CMake and by running the `tools/test_setup_*` suite — they are comments,
optional lookups with documented fallbacks, and branches that only exist when a platform is enabled. They were left
alone on purpose: rewriting upstream source is what makes the next merge harder. Fix or accept, per row.

| removed | still named by | what it does now |
| --- | --- | --- |
| `tests/hip/`, `tools/hip/` | `CMakeLists.txt`, `cmake/hip_backend.cmake`, `setup.py`, `src/prefill/gemm.cu`, `src/core/native_head.cpp`, and 3 more under `src/` / `tests/` | dead branches under `STRATA_ENABLE_HIP`; `setup.py`'s hipBLASLt-table lookup warns and falls back; its `from dll_closure import dll_closure` is `try/except`-guarded and falls back to the name list |
| `include/strata/hip_compat/`, `include/strata/platform/hip_compat/` | `CMakeLists.txt`, `cmake/hip_backend.cmake`, `include/strata/kernels/dp4a.hpp`, `src/core/device.cu` | the include path is added only under `STRATA_HIP_GFX906`; the rest are comments |
| `docs/AMD_HIP*.md`, `docs/INTEL*.md`, `docs/A770_SPEED_SPEC.md`, `docs/STRIX_HALO.md` | `AGENTS.md`, `setup.py`, `setup.sh`, `docs/AI_SETUP.md`, `docs/COMMUNITY_BENCHMARKS.md`, `cmake/hip_backend.cmake`, `CMakeLists.txt`, `include/strata/core/arch_defaults.hpp`, `include/strata/kernels/gfx_arch.hpp`, `src/core/device.cu`, `src/prefill/gemm.cu` | dead documentation links |
| `bench/gfx906_*`, `data/expert-profile-sycl.bin`, `tools/test_dll_closure.py` | nothing | clean |
| `sycl/` | `CMakeLists.txt` | `STRATA_ENABLE_SYCL` is OFF by default, so the build still configures; `-DSTRATA_ENABLE_SYCL=ON` now fails at `add_subdirectory(sycl)` |
| the experimental-speed-projection vector | `setup.py` | `sycl_setup()` and `--experimental-speed-projection` both fail with "is missing"; ~10 messages name `START-HERE.bat` / `UPDATE.bat` / `tools/hip/build_windows.bat` |
| `bench/results/`, `docs/paper/`, `docs/media/`, `sec-reports/` raw scanner output | `docs/COMMUNITY_BENCHMARKS.md`, `docs/DETAILS.md`, `docs/MODELS.md`, `docs/INSTALL.md`, `docs/SECURITY_AUDIT.md`, `serve/server.py`, ~9 files under `src/` and `include/` | dead links; comments cite `bench/results/...` as measurement provenance |

`AGENTS.md` is the one file left that still describes Windows and AMD: it is upstream's agent guidance and says the
project runs "on Windows or Linux" with "one NVIDIA or AMD graphics card". It was kept because it is upstream's and
because it is the file an agent session reads. Trim it or delete it if the fork should read as CUDA-only.

## 5. The port tests need a rebuild

`build-pretok/` was removed as build output. The next `ctest` needs:

```
.venv/bin/cmake -S . -B build-pretok -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=ON
.venv/bin/cmake --build build-pretok --target <test> -j 28
.venv/bin/ctest --test-dir build-pretok
```

A full `ctest` there reports 39 failures that are **not** failures: the engine's own tests need a library this
configuration never builds, and they abort before testing anything. docs/CXX_PORT.md explains it.

## Not outstanding

The Python → C++ port is done at its boundary: 14 targets, each held by a test against the reference's own bytes.
The remaining `needs-surface` rows in the ledger are files that need a subsystem this tree does not have, or that drive the server over HTTP — a boundary, not a backlog.
