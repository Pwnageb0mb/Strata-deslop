# Porting the Python to C++: what is already C++, what can move, and in what order

A request to "rewrite the Python as C++ where possible" needs the word *possible* pinned to something. The
repository answers it in two places, in its own words, and this file records the measurement behind them.

> Follow-up: the Python tools that have a C++ port and that nothing still runs (`make_profile`, `make_profile_mix`,
> `bench_eviction`, `replay_cache_policy`, `draft_vocab`, `ple_fp8_pack`, `embd_bf16_pack`), their Python tests and
> their `gen_*_golden.py` generators were removed. The C++ tools, tests and golden files are now the reference.
> Python tools below that are named as the reference no longer exist in the tree; see git history.

## How much of this is Python

Non-test lines, this checkout (`third_party/` and `bench/` left out):

| Part | Language | Lines |
| --- | --- | --- |
| `src/`, `include/`, `tools/vision/` - the engine, the kernels, the image encoder | C++ / CUDA | 111,556 |
| `serve/` - the HTTP server, the API translation, the web app's backend | Python | 12,399 |
| `tools/` - the packer, the tokenizer, the benchmarks | Python | 12,215 |
| `setup.py`, `setup.sh`, `chat.py` - the installer and the CLI | Python | 6,005 |

So the hot path is already C++: embeddings, the 48 layer graphs, the MoE router, attention, the sampler and
the KV cache all live in `src/`. What is Python is the API surface, the installer, and two data tools.

## Where the repository itself says the boundary is

**`docs/port-ledger.tsv` is the list**: every Python file of the runtime surface (`tools/`, `serve/`, `chat.py`,
`setup.py`), what holds it in C++, and the test that does the holding. It is the machine-readable form of this
file, and the thing to read a changed file against -
"has this been ported, and what would have to be re-verified". A row without a test is a claim, not a fact.

### Running the port tests

`build-pretok/` is the working build directory: configured `-DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=ON`
and built target by target, so it compiles the port tests without the engine.

```
.venv/bin/cmake -S . -B build-pretok -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=ON
.venv/bin/cmake --build build-pretok --target <test> -j 28
.venv/bin/ctest --test-dir build-pretok
```

**A full `ctest` there reports 39 failures. They are not failures.** Every one is
`FileNotFoundError: build-pretok/libstrata_kernels_cpu.a` - the engine's own tests need a library this
configuration never builds, so they abort before testing anything. The 18 that pass are the 18 port tests. To
run the engine's suite as well, build the library too (or use a build directory configured for the engine).
Reading those 39 as breakage wastes an afternoon; they are a property of the scratch build, not of the code.

- `src/program/generate.cpp`, header: *"There is no tokenizer here. `pack/full/tokenizer/` and
  `tools/strata_tokenizer.py` exist, and a C++ BPE is Phase 1's deliverable rather than this program's."* The
  engine takes a prompt as token ids because a tokenizer on one side of a parity comparison is a second
  variable.
- `tools/strata_tokenizer.py`, header: *"A C++ port follows and is tested against this one."* It also names
  the test that decides whether a port is correct: `decode(encode(s)) == s`, exact because the byte layer is
  lossless, asserted over multi-byte UTF-8, emoji, combining marks, whitespace runs and control bytes. A port
  that produces plausible ids but fails the round trip is wrong.
- `CMakeLists.txt`: *"`strata_plan` and the C++ `strata-pack` are added when their sources exist - a target
  with no sources would make the build graph claim something the tree does not have, which is worse than an
  absent target."*

Two named C++ deliverables, then, and no third. That is the honest scope of "where possible".

## Order

### 1. The tokenizer (`tools/strata_tokenizer.py`, 368 lines)

On the request path - every prompt is tokenized, and the prompts this model is built for are up to 262,144
tokens - and already specified as a C++ port with its own acceptance test. It reads only the pack's
`tokenizer/` directory (248,320 tokens, 247,587 merges), so a port needs no model weights and its test runs
without a GPU.

**Landed: the pre-tokenizer and the tokenizer.** `src/tokenizer/` - `pretokenizer.cpp`, `tokenizer.cpp`,
`utf8.cpp` - with `include/strata/tokenizer/`.

**Now used by the server at runtime.** The server's tokenizer is C++: `libstrata_text.so` (cmake target `strata_text`, `cmake/strata_text.cmake`; C ABI in `include/strata/text/strata_text.h` and `src/text/strata_text_capi.cpp`), loaded by the ctypes wrapper `serve/native_text.py`. The Dockerfile builds it and copies it beside the engine. It is looked up in `$STRATA_TEXT_LIB`, beside the engine binary, then `ROOT/engine`, `ROOT/engine-cuda12` and `ROOT/build`. If it is missing, the server refuses to start (the `STRATA_PY_TOKENIZER=1` fallback to the Python tokenizer was removed; `tools/strata_tokenizer.py` stays for the GGUF extract and the tools that import it). Measured on this host against the deployed UD-Q4_K_XL vocabulary (248,320 tokens): identical ids on 133 texts and on a 10 MB, 2.77M-token prompt, which took 0.68 s native against 2.01 s in Python (cold piece cache). The golden tests had not covered one bug in the heap path, for pieces over 64 symbols (long whitespace runs): a node already merged away could merge again. It is fixed.

The pattern leans on `\p{L}`, `\p{M}`, `\p{N}` and `\s`, so the classes had to come from somewhere. They come
from the reference's own module, generated into a 9 KB header by `tools/gen_unicode_classes.py` - **not** from
`std::isalpha`, ICU or Python's `unicodedata`, which are different Unicode versions: measured on this checkout,
`regex` 2026.9.10 against Python 3.12's `unicodedata` (Unicode 16.0.0) disagree on **4,644 codepoints in
`\p{L}`**, 42 in `\p{M}` and 13 in `\p{N}`. A scanner built on the wrong table produces plausible pieces and a
different prompt.

The pattern is also not "longest match wins": Python's `regex` takes the first alternative that matches and
gives characters back when the rest of a branch fails. Two cases that a reasonable-looking implementation gets
wrong, both now in the golden file: `"a  b"` is `{"a", " ", " b"}` (the last space joins the word; the first is
matched by the whitespace branch, one character short) and `"12ab"` is `{"1", "2", "ab"}`.

The tokenizer is the rest of the reference: the GPT-2 byte alphabet, the merge loop (**lowest rank first**, not
left to right), `decode` back to text, and the two classes of literal - type 4 matched always, type 3 only with
`parse_special` - plus the `plain` spans (#537) that keep a quoted literal as text.

Proof, two goldens from the reference's own output, both run by `ctest`:

- `tools/gen_pretok_golden.py` -> `pretokenizer_test`: **72 cases, 150 pieces, 0 disagreements**, and the pieces
  rebuild each text exactly.
- `tools/gen_tokenizer_golden.py` -> `tokenizer_test`: three synthetic vocabularies (the reference test's own
  toy, one where rank order and left-to-right order disagree, one with both kinds of literal) and 116 cases of
  ids and round trips: **0 disagreements**.

Two bugs the tests caught, both worth keeping in mind for the next slice: a `Tokenizer` that moves into a
`std::vector` dangled its `string_view` keys (fixed by holding the tokens in a `std::deque`, whose elements do
not move), and a golden whose fields were space-separated lost its alignment on the empty-string case - the
fields are tab-separated now, and the reader refuses a short line instead of reading past it.

**The heap path is ported too.** The reference switches above 64 symbols from rescanning every pair after each
merge (O(n^2), #268: seconds on an 8K-character CJK run) to a linked list and a heap keyed (rank, position). The
port does the same at the same threshold. Proof: the golden grew to 152 cases including long pieces (`x`*300,
`日`*40, `日本語`*40), and with the scan path disabled entirely - every piece forced through the heap - it still
reproduces all 152 ids, 0 disagreements.

**What is left of this slice:** nothing in `tools/strata_tokenizer.py` except `extract`, which reads the
vocabulary out of a GGUF and writes the pack's `tokenizer/` directory - that is tooling over the artifact layer
(`src/artifact/gguf_reader.cpp`), not the runtime tokenizer, and it belongs with the packer below.

### 2. The packers - two of them, and which one `strata-pack` means

`setup.py` calls **two** packers, for two different packing modes:

- `tools/iq_pack.py` (602 lines) - every tensor as the GGUF stores it, the experts read from the GGUF at start
  (seconds to build). This is the path `setup.py` takes for **every model except Q2_0**, so it is the one this
  deployment actually used (UD-Q4_K_XL, `--compat-bf16` from the family's `pack_args`). Its `--experts-bin` is the
  low-RAM mode.
- `tools/strata_pack.py` (452 lines) + `tools/pack_index.py` (260 lines) - the canonical Q2_0 pack: `experts.bin`,
  `dense.bin`, `embd.bin`, `manifest.json`, then the flat `index.txt` the engine's loader reads. `setup.py` takes
  this path for Q2_0 on AVX-512, and its own docstring is explicit that **it** is the port target: "the C++
  `tools/strata-pack` is a port of it, and if the two disagree the C++ is wrong".

So the target `CMakeLists.txt` names (`strata-pack`) is the second one. An earlier revision of this file named
`iq_pack.py` for it, which was wrong; `iq_pack.py` is the mode our model uses, and worth porting for that reason,
but it is not what the target name refers to.

**Acceptance is the same for both: byte-identical output against the Python tool.** `serve/` and the engine read
these files, and the canonical pack additionally carries a rule of its own - `strata_pack verify` re-reads the
written pack and requires `decode(pack) == dequant_ggml(source)` bit for bit, per tensor.

**Landed: the acceptance harness** (`tools/gen_pack_golden.py`, `tests/artifact/pack_golden/`). A synthetic
model (3,840 bytes, ten tensors) chosen to reach every branch of the packer's conversion table - a quantized
tensor served natively, F32 written as it is, F32 whose values are exactly BF16 (converted exactly), F32 whose
values are not (rounded, `--compat-bf16`), a quantized tensor dequantized then rounded, the PLE key in a native
encoding, F32 narrowed to F16 by the loader (kind 3), F16 widened to F32, the PLE table that is not in the pack
at all, and an expert - plus the reference's pack of it: `index.txt` (8 rows, kinds 0/2/3/4), `dense.bin`
(2,304 bytes) and `conversions.json` (5 records with their sha256 and errors). `--check` reruns the reference and
compares, so the fixture is known to be current.

`compat-bf16.json` is deliberately not part of it: it records the source path it was given, so it differs on
every machine. It is a note for a human, not an artifact the engine reads.

**Landed: the writer** (`include/strata/artifact/pack_writer.hpp`, `src/artifact/pack_writer.cpp`,
`tests/artifact/pack_test.cpp`). It packs a model into `index.txt` and `dense.bin`, and on the fixture above it
reproduces the reference's two files **byte for byte**: 8 tensors, 2 served natively, 5 conversions, arena 2048
bytes, dense 2304 bytes - the same numbers the reference printed.

The conversions are not reimplemented: `f32->bf16` is `strata::kernels::bf16_from_f32` and `f32->f16` is
`f16_from_f32`, the engine's own (ggml's rounding, and the loader's), and the dequantizers are
`include/strata/artifact/dequant.hpp`, transcribed from ggml. What the new file owns is the format: the FORM
table, the index rows, the two address spaces (`dense.bin` is a plain concatenation padded to 64 bytes; the index's
pool offsets are the running sum of the DESTINATION sizes, so a tensor the loader narrows to F16 advances the pool
by half of what it occupies), and the conversion records.

The one line allowed to differ is the index's first, which names the writing tool: a C++ packer claiming to be
`tools/iq_pack.py` would be a lie, and the engine does not read the comment lines. `pack_test` normalizes the tool
name on both sides and requires the rest of the line to match.

**Landed: `conversions.json`, and the packer slice is closed.** The records were already produced; what was missing
was the per-tensor `source_sha256`, and there is no SHA-256 anywhere else in the tree, so it is a small header of
its own (`include/strata/artifact/sha256.hpp`, with `tests/artifact/sha256_test.cpp` against the FIPS 180-4 vectors
and both padding boundaries - a message ending exactly on a block, and one needing a second block for its length).
The emitter is not a general JSON writer: one shape, laid out the way Python's `json.dumps(indent=1)` lays it out,
including the one detail that is an accident of the reference's code - on the path where the loader narrows a tensor
(kind 3) it sets `method` after `exact`/`max_abs_err`, so that key lands last. Reproduced, because the file is
compared byte for byte and `json.loads` does not care either way.

`pack_test` now compares all three files, and the float fields agree too: `max_abs_err` is Python's shortest
round-trip repr, and `std::to_chars` plus a `.0` for the whole numbers matches it on every value here. The residual
risk is a value whose shortest repr crosses Python's exponent threshold, which no conversion error in this pack does.

**Landed: the expert path.** `pack_expert_layout` writes `native_experts.txt` - v3, or v4 when a shard boundary
falls inside a layer (the per-role `gate,up,down` shard column, #255) - `pack_expert_layer_bytes` cuts
`experts.bin` (per expert in turn: gate | up | down, the order the reference's `reshape(n_expert, -1)` produces),
and `pack_experts_source_json` writes the sidecar. The other side of this format is the engine's own
`src/kernels/cpu/expert_layout.cpp`, which parses exactly these columns and refuses a version it does not know;
the writer was written against it, not against a guess.

The fixture grew one complete expert layer, and it had to: the expert count is read off the router's second
dimension, and it must be a multiple of 32, because ggml (and gguf-py) lay a Q8_0 block along the LAST axis -
which is the expert axis, `shape[2]`. 32 is the smallest count that is both layout-exercising and storable
quantized. `pack_test` now compares all six files byte for byte.

**What is left of `iq_pack.py`:** `--base` (`index_from_base`, packing a delta against an existing pack). Nothing
in the repo passes `--base` - `setup.py` uses the tool without it - so this is the last piece rather than the next
one. The standalone and expert paths, which are everything `setup.py` asks for, are complete and proven.

**Started on `strata_pack.py` (the real `strata-pack`):**

- `include/strata/artifact/canonical_codes.hpp` - the code packing every S2/S4/S8 form goes through (one code per
  element into bytes, LSB-first, 4 codes per byte, Q2_0's own convention kept for every form so the kernel has one
  unpacking loop) and the scale-precision choice (FP16 only when the round-trip is exact, FP32 otherwise).
  `tests/artifact/canonical_codes_test.cpp` compares both against the reference's own answers over 9 code patterns
  and 2 scale sets.
- `include/strata/artifact/canonical_mapping.hpp` - a source type's mapping into the canonical planes: the
  registry (`tools/canonical_xcheck.py`'s `Mapping`) and **all thirteen** of its source types - Q2_0, Q3_K, Q4_0,
  Q4_K, Q5_0, Q5_K, Q6_K, Q8_0, IQ4_NL, IQ4_XS and the three float types - each with encode and decode.
  `tests/artifact/canonical_mapping_test.cpp` compares the code plane, the scale plane, the offset plane and the
  decoded values against the reference's, and checks that a type with no mapping is refused rather than
  approximated.

- `include/strata/artifact/canonical_entry.hpp` - a tensor's planes in the pack and the manifest entry for them:
  `pack_tensor_entry` (the layout), `pack_decode_entry` (the decode that may use ONLY what the entry says) and
  `pack_verify_tensor` (the rule below). `tests/artifact/canonical_entry_test.cpp` compares the blob bytes, the
  entry and the decoded values for all thirteen types, and checks that an incomplete entry is refused.
- `include/strata/artifact/pack_index.hpp` + `src/artifact/pack_index.cpp` - the manifest entries flattened into
  the engine's `index.txt`: the columns the engine cannot re-derive (which activation a weight wants, and an
  explicit `kind` for how the engine's form differs from the pack's), every check the reference makes, and the
  three plane sizes **emitted** rather than left for the loader to re-derive.
  `tests/artifact/pack_index_test.cpp` compares the whole file - header, row order, every column, the pool total -
  and checks that a manifest whose planes are not contiguous is refused.
- `include/strata/artifact/expert_arena.hpp` - the expert arena's blob layout: `pack_expert_blob` (the relayout)
  and `unpack_expert_role` (its inverse). `tests/artifact/expert_arena_test.cpp` compares every byte of the blobs
  against the reference's own `experts.bin` and the inverse against its `rebuild_role`, checks the inverse
  reproduces each role's source run, that the layout is not the identity, and that the real model's 1,382,400-byte
  geometry is intact (the fixture runs at H=256/FF=64/NE=2, so those constants would otherwise go unchecked).

The arena is **85% of the pack by bytes and the part Phase 2 reads 480× per token**, and its layout is the one
thing no other slice shares: the Q2_0 codes and their fp16 scales in **separate planes**, with the gate and up
**rows interleaved** (`g0,u0,g1,u1,...`) so one pass over the activation serves both - the GGUF stores gate, up and
down as three separate tensors, so a fused gate/up pass would otherwise read two streams from three arrays. The
conversion is a pure relayout: every byte is copied, none computed, which is why `dequant(pack) == dequant(ggml)`
holds by construction rather than by check.

Two statements are kept apart on purpose, mirroring the reference: the forward follows `pack_layer.build` (plane
by plane) and the inverse follows `rebuild_role` (block by block). `rebuild_role`'s own docstring gives the reason
- "if it and `pack_layer` were the same code the arena would be unverified by construction" - and what verifies
the pair here is the reference's own `experts.bin` in the golden, not either of them.

The golden is generated by running the reference's `build()` over a **synthetic GGUF with the geometry scaled**
(`tools/gguf_writer.py` can write Q2_0, which `gguf-py` cannot), so its layout code is exercised unchanged at a
size that fits in the tree.

- `include/strata/artifact/pack_manifest.hpp` + `src/artifact/pack_manifest.cpp` - the pack's `manifest.json`
  (`pack_manifest_json`) and `pack_occupied` (the marker check). `tests/artifact/pack_manifest_test.cpp` compares
  the whole file against the reference's own `build()` output for a fixture model, and checks that each of the
  three pack markers is recognised.

**The manifest is written LAST, and that is what makes it the pack's completion marker.** `PACK_MARKERS` is
(manifest.json, index.txt, native_experts.txt): any one of them in a directory means it already holds a pack, and
a build refuses to overwrite it without `--force` - checked *before* anything is opened or written, because a build
that overwrote whatever pack was there used to do so silently. The manifest is not read by the engine (that is why
`pack_index.py` exists at all); it is the pack's description of itself, for the tools.

The JSON is laid out the way `json.dumps(indent=1)` lays it out, because the file is compared byte for byte and
two writers of one format have to agree: the key order is `tensor_entry`'s own construction order, a list is one
element per line, and there is no trailing newline. Two subtleties cost a round each - the opening brace of a
nested object belongs to the caller's line, not to the object's own indent, and a float is Python's shortest
round-trip repr with a `.0` appended for whole numbers. The one machine-specific part is the two source paths,
blanked on both sides.

**Two things this slice found, both by refusing a fixture:**

- The engine's loader reads a quantized tensor's planes as **one contiguous span**, so `pack_index.py` refuses a
  manifest whose planes are not contiguous - which in practice means every plane's byte size must be a multiple of
  64, because `put()` pads to 64 between planes. Real tensors are large enough that this always holds; the first
  fixture (three blocks a tensor) was not, and the reference refused it.
- The reference's `dst_bytes != src_bytes + widen_here` check counts **only the scale plane's** widening, not the
  offset plane's. It holds for every real pack because `offsets_fp16` is false for all 54 tensors that have an
  offset plane - `pack_index.py`'s own note says so - but a fixture with fp16-exact offsets trips it. The port
  reproduces the arithmetic rather than "fixing" it, so the C++ refuses exactly what the reference refuses; the
  gap is commented where it lives.

**Why the C++ port takes entries and not a manifest.** `pack_index.py` reads `manifest.json` because the *engine*
must not have a JSON parser: it would be a new, unaudited component whose failure mode is a wrong byte offset. A
C++ `strata-pack` that writes the manifest itself already holds these entries in memory, so it needs no parser
either - and the `--pack <dir>` re-derivation mode is the one part of the Python tool that does not port.

**`verify` is the rule the whole tool is organised around: for every tensor, `decode(pack) == dequant_ggml(source)`
BIT FOR BIT.** It is a comparison against the model, not the pack against itself, and it is exact rather than a
tolerance, because a pack that is merely close is a pack whose weights differ from the model's. A shape difference
counts too: it means the entry and the source disagree about the element count, which is the mistake Q2_0's packed
code representation once made (1/4 of the truth, and undetectable by its own decoder).

Two things make that check trustworthy rather than decorative, and the test asserts both: the engine's
transcription and the reference's own dequantization must agree on the same bytes first, and a verifier that cannot
fail is worth nothing - so one flipped bit in the segment must be reported as a mismatch.

The layout rule is `put()`: every plane starts on a **64-byte boundary**, and a plane's offset is **absolute in the
segment** the tensor is written into, not relative to the tensor's own bytes. A port that made them relative would
decode its own output perfectly and disagree with the reference on every tensor but the first. P16 is written only
where the round-trip is exact, because the alternative is a silently lossy tensor; the reference raises "P16 would
lose precision, so this tensor needs P32" instead, and so does this.

Q4_K and Q5_K are the first with an **offset plane** (`offset = -(dmin*m)`, the negation being exact so adding it
is bit-identical to subtracting the min) and the first whose group is smaller than the block: 8 groups of 32 in a
256-element block, because each 64-element step of the source takes the low AND high nibbles of the same 32 scale
bytes. Their scales come out of `get_scale_min_k4`, where the low four groups keep their 6 bits in a byte and the
high four borrow the top two bits of the low groups' bytes - reading `scb[j]` alone is right for the first half of
the groups and silently wrong for the second. Q5_K adds a 5th bit whose mask walks 1,4,16,64 while the 32 mask
bytes are re-read by every step, the same trap as Q3_K's hmask.

The traps the mappings pin, all of which still produce plausible weights when got wrong:

- Q8_0's code plane is **unsigned** (the source's `int8_t` stored +128, which is why the two Q8_0 tensors once
  decoded 128 codes from where they belonged).
- Q4_0's and IQ4_NL's **low nibbles fill the first half** of a block and their high nibbles the second - not
  interleaved, with Q4_0's -8 applied before the multiply.
- Q2_0's codes must be **one 2-bit code per element**, not the 16 packed bytes the source holds: a packed
  representation passes every check its own decoder gives it, and `codes.size` is what the pack reads as a
  tensor's element count (for Q2_0 that came out at 1/4 of the truth).
- IQ4_NL's code is an **index into a non-affine table** (`kvalues_iq4nl`), so it has no bias analogue at all.
- BF16 widens to **P32, not P16**: bf16 has the f32 exponent range and fp16 does not, so an fp16 container would
  send large-magnitude weights to infinity.

The rest of that tool is the larger job, and its shape is worth stating before starting: the encoding is driven by
`MAPPINGS` in `tools/canonical_xcheck.py`, which its docstring calls the specification of record (measured
bit-exact over 757,139,840 blocks) - and **all thirteen of its mappings are now ported**, as are the per-tensor
plane layout, the manifest-driven decode and `verify`, the engine's `index.txt`, the expert arena with its
inverse, the manifest, and now the build itself. Note that `strata_pack.py`'s docstring points at
`docs/pack-format.md` and **that file does not exist in the tree**: the code is the spec.

- `include/strata/artifact/pack_build.hpp` + `src/artifact/pack_build.cpp` - `pack_build`, the orchestration over
  everything above: the expert arena, the two segments, the hashes, the manifest. `tests/artifact/pack_build_test.cpp`
  builds a whole pack from the fixture model and compares **every file** against the reference's own `build()` and
  `pack_index.py` output - `dense.bin`, `embd.bin`, `experts.bin`, `index.txt`, `manifest.json` - plus the source
  hash, which is computed here and matched against the reference's hash of the same bytes.

**Two things about the order, both from the reference and both load-bearing.** The marker check comes *first*,
before anything is opened or written: a build used to overwrite the four files of whatever pack was in the
directory and say nothing. And the manifest is written *last*, which is what makes it the completion marker - a
build that stops part-way leaves `experts.bin` without a manifest, so the next build makes the whole thing again
rather than leaving the old manifest over partly new data.

The engine's `index.txt` stays a separate step, as it is in the reference (`setup.py` runs `pack_index.py` after
the build): `pack_build` fills `PackBuildResult::tensors`, and `pack_index_text` turns that into the file.

**The pack-layout arena TOOL (`tools/pack_layer.py`) is ported.** The layout FUNCTION was already here
(`expert_arena.hpp`, held by `expert_arena_test`); what was missing was the tool around it - read a model's GGUF,
cut every layer's experts into blobs, write the arena and the sidecar manifest. `strata-pack-layer` does that, and
it reads the geometry FROM THE FILE rather than hardcoding it: the reference's H/FF/NE are module constants it
ASSERTS the tensors against, so a tool that derives them cannot disagree with the file it was given. The manifest
is `json.dumps(indent=2)`'s layout, hand-written for its one shape as `pack_manifest.cpp` did for indent=1.
`tools/gen_pack_layer_golden.py` commits a fixture GGUF and records the reference's own `build()` over it;
`tests/artifact/pack_layer_test.cpp` runs the tool and compares the arena's digest, the per-layer stdout and the
manifest, byte for byte.

**The MTP draft head's packer (`tools/mtp_pack.py`) is ported.** It is the third GGUF-writing tool and the only
one that is not hand-rolled: it uses llama.cpp's gguf-py, so `strata-mtp-pack` reproduces gguf-py's container
layout - `general.architecture` as the first KV (the writer's own constructor adds it), a tensor directory
carrying the RUNNING offset of the aligned tensor sizes, and a data section that is the tensors in insertion
order with alignment padding after each. `gguf_write.hpp` supplies the primitives and grew only an offset
argument and a padding helper.

The arithmetic is in `mtp_pack_quant.hpp`, and it is the reference's, not ggml's. Its Q2_0 is not the main
model's: it searches 17 candidate block scales for the one that MINIMIZES squared error over the ternary grid,
then quantizes against the STORED fp16 scale - where `gguf_writer.py` and `canonical_mapping.hpp` set `d = amax`
and never use the +2 level. Its Q8_0 is not `q8_0_quant.hpp`'s either: that one rounds half away from zero
(ggml's `roundf`, transcribed for `mtp_rt.py`) and this one rounds with numpy's `rint`, half to EVEN - the same
everywhere except on an exact tie, and an exact tie is a different byte. The squared-error sums and the report's
means go through numpy's PAIRWISE summation, transcribed: a plain left-to-right float32 sum is a different
number, and a different number picks a different scale on a near-tie.

`tools/gen_mtp_pack_golden.py` writes a committable checkpoint and records the reference's output for all three
expert formats; `tests/artifact/mtp_pack_test.cpp` compares the GGUF byte for byte and the sidecar report (whose
per-tensor `seconds` is wall-clock and is the one field normalized on both sides).

### 3. The expert profile and the eviction benchmark (`tools/make_profile.py`, `tools/bench_eviction.py`)

`make_profile.py` (126 lines) reads `--dump-routing` traces, ranks every (layer, expert) pair and writes the
profile `--expert-profile` seats the VRAM tier from; `bench_eviction.py` (405 lines) replays the same traces
against candidate eviction policies (fill-only, lru, lfu-decay, Belady's MIN) without a GPU. Both are pure logic
over local binary inputs with deterministic output - no network, no third-party package, no new surface.

**This one needed a refactor before it could be a port.** `read_expert_profile` and `write_expert_profile` already
existed in C++, in `src/core/expert_cache.cpp` - which lives in `strata_engine`, a library built only when a GPU
backend is on. The profile FORMAT is file I/O and has nothing to do with a GPU, so a CPU-only tool could not link
the one implementation of the format that already existed and would have had to write a second one. The format,
its ranking and the per-layer slot rule moved to `strata_profile` (`include/strata/core/expert_profile.hpp`,
`src/core/expert_profile.cpp`), built unconditionally; `expert_cache.hpp` includes it and `layer_slot_range` is
now `profile_slot_range`, one rule with two callers instead of two copies that can drift.

- `strata-profile` (`src/core/profile_main.cpp`) - the `make_profile.py` CLI. Writes the profile, then reads it
  back and asserts the round trip, as the reference does.
- `tests/core/expert_profile_test.cpp` - against the reference's own output. `tools/gen_eviction_golden.py` runs
  both reference tools over two committed traces (deliberately awkward: out-of-range ids, the maximum layer and
  expert, repeated pairs that force a tie) and records the ordered codes, the frequencies and the reference's
  profile BYTES for each mode (default, `--reorder`, `--no-base`). The port's reader, its frequencies and all
  three modes match byte for byte.
- `strata-evict` (`src/core/evict_main.cpp`) - the `bench_eviction.py` CLI: replays the traces against all four
  policies at any slot count, both scopes, with a profile-seated start, and writes the reference's CSV.
- **The reference CRASHES on a configuration the port refuses.** `--per-layer` with fewer slots than layers makes
  `q = slots / n_layers` 0, so every layer but the last has no capacity; `Lru.lookup` tests `len(od) >= capacity`
  (true for an empty cache at capacity 0) and then pops an empty OrderedDict - `KeyError: 'dictionary is empty'`.
  There is no reference output for that configuration to be a port of, so `eviction_config_error` refuses it with
  the reason rather than reporting numbers the reference cannot produce. `fill-only` never evicts, so it is allowed.
- `tests/core/eviction_sim_test.cpp` - the whole report and the whole CSV, per configuration. The tight-slot
  configuration is the point: 10 distinct pairs against 4 and 8 slots, so the policies evict and diverge (oracle
  reaches 4 hits where fill-only gets 3). **A fixture that fits every slot count has all four policies reporting
  the same rate, and a golden like that passes for a port that got every eviction wrong** - the first version of
  this fixture did exactly that.
- `strata-profile-mix` (`src/core/profile_mix_main.cpp`) - the `make_profile_mix.py` CLI: a profile from several
  traces each counting the SAME (every trace becomes a distribution and is weighted), where `strata-profile` adds
  raw counts and so lets the longest trace decide. The reference keeps them apart and so does this: the two
  rankings differ, and a flag on one code path would hide that. Accepted on the reference's own profile bytes for
  two spec sets, one of which covers a weight, a ZERO weight (a pair present with score 0 is ranked, not filled)
  and an empty trace, which the reference skips.
- `strata-cache-replay` (`src/core/cache_replay_main.cpp` + `src/core/cache_replay.cpp`) - the
  `replay_cache_policy.py` CLI: the engine's OWN adaptive policy (`adapt()`), replayed offline so a parameter set
  can be measured on a trace without a GPU. Accepted on the reference's own stdout, line for line, for two
  parameter sets - and on the window GROUPING it loaded, which is the subtle part (a window is a run of groups
  whose layer rises; consecutive records with one layer are one group; only windows with one group per layer are
  kept, so a short trace loads as none at all).

  **THE MODEL AND THE ENGINE DISAGREE, AND IT IS IN THE ENGINE'S `adapt()`.** The reference sorts its candidate
  and victim lists with `np.argsort(..., kind="stable")`; `generate.cpp`'s inline `adapt()` sorts the same lists
  with `std::sort`, which is not stable. Experts with EQUAL usage - which any trace with repeated routing produces
  constantly - are therefore ordered differently, and the two pick different victims. The simulated rate and the
  engine's measured rate can differ for the same trace and parameters, and neither is "wrong": the engine is what
  ships, the reference is the model that was used to size things. This port follows the reference, because the
  reference's own output is its acceptance; changing either side to match the other is a separate decision with a
  measured consequence, and not one to make silently inside a port. **The engine's `adapt()` cannot be reused here
  either way**: it is a lambda closed over the live cache, the exchange buffers and the device, not a function of
  (usage, resident, policy).

- The reader is one function, not two. The reference walks the same trace records twice - `make_profile` for
  frequencies, `bench_eviction` for order - and the port derives the frequencies from the ordered codes, which is
  equal by construction. `replay_cache_policy`'s reader is a THIRD shape and stays separate on purpose: it takes
  `k` from the first record and assumes every record is that wide, where the other two read each record's own `k`
  and skip out-of-range ids. The port keeps each one's semantics rather than unifying them into a behaviour
  neither has.

- `strata-draft-vocab` (`src/tokenizer/draft_vocab_main.cpp`) - the `draft_vocab.py` CLI: the vocabulary rows the
  MTP draft head may propose, extended by whole scripts (`--add`) or by the most frequent tokens of a text corpus
  that cover `--coverage` of its occurrences (`--corpus`). Accepted on the reference's own stdout for four
  configurations AND on the subset bytes it wrote. The fixture vocabulary is deliberately awkward: it carries a
  token whose bytes are a PREFIX of a multi-byte character (the reference's `token_text` gives None for it, and a
  port that read it as text would count a Han token that is not one) and a CONTROL literal, so `token_type` is
  exercised rather than assumed all-normal.
- `include/strata/tokenizer/gguf_vocab.hpp` - the first way to load a real vocabulary into the C++ Tokenizer. The
  engine does not tokenize (it takes a prompt as token ids), so until now the port's tests built vocabs from a
  golden. It is its own header so `tokenizer.hpp` keeps no dependency on the GGUF reader, and it refuses a GGUF
  whose `tokenizer.ggml.pre` is not `qwen35` rather than tokenizing another pre-tokenizer's text with this one.
- **`MetaValue::items` holds a SAMPLE of an array, at most 64.** That is deliberate (the engine's metadata needs
  the count, not the contents, and a vocabulary is 248,320 strings) but it is a trap: a caller that reads `items`
  as the array gets 64 tokens of 271 and no error. The loader hit it - the merge `Ġ t` named token 116, outside
  the sample, and construction failed with "names a token outside the vocabulary" rather than anything pointing at
  the reader. `GgufFile::array_items(key)` now re-reads an array in full from the mapped header (the array's byte
  offset is recorded in `MetaValue::at`), so a caller that needs the contents asks for them.

### 4. Not portable without inventing surface

`serve/server.py` (6,521 lines) is `http.server`, JSON, the OpenAI and Anthropic request translation and the
web app's backend; `serve/frontend.py` renders the chat template through **jinja2** (`serve/chat_template.jinja`);
`serve/mcp.py` is an MCP client over stdio; `setup.py` is the installer's questions and engine
selection. Moving any of these to C++ means reimplementing an HTTP server, a Jinja template engine, a JSON
schema layer or a package installer that already exists - new surface, no measured gain, and a second place for
the API contract to drift. They stay in Python, which is what the project decided when it put the engine behind
a pipe and kept the API in front of it.

The bridge that makes slices 1 and 2 cheap already exists: the server drives the engine as a child process
(`strata generate`, a FIFO and control lines), so a C++ tool is a new executable, not a new binding.

## What "done" looks like for a port

Every slice above lands with the reference implementation still in the tree and a test that compares against
it - the round-trip corpus for the tokenizer, a byte-for-byte pack for the packer. A port that cannot show that
comparison is a rewrite of the specification, not a port.
