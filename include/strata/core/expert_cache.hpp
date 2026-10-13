// include/strata/core/expert_cache.hpp - R4: the VRAM-resident expert tier.
//
// **WHY THIS EXISTS, IN ONE PARAGRAPH.** The CPU expert pool is the engine's largest single cost: measured,
// **663.6 MB of expert bytes per token at ~40 GB/s = 16.2 ms**, on a token of ~53 ms, and `test_expert_pool`
// says the pipeline hides **1.055 ms of 19.035** - the CPU work is 96% exposed because the residual chain is
// strictly serial. The only way past it is to stop reading those bytes: put some of the 48 x 512 experts in
// VRAM and let the GPU compute them, so the CPU reads only the misses.
//
// **THE DESIGN IS SIZED AGAINST MEASUREMENTS, NOT AGAINST THE PLAN'S ASSUMPTIONS.**
//
//   * `h_expert` at 4,105 slots, **ten-fold leave-one-out: 0.6447 [0.6110, 0.6750]** (`tools/routing_kfold.py`).
//     The 0.6573 the README published is in-sample and 0.4720 measured a single-prompt CORPUS, not the metric.
//   * VRAM, measured on a live run rather than at startup: **6414 MiB free** with the engine at steady state,
//     against **5413 MiB** for 4,105 slots. It fits, with ~1 GB spare. `strata-device`'s planner agrees and says
//     FITS, but the planner runs before the dense arena exists, so the live figure is the one that counts.
//   * `h_layer` held-out is **0.0456** - only 4.6% of (layer, token) pairs have all ten experts resident - so
//     this cannot be a per-layer grouped kernel and the three-way split is not an optimisation.
//
// **WHAT THIS FILE IS.** The SLOT STORAGE and the RESIDENCY TABLE: it allocates the VRAM, fills it from the host
// arena, and answers `(layer, expert) -> slot or -1`. It computes nothing itself: the GPU computes the experts it
// holds in the verify window's grouped kernels and the prompt path (the hit/miss split), and the CPU pool the rest.
// (Written when nothing consumed the slots yet and `--expert-cache` defaulted to 0; setup's configs use `auto`.)
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "strata/core/expert_profile.hpp"
#include "strata/core/vmm.hpp"

namespace strata::core {

/// Where one expert's VRAM lives, or -1.  `int32_t` and not a bool because the slot index is what the kernel
/// needs, and a table that has to be re-derived from a bitmask is a table that will be re-derived differently
/// in the two places that use it.
inline constexpr int32_t kNotResident = -1;

/// The profile's format, its ranking and the routing trace it is built from live in
/// `strata/core/expert_profile.hpp` - file I/O and arithmetic with no GPU dependency, which is why a CPU-only
/// tool (the profile builder, the eviction benchmark) can link them.  `read_expert_profile`,
/// `write_expert_profile` and `rank_learned_profile` used to be declared here; the cache uses them from there.

/// Free device memory for the expert cache.  On a unified memory system - one memory for the CPU and the GPU, such as
/// the DGX Spark (GB10); CUDA reports it as cudaDevAttrIntegrated - cudaMemGetInfo counts only MemFree, not the page cache - which, with the experts read from the GGUF in place, is mostly those
/// files' clean pages and gives way to a device allocation.  There the figure is MemAvailable less 6 GiB for the OS
/// and the engine's host side (STRATA_UMA_HEADROOM_GIB).  Elsewhere it is cudaMemGetInfo's.
size_t device_free_bytes();

class ExpertCache {
public:
    ExpertCache() = default;
    ~ExpertCache();
    ExpertCache(const ExpertCache&) = delete;
    ExpertCache& operator=(const ExpertCache&) = delete;

    /// Allocates `n_slots * blob_bytes` of DEVICE memory and a `n_layers * n_expert` residency table, and
    /// **CHECKS THE ALLOCATION AGAINST WHAT THE CARD ACTUALLY HAS** rather than assuming the planner was right.
    /// A cache that silently allocates less than it was asked for would report a hit rate for slots it does
    /// not have, which is the shape of error this project keeps paying for.
    bool open(int64_t n_slots, int64_t n_layers, int64_t n_expert, int64_t blob_bytes, std::string& err);
    /// Plan v0.3 P6: slots of the given sizes, back to back (a native pack's blobs differ per layer, and a
    /// profile-filled tier never moves an expert to another layer's slot, so each slot keeps its first size).
    bool open_sized(const std::vector<int64_t>& slot_bytes, int64_t n_layers, int64_t n_expert, std::string& err);
    /// Byte offset of each slot in the arena (null for uniform slots).
    const uint64_t* slot_offsets() const { return off_.empty() ? nullptr : off_.data(); }
    void close();

    bool valid() const { return base_ != nullptr; }
    /// The slots usable now: all of them, unless `shrink` gave the tail's VRAM back (#533) - then the ones wholly
    /// inside the VRAM still backed, always a prefix.  The prompt path's loan (the last slots) and every report
    /// follow this; `full_slots()` is what `open` allocated.
    int64_t slots() const { return live_slots_; }
    int64_t full_slots() const { return slots_; }
    /// Slots actually claimed.  Not the same as `slots()` - the cache does not evict, so a run that routes
    /// fewer distinct experts than there are slots leaves the rest empty.
    int64_t resident() const { return per_layer_ ? admitted_ : next_free_; }
    /// The bytes of `slots()` (all of the arena unless shrunk); `full_bytes()`: of every slot.
    int64_t bytes() const { return slot_end(live_slots_); }
    int64_t full_bytes() const { return slot_end(slots_); }
    double gib() const { return (double) bytes() / 1073741824.0; }

    /// **#533: HOT VRAM RESIZE, OPT-IN (--vram-elastic).**  With a segment size set before `open`/`open_sized`, the
    /// arena is one reserved address range backed by physical segments of that size (CUDA virtual memory
    /// management) instead of one cudaMalloc, so `shrink` can give the tail's VRAM back to the driver - for another
    /// program - and `grow` can take it again, while every slot keeps its address (the captured graphs, the verify
    /// plan's pointers and the prompt path's loan all hold addresses into it).  0 (the default): one cudaMalloc, as
    /// always.  CUDA only: a HIP build refuses it at open.  The slots past `slots()` after a shrink must hold no
    /// resident expert and no loan: the CALLER evicts them first (they become CPU misses).
    void set_segment_bytes(int64_t seg_bytes) { seg_req_ = seg_bytes > 0 ? seg_bytes : 0; }
    bool segmented() const { return !segs_.empty(); }
    int64_t segment_bytes() const { return seg_; }
    /// Bytes of the arena backed by VRAM now (a whole number of segments; the arena when not segmented).
    int64_t mapped_bytes() const;
    /// Unmaps every segment past the first `keep_bytes` (rounded UP to a segment boundary): `slots()` becomes the
    /// slots wholly inside what stays.  Waits for the device first.  False with `err` when not segmented or a driver
    /// call failed.
    bool shrink(int64_t keep_bytes, std::string& err);
    /// Maps segments again up to `want_bytes` (rounded DOWN to a segment, at most the arena; the last, shorter
    /// segment only when `want_bytes` covers the arena).  Stops at the first segment the driver cannot back (false,
    /// `err`: what was mapped by then stays).  The new slots are empty until the caller fills them.
    bool grow(int64_t want_bytes, std::string& err);
    /// The number of leading slots that fit wholly inside the first `bytes` bytes.
    int64_t slots_within(int64_t bytes) const;
    /// The bytes the first `n` slots span.
    int64_t bytes_of(int64_t n) const { return slot_end(n < 0 ? 0 : n > slots_ ? slots_ : n); }

    /// `(layer, expert)` -> slot index, or `kNotResident`.  Bounds-checked: a bad layer or expert returns
    /// `kNotResident` rather than reading whatever is adjacent in the table.
    int32_t slot_of(int64_t layer, int64_t expert) const;
    /// Claims the next free slot for `(layer, expert)`.  Returns the slot, or `kNotResident` when the cache is
    /// full - **it never evicts**, because eviction policy is a measured question (`R4.1`'s LFU-decay vs LRU
    /// sweep) and a placeholder policy would set the hit rate that everything downstream is then sized against.
    int32_t admit(int64_t layer, int64_t expert);

    /// Publish a same-layer replacement after its slot copy has completed.
    void replace(int64_t layer, int32_t old_expert, int32_t new_expert) {
        auto& old = residency_[(size_t) layer * n_expert_ + old_expert];
        residency_[(size_t) layer * n_expert_ + new_expert] = old;
        old = kNotResident;
    }

    /// **R4.2g: GIVE EACH LAYER ITS OWN SLOTS.  ROUND 328 MEASURED WHY THE GLOBAL FORM CANNOT WORK.**
    ///
    /// `admit`'s original policy hands out slots in ARRIVAL ORDER from one counter shared by every layer
    /// (`if (next_free_ >= slots_) return kNotResident;`).  A position looks at 10 experts in each of 48
    /// layers, so the first `n_slots` distinct (layer, expert) pairs are **about 26 layers of POSITION 0** -
    /// the cache fills entirely inside the first position it ever sees and never changes again, so hits are
    /// confined to those few layers for the rest of the run.
    ///
    /// Measured, `--expert-cache 256`: **1781 of 60000 = 2.97%**.  `Memory/cache_allocation.py` scores the
    /// same run's routing and says **8 slots per layer is worth 21.4%** and 64 is worth **70.4%**.  The cache
    /// was never short of capacity - it was handing all of it to two layers.
    ///
    /// With this on, layer `l` may only use slots `[l*q, (l+1)*q)` where `q = slots()/n_layers`, the last
    /// layer taking the remainder.  **EVICTION IS STILL ABSENT** - the property the comment above calls
    /// deliberate is preserved exactly; only WHICH slot a layer may take changes.  Default off, so every
    /// caller that predates this flag is bit-for-bit unaffected.
    void set_per_layer_admission(bool on) { per_layer_ = on; }
    bool per_layer_admission() const { return per_layer_; }
    /// The slot range layer `l` may admit into under per-layer admission.  Exposed so a test can check it.
    void layer_slot_range(int64_t layer, int64_t& lo, int64_t& hi) const;

    /// The device address of one slot.
    uint8_t* device_slot(int32_t slot);
    const uint8_t* device_slot(int32_t slot) const;

    /// Copies `(layer, expert)`'s blob from `host_blob` into `slot` on `stream`.  Asynchronous: the caller
    /// orders it.  Returns false if the indices are out of range rather than reading past the arena.
    /// `bytes` (plan v0.3 P6): the blob's own size when it is smaller than the slot (a native pack); 0 = the slot.
    bool fill_slot(int32_t slot, const uint8_t* host_blob, void* stream, std::string& err, int64_t bytes = 0);

    /// **THE SAME COPY, BUT BLOCKING, AND THE PROFILE FILL NEEDS IT.**  `fill_slot` is asynchronous because on
    /// the token path the fill and the kernel that reads it are ordered by one stream and waiting would be a
    /// synchronisation per layer.  At STARTUP there is no such ordering to lean on: the source is pageable host
    /// memory, `verify_slot` reads the slot back on the LEGACY stream, and a legacy-stream copy is not ordered
    /// against a `cudaStreamNonBlocking` one.  The first version used the async form and `verify_slot` refused
    /// the whole run with "slot 0 differs from the arena at byte 0" - which is the check doing its job.
    bool fill_slot_blocking(int32_t slot, const uint8_t* host_blob, std::string& err, int64_t bytes = 0);
    /// One slot of a pipelined startup fill.  `host` may be null when the `resolve` callback of fill_slots_pipelined
    /// supplies the pointer.  `tag` is the caller's own index (reported back on a failure).
    struct FillJob {
        int32_t slot = -1;
        int64_t bytes = 0;      ///< the blob's own size (0 = the slot)
        const uint8_t* host = nullptr;
        int64_t tag = 0;
    };
    /// The startup profile fill, pipelined: `threads` readers each take jobs, `resolve(job)` hands them the host bytes
    /// (a mapped file's pages fault in on that thread, so the reads overlap), the bytes are copied into the reader's own
    /// ring of page-locked staging buffers and sent with cudaMemcpyAsync on the reader's own stream, a buffer reused
    /// only after its copy's event.  Every stream has finished when this returns, so verify_slot (legacy stream) sees
    /// the slots.  Same bytes as `fill_slot_blocking` per job.  `threads` 0 (or no page-locked memory): the blocking copy
    /// per job in order, as before.  `resolve` runs on the reader threads (it must be thread-safe) unless threads is 0.
    /// Returns false with `err` set and `failed_tag` = the lowest failing job's tag.
    /// Reader threads for the pipelined fill: STRATA_FILL_THREADS (0 = the old blocking copies, 1 = one reader), default
    /// min(16, hardware threads).
    static int fill_threads_default();
    bool fill_slots_pipelined(const std::vector<FillJob>& jobs,
                              const std::function<const uint8_t*(const FillJob&)>& resolve, int threads,
                              std::string& err, int64_t& failed_tag);
    /// perf-review D-4: `fill_slot_blocking`'s copy on the same (legacy) stream, but queued: many slots are refilled
    /// with one `sync_queued` at the end instead of a wait per slot. Same ordering against earlier work, same bytes.
    bool fill_slot_queued(int32_t slot, const uint8_t* host_blob, std::string& err, int64_t bytes = 0);
    bool sync_queued(std::string& err);

    /// Reads `slot` back to the host and compares it to `host_blob`, byte for byte.  **THE ONLY THING THAT SAYS
    /// THE CACHE HOLDS THE EXPERT IT CLAIMS TO.**  A slot table that is right about indices and wrong about
    /// bytes produces a plausible token, which is exactly the failure this project has paid for most often.
    bool verify_slot(int32_t slot, const uint8_t* host_blob, std::string& err, int64_t bytes = 0);

    /// Slots filled so far, for the startup report.
    int64_t fills() const { return fills_; }

    /// The elastic K/V (--kv-grow): the arena in a VMM range (vmm.hpp) instead of one cudaMalloc, so the K/V can take
    /// single chunks of it and give them back. Applies to the next `open`; ignored where VMM is not available.
    static void set_vmm(bool enabled);
    /// The arena's range (null: one cudaMalloc).
    VmmRange* vmm_range() { return vmm_.get(); }
    /// Byte offset of slot `s` in the arena (s == slots(): the end).
    uint64_t slot_offset(int64_t s) const { return off_.empty() ? (uint64_t) s * (uint64_t) blob_ : off_[(size_t) s]; }

private:
#if defined(STRATA_USE_HIP)
    bool ensure_blocking_staging(std::size_t bytes, std::string& err);
    uint8_t* blocking_staging_ = nullptr;
    std::size_t blocking_staging_bytes_ = 0;
#endif
    int64_t slot_end(int64_t n) const { return off_.empty() ? n * blob_ : (int64_t) off_[(size_t) n]; }
    bool open_segmented(uint64_t want, std::string& err);
    void release_segmented();
    uint8_t* base_ = nullptr;
    int64_t live_slots_ = 0;            ///< #533: slots() - all of them unless shrunk
    int64_t seg_req_ = 0;               ///< #533: the segment size asked for (0: one cudaMalloc)
    int64_t seg_ = 0;                   ///< #533: the segment size used (a multiple of the driver's granularity)
    uint64_t reserved_ = 0;             ///< #533: the reserved address range's size
    std::vector<unsigned long long> segs_;   ///< #533: each segment's physical handle (0: unmapped)
    std::vector<int64_t> seg_size_;     ///< #533: each segment's size (the last one may be shorter)
    int64_t mapped_segs_ = 0;           ///< #533: segments [0, mapped_segs_) are backed
    std::unique_ptr<VmmRange> vmm_;    ///< the arena's range when it is in VMM (set_vmm)
    std::vector<int32_t> residency_;   ///< [n_layers * n_expert] -> slot or kNotResident
    int64_t slots_ = 0;
    int64_t n_layers_ = 0;
    int64_t n_expert_ = 0;
    int64_t blob_ = 0;
    int64_t next_free_ = 0;
    int64_t fills_ = 0;
    /// R4.2g.  `per_layer_` off (the default) leaves `next_free_` as the only admission counter, so the
    /// pre-existing path is untouched.
    bool per_layer_ = false;
    std::vector<int32_t> layer_next_;   ///< [n_layers] -> that layer's next free slot
    std::vector<uint64_t> off_;         ///< plan v0.3 P6: slot offsets (slots + 1 entries) when sized
    int64_t admitted_ = 0;
};

}  // namespace strata::core
