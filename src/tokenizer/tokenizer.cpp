// src/tokenizer/tokenizer.cpp
#include "strata/tokenizer/tokenizer.hpp"

#include "strata/tokenizer/pretokenizer.hpp"
#include "strata/tokenizer/utf8.hpp"

#include <algorithm>
#include <atomic>
#include <functional>
#include <queue>
#include <stdexcept>
#include <unordered_map>

namespace strata::tokenizer {
namespace {

std::uint64_t pair_key(int left, int right) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(left)) << 32) |
           static_cast<std::uint32_t>(right);
}

// The GPT-2 byte encoder: 256 bytes -> 256 printable codepoints, reversibly.  Bytes 33..126, 161..172 and
// 174..255 map to themselves; the other 68 (space, newline, the C0 controls and the high range that would be
// invisible) are shifted to 256+n.  The ORDER of that shift is the whole trick and getting it wrong is
// invisible in the output ids, so this follows the reference's `bytes_to_unicode` line for line.
std::vector<std::uint32_t> byte_alphabet() {
    std::vector<std::uint32_t> bs;
    for (int b = 0x21; b < 0x7F; ++b) {
        bs.push_back(static_cast<std::uint32_t>(b));
    }
    for (int b = 0xA1; b < 0xAD; ++b) {
        bs.push_back(static_cast<std::uint32_t>(b));
    }
    for (int b = 0xAE; b < 0x100; ++b) {
        bs.push_back(static_cast<std::uint32_t>(b));
    }
    std::vector<std::uint32_t> cs = bs;
    std::uint32_t n = 0;
    for (std::uint32_t b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }
    std::vector<std::uint32_t> out(256, 0);
    for (std::size_t i = 0; i < bs.size(); ++i) {
        if (bs[i] < 256) {
            out[bs[i]] = cs[i];
        }
    }
    return out;
}

const std::vector<std::uint32_t>& alphabet() {
    static const std::vector<std::uint32_t> table = byte_alphabet();
    return table;
}

}  // namespace

Tokenizer::Tokenizer(std::vector<std::string> tokens, std::vector<std::string> merges,
                     std::vector<int> token_types)
    : tokens_(tokens.begin(), tokens.end()) {
    static std::atomic<std::uint64_t> next_uid{1};
    uid_ = next_uid.fetch_add(1);
    for (std::size_t i = 0; i < tokens_.size(); ++i) {
        if (!ids_.emplace(tokens_[i], static_cast<int>(i)).second) {
            throw std::invalid_argument("vocabulary has duplicate tokens: " + std::to_string(tokens_.size()) +
                                        " entries, " + std::to_string(ids_.size()) + " unique");
        }
    }
    for (std::size_t i = 0; i < merges.size(); ++i) {
        const std::string& m = merges[i];
        const std::size_t sp = m.find(' ');
        if (sp == std::string::npos || m.find(' ', sp + 1) != std::string::npos || sp == 0 || sp + 1 == m.size()) {
            throw std::invalid_argument("merge " + std::to_string(i) + " is not a pair: " + m);
        }
        const auto left = ids_.find(std::string_view(m).substr(0, sp));
        const auto right = ids_.find(std::string_view(m).substr(sp + 1));
        if (left == ids_.end() || right == ids_.end()) {
            throw std::invalid_argument("merge " + std::to_string(i) + " names a token outside the vocabulary: " + m);
        }
        rank_[pair_key(left->second, right->second)] = static_cast<int>(i);
        const auto whole = ids_.find(m.substr(0, sp) + m.substr(sp + 1));
        merge_to_.push_back(whole == ids_.end() ? -1 : whole->second);
    }

    // Every byte's token, worked out once: the byte alphabet's codepoint, encoded as UTF-8, looked up.
    byte_id_.assign(256, -1);
    for (std::size_t b = 0; b < 256; ++b) {
        std::string ch;
        utf8::append(ch, alphabet()[b]);
        const auto it = ids_.find(ch);
        byte_id_[b] = it == ids_.end() ? -1 : it->second;
    }

    // The literals, longest first: `<|im_end|>` must win over a shorter literal that prefixes it.  stable_sort
    // so equal lengths keep their vocabulary order, as the reference's sorted() does.
    if (!token_types.empty()) {
        for (std::size_t i = 0; i < token_types.size() && i < tokens_.size(); ++i) {
            if (token_types[i] == 3 || token_types[i] == 4) {
                specials_.push_back(Special{tokens_[i], static_cast<int>(i), token_types[i]});
            }
        }
        std::stable_sort(specials_.begin(), specials_.end(),
                         [](const Special& a, const Special& b) { return a.text.size() > b.text.size(); });
        for (const Special& sp : specials_) {
            if (!sp.text.empty()) {
                special_first_[static_cast<unsigned char>(sp.text[0])] = true;
            }
        }
    }
}

std::vector<int> Tokenizer::merge_scan(std::vector<int> syms) const {
    // Lowest rank first, repeatedly - not left to right.  Equal ranks take the leftmost pair, which is what the
    // reference's strict `<` does.
    while (syms.size() > 1) {
        std::size_t best = 0;
        int best_rank = -1;
        for (std::size_t k = 0; k + 1 < syms.size(); ++k) {
            const auto it = rank_.find(pair_key(syms[k], syms[k + 1]));
            if (it != rank_.end() && (best_rank < 0 || it->second < best_rank)) {
                best_rank = it->second;
                best = k;
            }
        }
        if (best_rank < 0) {
            break;
        }
        syms[best] = merged(static_cast<std::size_t>(best_rank));
        syms.erase(syms.begin() + static_cast<std::ptrdiff_t>(best) + 1);
    }
    return syms;
}

std::vector<int> Tokenizer::merge_heap(const std::vector<int>& syms) const {
    // The same result as merge_scan without rescanning every pair after each merge: the symbols as a linked
    // list, every adjacent pair with a rank in a heap keyed (rank, position).  Popping the lowest rank and then
    // the lowest position is the order merge_scan picks, and a symbol's position only changes when its left
    // neighbour merges - so a popped pair whose rank no longer matches what is at that position is stale.
    // (#268: the scan is O(n^2) and took seconds on an 8K-character CJK run.)
    const std::size_t n = syms.size();
    std::vector<int> parts = syms;
    std::vector<std::ptrdiff_t> nxt(n), prv(n);
    for (std::size_t k = 0; k < n; ++k) {
        nxt[k] = static_cast<std::ptrdiff_t>(k) + 1;
        prv[k] = static_cast<std::ptrdiff_t>(k) - 1;
    }
    nxt[n - 1] = -1;
    std::priority_queue<std::pair<int, std::size_t>, std::vector<std::pair<int, std::size_t>>,
                        std::greater<std::pair<int, std::size_t>>> heap;
    for (std::size_t k = 0; k + 1 < n; ++k) {
        const auto it = rank_.find(pair_key(parts[k], parts[k + 1]));
        if (it != rank_.end()) {
            heap.emplace(it->second, k);
        }
    }
    while (!heap.empty()) {
        const auto top = heap.top();
        heap.pop();
        const std::size_t i = top.second;
        if (parts[i] < 0) {
            continue;                                   // i was merged into its left neighbour: a dead node
        }
        const std::ptrdiff_t j = nxt[i];
        if (j < 0) {
            continue;
        }
        const auto here = rank_.find(pair_key(parts[i], parts[j]));
        if (here == rank_.end() || here->second != top.first) {
            continue;                                   // stale: one of the two symbols has merged since
        }
        parts[i] = merged(static_cast<std::size_t>(top.first));
        parts[j] = -1;                                  // dead: a stale entry for j must not merge it again
        const std::ptrdiff_t k = nxt[j];
        nxt[i] = k;
        if (k >= 0) {
            prv[k] = static_cast<std::ptrdiff_t>(i);
            const auto next = rank_.find(pair_key(parts[i], parts[k]));
            if (next != rank_.end()) {
                heap.emplace(next->second, i);
            }
        }
        const std::ptrdiff_t p = prv[i];
        if (p >= 0) {
            const auto prev = rank_.find(pair_key(parts[p], parts[i]));
            if (prev != rank_.end()) {
                heap.emplace(prev->second, static_cast<std::size_t>(p));
            }
        }
    }
    std::vector<int> out;
    out.reserve(n);
    for (std::ptrdiff_t i = 0; i >= 0; i = nxt[i]) {
        out.push_back(parts[i]);
    }
    return out;
}

int Tokenizer::merged(std::size_t rank) const {
    const int target = merge_to_[rank];
    if (target < 0) {
        throw std::runtime_error("BPE produced a token outside the vocabulary");
    }
    return target;
}

std::vector<int> Tokenizer::encode_plain(std::string_view text) const {
    // A piece's ids depend on the piece alone, so repeated pieces (most of a resent conversation) are looked up
    // instead of merged again, as the reference's _piece_ids does.  Per thread: the handle is shared by the
    // server's request threads, and a thread-local cache needs no lock.
    struct PieceCache {
        std::uint64_t owner = 0;
        std::unordered_map<std::string, std::vector<int>> ids;
    };
    thread_local PieceCache cache;
    if (cache.owner != uid_) {
        cache.owner = uid_;
        cache.ids.clear();
    }
    std::vector<int> out;
    out.reserve(text.size() / 3 + 1);
    for (const std::string_view piece : pretokenize(text)) {
        const auto hit = cache.ids.find(std::string(piece));
        if (hit != cache.ids.end()) {
            out.insert(out.end(), hit->second.begin(), hit->second.end());
            continue;
        }
        std::vector<int> syms;
        syms.reserve(piece.size());
        for (const char raw : piece) {
            const int id = byte_id_[static_cast<unsigned char>(raw)];
            if (id < 0) {
                throw std::runtime_error("a byte is not in the vocabulary");
            }
            syms.push_back(id);
        }
        std::vector<int> done = syms.size() > kScanMax ? merge_heap(syms) : merge_scan(std::move(syms));
        out.insert(out.end(), done.begin(), done.end());
        if (cache.ids.size() < kPieceCacheMax) {
            cache.ids.emplace(std::string(piece), std::move(done));
        }
    }
    return out;
}

std::vector<int> Tokenizer::encode(std::string_view text, bool parse_special,
                                  const std::vector<std::pair<std::size_t, std::size_t>>& plain) const {
    if (specials_.empty()) {
        return encode_plain(text);
    }
    const auto in_plain = [&plain](std::size_t at) {
        for (const auto& span : plain) {
            if (at >= span.first && at < span.second) {
                return true;
            }
        }
        return false;
    };
    std::vector<int> out;
    std::size_t pos = 0, i = 0;
    while (i < text.size()) {
        if (!special_first_[static_cast<unsigned char>(text[i])]) {
            ++i;                                        // no literal starts with this byte
            continue;
        }
        const Special* hit = nullptr;
        for (const Special& s : specials_) {
            if (s.type == 3 && !parse_special) {
                continue;
            }
            if (text.size() - i >= s.text.size() && text.compare(i, s.text.size(), s.text) == 0) {
                hit = &s;
                break;
            }
        }
        if (hit != nullptr && !in_plain(i)) {
            if (i > pos) {
                const std::vector<int> head = encode_plain(text.substr(pos, i - pos));
                out.insert(out.end(), head.begin(), head.end());
            }
            out.push_back(hit->id);
            i += hit->text.size();
            pos = i;
            continue;
        }
        if (hit != nullptr) {
            // The reference's finditer finds this literal, skips it (it starts inside a plain span) and resumes
            // AFTER it, so a literal that begins inside the skipped one is never seen.  `pos` stays: the skipped
            // text is still ordinary text.
            i += hit->text.size();
            continue;
        }
        ++i;
    }
    if (pos < text.size()) {
        const std::vector<int> tail = encode_plain(text.substr(pos));
        out.insert(out.end(), tail.begin(), tail.end());
    }
    return out;
}

std::string Tokenizer::token_bytes(int id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= tokens_.size()) {
        throw std::out_of_range("token id " + std::to_string(id) + " is outside the vocabulary (" +
                                std::to_string(tokens_.size()) + ")");
    }
    static const std::unordered_map<std::uint32_t, unsigned char> reverse = [] {
        std::unordered_map<std::uint32_t, unsigned char> m;
        for (std::size_t b = 0; b < 256; ++b) {
            m[alphabet()[b]] = static_cast<unsigned char>(b);
        }
        return m;
    }();
    std::string out;
    const utf8::Cps cps = utf8::decode(tokens_[static_cast<std::size_t>(id)]);
    for (const std::uint32_t cp : cps.cp) {
        const auto it = reverse.find(cp);
        if (it == reverse.end()) {
            throw std::runtime_error("a token contains a character outside the byte alphabet");
        }
        out.push_back(static_cast<char>(it->second));
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string bytes;
    for (const int id : ids) {
        bytes += token_bytes(id);
    }
    return utf8::decode_replacing(bytes);
}

}  // namespace strata::tokenizer
