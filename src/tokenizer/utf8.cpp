// src/tokenizer/utf8.cpp
#include "strata/tokenizer/utf8.hpp"

namespace strata::tokenizer::utf8 {
namespace {

// The length of the sequence starting with `b0`, from its leading bits.  A continuation byte or a byte that
// cannot start a sequence is 1, so an invalid byte stands alone instead of swallowing its neighbours.
std::size_t lead_len(unsigned char b0) {
    if (b0 >= 0xF0 && b0 <= 0xF7) {
        return 4;
    }
    if (b0 >= 0xE0 && b0 <= 0xEF) {
        return 3;
    }
    if (b0 >= 0xC2 && b0 <= 0xDF) {
        return 2;                                  // C0 and C1 are overlong: not a lead byte
    }
    return 1;
}

}  // namespace

Cps decode(std::string_view text) {
    Cps c;
    std::size_t i = 0;
    while (i < text.size()) {
        const auto b0 = static_cast<unsigned char>(text[i]);
        std::size_t len = lead_len(b0);
        if (i + len > text.size()) {
            len = 1;
        }
        std::uint32_t v = 0;
        if (len == 1) {
            v = b0;
        } else {
            v = b0 & ((1u << (7 - len)) - 1u);
            for (std::size_t k = 1; k < len; ++k) {
                const auto bk = static_cast<unsigned char>(text[i + k]);
                if ((bk & 0xC0u) != 0x80u) {       // not a continuation byte: the byte stands alone
                    len = 1;
                    v = b0;
                    break;
                }
                v = (v << 6) | (bk & 0x3Fu);
            }
        }
        c.cp.push_back(v);
        c.off.push_back(i);
        i += len;
    }
    c.off.push_back(text.size());
    return c;
}

void append(std::string& out, std::uint32_t cp) {
    if (cp > 0x10FFFFu || (cp >= 0xD800u && cp <= 0xDFFFu)) {
        cp = 0xFFFDu;
    }
    if (cp < 0x80u) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800u) {
        out.push_back(static_cast<char>(0xC0u | (cp >> 6)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else if (cp < 0x10000u) {
        out.push_back(static_cast<char>(0xE0u | (cp >> 12)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else {
        out.push_back(static_cast<char>(0xF0u | (cp >> 18)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    }
}

std::string decode_replacing(std::string_view bytes) {
    std::string out;
    std::size_t i = 0;
    while (i < bytes.size()) {
        const auto b0 = static_cast<unsigned char>(bytes[i]);
        std::size_t len = lead_len(b0);
        bool ok = i + len <= bytes.size();
        if (ok && len > 1) {
            for (std::size_t k = 1; k < len; ++k) {
                if ((static_cast<unsigned char>(bytes[i + k]) & 0xC0u) != 0x80u) {
                    ok = false;
                    break;
                }
            }
        }
        if (!ok) {
            append(out, 0xFFFDu);
            ++i;
            continue;
        }
        if (len == 1) {
            out.push_back(bytes[i]);
        } else {
            out.append(bytes.substr(i, len));
        }
        i += len;
    }
    return out;
}

bool is_whole_utf8(std::string_view bytes) {
    std::size_t i = 0;
    while (i < bytes.size()) {
        const unsigned char c = (unsigned char) bytes[i];
        std::size_t extra = 0;
        std::uint32_t cp = 0;
        if (c < 0x80) {
            ++i;
            continue;
        } else if ((c & 0xE0) == 0xC0) {
            extra = 1;
            cp = c & 0x1Fu;
        } else if ((c & 0xF0) == 0xE0) {
            extra = 2;
            cp = c & 0x0Fu;
        } else if ((c & 0xF8) == 0xF0) {
            extra = 3;
            cp = c & 0x07u;
        } else {
            return false;   // a continuation byte here, or 0xF8..0xFF, which is never a lead byte
        }
        if (i + extra >= bytes.size()) return false;   // truncated: the continuation bytes are not all there
        for (std::size_t k = 1; k <= extra; ++k) {
            const unsigned char cc = (unsigned char) bytes[i + k];
            if ((cc & 0xC0) != 0x80) return false;
            cp = (cp << 6) | (cc & 0x3Fu);
        }
        // what Python's utf-8 codec refuses: an overlong form, a surrogate, or a value past U+10FFFF
        if ((extra == 1 && cp < 0x80u) || (extra == 2 && cp < 0x800u) || (extra == 3 && cp < 0x10000u)) return false;
        if (cp >= 0xD800u && cp <= 0xDFFFu) return false;
        if (cp > 0x10FFFFu) return false;
        i += extra + 1;
    }
    return true;
}

}  // namespace strata::tokenizer::utf8
