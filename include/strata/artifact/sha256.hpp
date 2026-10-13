// include/strata/artifact/sha256.hpp - SHA-256 (FIPS 180-4), header-only.
//
// The pack's conversions.json records the sha256 of each converted tensor's source bytes, and the reference packer
// hashes with Python's hashlib.  A C++ packer has to agree with it byte for byte, so the hash has to be the real
// thing, not a checksum: tests/artifact/sha256_test.cpp checks the published vectors (FIPS 180-4 / NIST), including
// the two padding boundaries - a message that ends exactly on a block and one that needs a second block for the
// length alone.
//
// It lives in the artifact layer because the packer is the only user: nothing else in the tree hashes anything.
#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <string>

namespace strata {

class Sha256 {
public:
    Sha256() { reset(); }

    void reset() {
        static const uint32_t init[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                                         0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
        std::memcpy(h_, init, sizeof(h_));
        len_ = 0;
        fill_ = 0;
    }

    void update(const void* data, std::size_t n) {
        const uint8_t* p = (const uint8_t*)data;
        len_ += n;
        while (n) {
            const std::size_t take = n < (64 - fill_) ? n : (64 - fill_);
            std::memcpy(buf_ + fill_, p, take);
            fill_ += take;
            p += take;
            n -= take;
            if (fill_ == 64) {
                block(buf_);
                fill_ = 0;
            }
        }
    }

    /// The 32 bytes of the digest.  Final: the object must be reset to hash anything else.
    std::array<uint8_t, 32> finish() {
        const uint64_t bits = len_ * 8;
        const uint8_t pad = 0x80;
        update(&pad, 1);
        const uint8_t zero = 0;
        while (fill_ != 56) update(&zero, 1);
        uint8_t tail[8];
        for (int i = 0; i < 8; ++i) tail[i] = (uint8_t)(bits >> (56 - 8 * i));
        update(tail, 8);
        std::array<uint8_t, 32> out{};
        for (int i = 0; i < 8; ++i) {
            out[4 * i] = (uint8_t)(h_[i] >> 24);
            out[4 * i + 1] = (uint8_t)(h_[i] >> 16);
            out[4 * i + 2] = (uint8_t)(h_[i] >> 8);
            out[4 * i + 3] = (uint8_t)h_[i];
        }
        return out;
    }

    /// The digest of one buffer, as lowercase hex.
    static std::string hex(const void* data, std::size_t n) {
        Sha256 h;
        h.update(data, n);
        const std::array<uint8_t, 32> d = h.finish();
        static const char* digits = "0123456789abcdef";
        std::string s(64, '0');
        for (int i = 0; i < 32; ++i) {
            s[2 * i] = digits[d[i] >> 4];
            s[2 * i + 1] = digits[d[i] & 0xF];
        }
        return s;
    }

private:
    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

    void block(const uint8_t* p) {
        static const uint32_t K[64] = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
            0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
            0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
            0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
            0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = ((uint32_t)p[4 * i] << 24) | ((uint32_t)p[4 * i + 1] << 16) | ((uint32_t)p[4 * i + 2] << 8) |
                   (uint32_t)p[4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = h + S1 + ch + K[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + maj;
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h_[0] += a;
        h_[1] += b;
        h_[2] += c;
        h_[3] += d;
        h_[4] += e;
        h_[5] += f;
        h_[6] += g;
        h_[7] += h;
    }

    uint32_t h_[8];
    uint64_t len_ = 0;
    std::size_t fill_ = 0;
    uint8_t buf_[64];
};

} // namespace strata
