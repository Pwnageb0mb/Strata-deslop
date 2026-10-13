// tests/artifact/sha256_test.cpp - the published vectors, and the two boundaries padding can hide a bug at.
//
// conversions.json's source_sha256 has to agree with Python's hashlib, so the hash has to be SHA-256, not a
// checksum that merely looks like one.  The vectors below are the ones in FIPS 180-4 and its errata (NIST's own
// examples): the empty message, "abc", the 56- and 112-byte ones, and a million 'a's.
//
// The two cases that matter beyond the vectors are the padding boundaries: a message that ends exactly on a block
// (64 bytes, where 0x80 alone would leave no room for the length) and one of 56 bytes (where the length does not
// fit in the block and a second block is needed for it alone).  Both are in the table.  The last case feeds the
// same message in uneven pieces, which is how the packer hashes a tensor: one buffer, but a streamed hash that
// drops a byte at a chunk boundary would still pass every whole-buffer vector.
#include "strata/artifact/sha256.hpp"

#include <cstdio>
#include <string>

namespace {

int failures = 0;

void check(const std::string& what, const std::string& got, const std::string& want) {
    if (got != want) {
        std::fprintf(stderr, "FAIL %s\n  got  %s\n  want %s\n", what.c_str(), got.c_str(), want.c_str());
        ++failures;
    }
}

} // namespace

int main() {
    // FIPS 180-4 / NIST examples.
    check("empty", strata::Sha256::hex("", 0),
          "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    check("\"abc\"", strata::Sha256::hex("abc", 3),
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    const char* m448 = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    check("448-bit message", strata::Sha256::hex(m448, 56),
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    const char* m896 = "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmn"
                       "hijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu";
    check("896-bit message", strata::Sha256::hex(m896, 112),
          "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1");

    // The padding boundaries: 55 bytes fits the length in one block, 56 does not, 64 is a whole block.
    const std::string a55(55, 'a'), a56(56, 'a'), a64(64, 'a');
    check("55 bytes", strata::Sha256::hex(a55.data(), a55.size()),
          "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318");
    check("56 bytes", strata::Sha256::hex(a56.data(), a56.size()),
          "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a");
    check("64 bytes", strata::Sha256::hex(a64.data(), a64.size()),
          "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb");

    // A million 'a's, streamed in pieces that straddle the block boundary.
    strata::Sha256 big;
    const std::string chunk(997, 'a'); // prime, so the block boundary moves on every call
    std::size_t left = 1000000;
    while (left) {
        const std::size_t take = left < chunk.size() ? left : chunk.size();
        big.update(chunk.data(), take);
        left -= take;
    }
    const std::array<uint8_t, 32> d = big.finish();
    static const char* digits = "0123456789abcdef";
    std::string hex;
    for (uint8_t b : d) {
        hex += digits[b >> 4];
        hex += digits[b & 0xF];
    }
    check("1,000,000 'a' streamed", hex, "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");

    if (failures) {
        std::fprintf(stderr, "sha256_test: FAILED (%d)\n", failures);
        return 1;
    }
    std::printf("sha256_test: the FIPS 180-4 vectors and both padding boundaries match\n");
    return 0;
}
