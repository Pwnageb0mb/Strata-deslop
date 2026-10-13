/* include/strata/text/strata_text.h - the C ABI of libstrata_text: the server's tokenizer.
 *
 * It wraps strata::tokenizer::Tokenizer (the C++ port of tools/strata_tokenizer.py) behind plain C functions so
 * serve/native_text.py can call it through ctypes.  No exception crosses this boundary: a failing call returns
 * NULL or a negative value and strata_text_error() says why (per thread).  A handle is immutable after load, so
 * strata_text_encode may be called on one handle from many threads at once.
 */
#ifndef STRATA_TEXT_H
#define STRATA_TEXT_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  define STRATA_TEXT_API __declspec(dllexport)
#else
#  define STRATA_TEXT_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct strata_text strata_text;

/* ABI version of this header; bumped when a signature changes. */
STRATA_TEXT_API int strata_text_abi_version(void);

/* The message of the last failure on the calling thread ("" if none).  Valid until the next call on this thread. */
STRATA_TEXT_API const char* strata_text_error(void);

/* Load <dir>/vocab.json, merges.txt and token_type.json.  NULL on failure (see strata_text_error). */
STRATA_TEXT_API strata_text* strata_text_load(const char* dir);
STRATA_TEXT_API void strata_text_free(strata_text* h);

STRATA_TEXT_API int64_t strata_text_vocab_size(const strata_text* h);

/* Encode utf8[0..len).  `spans` is n_spans pairs of int64 BYTE offsets [start, end) that stay ordinary text even
 * where they spell a special literal.  Writes up to `cap` ids to `out` and returns the number of ids the text
 * needs; if that is > cap nothing is written and the caller retries with a larger buffer.  An id is never
 * produced for less than one input byte, so cap = len is always enough.  -1 on failure. */
STRATA_TEXT_API int64_t strata_text_encode(const strata_text* h, const char* utf8, int64_t len, int parse_special,
                                           const int64_t* spans, int64_t n_spans, int32_t* out, int64_t cap);

/* Every token at once.  raw=1: each token's raw bytes (byte-level BPE decoded, what token_bytes(i) gives);
 * raw=0: the vocabulary strings as stored (UTF-8 of the byte-alphabet characters).  offsets has vocab_size+1
 * entries; token i is blob[offsets[i] .. offsets[i+1]).  Two-call pattern: pass blob=NULL to get the total size;
 * otherwise the blob is written when blob_cap is large enough.  Returns the blob size, -1 on failure.  A token
 * that is not valid byte-alphabet text (never in a real vocabulary) is empty when raw=1. */
STRATA_TEXT_API int64_t strata_text_export_tokens(const strata_text* h, int raw, char* blob, int64_t blob_cap,
                                                  int64_t* offsets);

/* The type-3 (CONTROL) tokens, as text, separated by '\n'.  Same two-call pattern; returns the size, -1 on failure.
 * Order is the vocabulary order. */
STRATA_TEXT_API int64_t strata_text_control_tokens(const strata_text* h, char* buf, int64_t cap);

#ifdef __cplusplus
}
#endif
#endif
