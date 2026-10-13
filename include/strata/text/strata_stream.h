/* include/strata/text/strata_stream.h - libstrata_text, the output side of one generation: incremental UTF-8
 * decoding of token bytes, stop strings, the reasoning / content / tool-call parser, and the reasoning-loop check.
 *
 * C implementations of the detokenizer, stop matcher, repeat-coverage check and
 * OutputParser.  Same
 * conventions as strata_text.h: plain C, no exception crosses it (a failing call returns a negative value or NULL and
 * strata_stream_error() says why, per thread), opaque handles.  A handle belongs to one generation and is not
 * thread-safe; different handles may be used from different threads.
 *
 * Output convention: a call that produces text returns its length in bytes and leaves the bytes in the handle,
 * where the matching *_data(h) reads them; they stay valid until the next call on that handle.  Strings are UTF-8
 * (a lone surrogate travels as its 3-byte form).
 */
#ifndef STRATA_STREAM_H
#define STRATA_STREAM_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  define STRATA_STREAM_API __declspec(dllexport)
#else
#  define STRATA_STREAM_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Bumped when a signature or the event encoding changes. */
STRATA_STREAM_API int strata_stream_abi_version(void);
/* The message of the last failure on the calling thread ("" if none). */
STRATA_STREAM_API const char* strata_stream_error(void);

/* ---- Detokenizer: bytes of successive tokens -> text.  An incomplete character at the end is held; an invalid
 * byte becomes U+FFFD, as bytes.decode("utf-8", "replace") makes it. */
typedef struct strata_detok strata_detok;
STRATA_STREAM_API strata_detok* strata_detok_new(void);
STRATA_STREAM_API void strata_detok_free(strata_detok* h);
/* -> length of the text that is complete now (>= 0). */
STRATA_STREAM_API int64_t strata_detok_push(strata_detok* h, const char* bytes, int64_t len);
STRATA_STREAM_API const char* strata_detok_data(const strata_detok* h);
/* 1 when the first bytes of a character are held. */
STRATA_STREAM_API int strata_detok_pending(const strata_detok* h);

/* ---- StopMatcher.  `blob` holds the n stop strings back to back; string k is blob[offsets[k] .. offsets[k+1]). */
typedef struct strata_stop strata_stop;
STRATA_STREAM_API strata_stop* strata_stop_new(const char* blob, const int64_t* offsets, int64_t n);
STRATA_STREAM_API void strata_stop_free(strata_stop* h);
/* -> length of the text that is safe to send; after a match strata_stop_hit() is the string's index. */
STRATA_STREAM_API int64_t strata_stop_push(strata_stop* h, const char* text, int64_t len);
/* -> length of the held tail (it is also cleared). */
STRATA_STREAM_API int64_t strata_stop_flush(strata_stop* h);
STRATA_STREAM_API const char* strata_stop_data(const strata_stop* h);
/* The index of the stop string that matched, -1 if none yet. */
STRATA_STREAM_API int64_t strata_stop_hit(const strata_stop* h);
/* The held tail (not NUL-terminated): its length, and its bytes. */
STRATA_STREAM_API int64_t strata_stop_held_len(const strata_stop* h);
STRATA_STREAM_API const char* strata_stop_held(const strata_stop* h);

/* ---- OutputParser */
typedef struct strata_parser strata_parser;

/* What the caller reads after every call (the Python parser's attributes). */
typedef struct strata_parser_status {
    int32_t state;             /* 0 reasoning, 1 content, 2 call, 3 json, 4 rcall */
    int32_t pending;           /* calls found in the reasoning that are waiting for what follows them */
    int32_t rescued;
    int32_t refused;
    int32_t buf_is_think_end;  /* buf is exactly "</think>" */
    int32_t reserved;
    int64_t buf_len;           /* bytes held in buf */
} strata_parser_status;

/* tools_json: {"any": bool, "tools": {NAME: [the names of its string-typed parameters], ...}} (the caller reduces the
 * request's tool list to this; "any" is whether the list was non-empty).  NULL / len 0: no tools. */
STRATA_STREAM_API strata_parser* strata_parser_new(int thinking, int stream_tools, int recover, const char* tools_json,
                                                   int64_t len);
STRATA_STREAM_API void strata_parser_free(strata_parser* h);
/* Feed a text delta -> the byte length of the encoded events (>= 0), or -1 when the Python parser would raise
 * ValueError (a tool call that does not parse and recover is off): strata_stream_error() is its message. */
STRATA_STREAM_API int64_t strata_parser_feed(strata_parser* h, const char* text, int64_t len, strata_parser_status* st);
/* End of generation; natural_stop = 1 when the turn ended on a stop (reason None or "stop"). */
STRATA_STREAM_API int64_t strata_parser_finish(strata_parser* h, int natural_stop, strata_parser_status* st);
STRATA_STREAM_API void strata_parser_status_get(const strata_parser* h, strata_parser_status* st);
/* The encoded events of the last feed / finish.  Records, host byte order (little endian):
 *   u8 kind: 1 reasoning, 2 content            then u32 n, n bytes of text
 *            3 tool_call                       then i32 cid, u8 same, u32 n, name, u32 m, arguments (compact JSON)
 *            4 tool_start                      then i32 cid, u32 n, name
 *            5 tool_args                       then i32 cid, u32 n, a piece of the arguments' JSON
 * cid numbers the calls announced by tool_start (0, 1, ...); a tool_call with cid >= 0 finishes that call (it keeps
 * its id), -1 is a call with no tool_start.  same = 1: the object announced by tool_start itself. */
STRATA_STREAM_API const char* strata_parser_data(const strata_parser* h);
/* The held buffer (parser.buf): buf_len bytes, not NUL-terminated. */
STRATA_STREAM_API const char* strata_parser_buf(const strata_parser* h);

/* ---- Pipe: one token -> events in one call.  Runs the detokenizer, the parser and the server's cut() at the stop
 * strings (the three handles above, not owned: they must outlive the pipe, and stay usable by themselves) and
 * returns what the server reads after each token.
 * strata_pipe_step returns a flags word: bits 0-2 the parser state, the STRATA_PIPE_* bits below, and from bit 16 the
 * length n of the events written to the buffer given to strata_pipe_new (or, with STRATA_PIPE_OVERFLOW, left in
 * strata_pipe_data).  With STRATA_PIPE_ONE_REASONING / _CONTENT the buffer holds just the text of the one event
 * (nothing else came out); otherwise it holds records as strata_parser_data describes them.  Negative: -1 the parser
 * raised (strata_stream_error), -2 the token id is outside the vocabulary.  blob / offsets: every token's raw bytes,
 * as strata_text_export_tokens(raw=1) writes them (not copied: they must outlive the pipe).  st (may be NULL) is
 * written after every step, as strata_parser_status_get would. */
#define STRATA_PIPE_BUF 0x8           /* parser.buf is not empty */
#define STRATA_PIPE_PENDING 0x10      /* the parser holds calls found in the reasoning */
#define STRATA_PIPE_DETOK 0x20        /* the detokenizer holds the first bytes of a character */
#define STRATA_PIPE_HIT 0x40          /* a stop string matched */
#define STRATA_PIPE_THINK_END 0x80    /* parser.buf is exactly "</think>" */
#define STRATA_PIPE_THINKING 0x100    /* the state is reasoning or rcall */
#define STRATA_PIPE_OVERFLOW 0x200
#define STRATA_PIPE_ONE_REASONING 0x400
#define STRATA_PIPE_ONE_CONTENT 0x800
typedef struct strata_pipe strata_pipe;
STRATA_STREAM_API strata_pipe* strata_pipe_new(const char* blob, const int64_t* offsets, int64_t vocab, strata_detok* det,
                                               strata_stop* stop, strata_parser* par, strata_parser_status* st,
                                               char* buf, int64_t cap);
STRATA_STREAM_API void strata_pipe_free(strata_pipe* h);
/* track_tail = 0: the token's text is not added to the tail (the server's forced wrap-up tokens). */
STRATA_STREAM_API int64_t strata_pipe_step(strata_pipe* h, int32_t token, int track_tail);
/* The flags word without stepping. */
STRATA_STREAM_API int64_t strata_pipe_status(const strata_pipe* h);
STRATA_STREAM_API const char* strata_pipe_data(const strata_pipe* h);
/* The last bytes (up to 8) of the text the tracked steps produced: enough for its last two characters. */
STRATA_STREAM_API int64_t strata_pipe_tail(const strata_pipe* h);
STRATA_STREAM_API const char* strata_pipe_tail_data(const strata_pipe* h);

/* ---- reasoning_repeat_coverage(text) of serve/server.py, over UTF-8 text.  -1.0 on failure. */
STRATA_STREAM_API double strata_repeat_coverage(const char* text, int64_t len);

#ifdef __cplusplus
}
#endif
#endif
