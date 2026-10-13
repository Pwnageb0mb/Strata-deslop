/* include/strata/text/strata_request.h - the request translation of libstrata_text (C ABI).
 *
 * serve/frontend.py openai_to_messages() and anthropic_to_messages() in C++: a chat / messages request as JSON in,
 * {"messages": [...], "tools": [...] | null, "kwargs": {...}} as JSON out, key order and values as the Python gives them
 * (json.dumps of the Python result).  Same rules as strata_text.h: no exception crosses this boundary, a failing call
 * returns -1 and strata_request_error() / strata_request_error_kind() say why (per thread).
 *
 * The one thing left to the caller is the stable tool order (frontend._stable_tool_order), which keeps process-wide
 * state shared with the Python path: the wrapper applies it to the returned tools.
 */
#ifndef STRATA_REQUEST_H
#define STRATA_REQUEST_H

#include <stddef.h>
#include <stdint.h>

#include "strata/text/strata_text.h"   /* STRATA_TEXT_API */

#ifdef __cplusplus
extern "C" {
#endif

/* ABI version of this header; bumped when a signature changes. */
STRATA_TEXT_API int strata_request_abi_version(void);

/* The message of the last failure on the calling thread ("" if none).  Valid until the next call on this thread. */
STRATA_TEXT_API const char* strata_request_error(void);

/* What the last failure on this thread was: 0 none; 1 the reference raises ValueError with that message (a client
 * error: the caller raises it); 2 the request has a shape this port does not take over (the reference would raise
 * TypeError / AttributeError / KeyError, or reads it in a way that is not ported): the caller runs the Python
 * reference; 3 the input is not valid JSON. */
STRATA_TEXT_API int strata_request_error_kind(void);

/* Convert.  api: 0 = OpenAI chat completions, 1 = Anthropic messages (think_unasked as the Python argument; ignored
 * for 0).  req_json: the request object as JSON.  Returns the size in bytes of the result JSON (ASCII only), -1 on
 * failure.  The text is kept per thread until the next call on this thread: fetch it with strata_request_result. */
STRATA_TEXT_API int64_t strata_request_convert(int api, const char* req_json, int64_t len, int think_unasked);

/* Copy the last result of this thread (no terminating NUL) into buf, at most cap bytes; returns its size. */
STRATA_TEXT_API int64_t strata_request_result(char* buf, int64_t cap);

#ifdef __cplusplus
}
#endif
#endif
