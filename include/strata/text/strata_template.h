/* include/strata/text/strata_template.h - the chat-template renderer of libstrata_text (C ABI).
 *
 * Renders a Jinja chat template (the pack's chat_template.jinja) with llama.cpp's Jinja engine (third_party/llama_jinja),
 * set up like serve/frontend.py ChatTemplate: trim_blocks and lstrip_blocks on, the globals raise_exception and the
 * tojson filter (ensure_ascii False by default).  Same rules as strata_text.h: no exception crosses this boundary, a
 * failing call returns NULL or a negative value and strata_template_error() says why (per thread).  A handle is
 * safe to render from many threads (renders on one handle are serialised).
 */
#ifndef STRATA_TEMPLATE_H
#define STRATA_TEMPLATE_H

#include <stddef.h>
#include <stdint.h>

#include "strata/text/strata_text.h"   /* STRATA_TEXT_API */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct strata_template strata_template;

/* ABI version of this header; bumped when a signature changes. */
STRATA_TEXT_API int strata_template_abi_version(void);

/* The message of the last failure on the calling thread ("" if none).  Valid until the next call on this thread. */
STRATA_TEXT_API const char* strata_template_error(void);

/* What the last failure on this thread was: 0 none, 1 the template called raise_exception (the message is the
 * template's own text, no prefix: a client error), 2 anything else (syntax error, bad input JSON, a runtime error
 * in the template). */
STRATA_TEXT_API int strata_template_error_kind(void);

/* Parse the template source utf8[0..len).  NULL on failure (see strata_template_error). */
STRATA_TEXT_API strata_template* strata_template_load(const char* utf8, int64_t len);
STRATA_TEXT_API void strata_template_free(strata_template* h);

/* Render.  messages_json: a JSON array (required); tools_json: a JSON value, or NULL for "none" (the template sees
 * tools = none); kwargs_json: a JSON object of extra template variables (enable_thinking, ...) or NULL.  The
 * template variables are messages, tools, add_generation_prompt, then the kwargs.  Returns the size in bytes of the
 * rendered UTF-8 text, -1 on failure.  The text is kept per thread until the next render on this thread: fetch it with
 * strata_template_result. */
STRATA_TEXT_API int64_t strata_template_render(const strata_template* h, const char* messages_json,
                                               const char* tools_json, const char* kwargs_json,
                                               int add_generation_prompt);

/* Copy the last rendered text of this thread (no terminating NUL) into buf, at most cap bytes; returns its size. */
STRATA_TEXT_API int64_t strata_template_result(char* buf, int64_t cap);

#ifdef __cplusplus
}
#endif
#endif
