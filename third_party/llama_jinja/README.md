# llama_jinja

The Jinja engine of llama.cpp (`common/jinja/`: lexer, parser, runtime, value, string, utils), MIT licence
(`LICENSE`, "Copyright (c) 2023-2026 The ggml authors"), copied from llama.cpp commit
9723942adc518b43c4b95dc4dce6906903eb5e09, plus `common/unicode.{h,cpp}` (in `shim/`).
`caps.cpp` (llama.cpp's template probing) is not used.  `shim/json.h` replaces llama.cpp's `common/json.h`
with an alias of `nlohmann::ordered_json` (see `third_party/nlohmann/json.hpp`, MIT, Niels Lohmann).
Used by `src/text/template_capi.cpp` (libstrata_text).

Local changes (all in `jinja/`; the output is unchanged: checked against the unmodified runtime on the deployed
template and 34 small templates, and against jinja2 by `tools/test_native_template.py`).  They take a render of a
201-message, 104K-character conversation from 24.6 ms to about 2.5 ms in C++ (3.1 ms per `ChatTemplate.render` call,
jinja2: 5.1 ms; measured by `python -m unittest tools.test_native_template` with `STRATA_TEXT_LIB` set).  Found with a
sampling profiler and an allocation counter on that conversation: the cost was copying and allocating, not the
template logic (64,000 allocations per render before, 19,000 after).

1. `runtime.cpp`: `global_builtins()` and `get_builtins()` are bound by `const auto &` instead of copying the whole
   function table on every identifier and method lookup.
2. `runtime.h` `context`: a new scope (for loop, macro call) points to its parent and looks variables up through the
   chain; it no longer copies every variable of the parent (that was about half the time). Variables are a plain
   `unordered_map<string, value>`; a lookup no longer makes an undefined value first. `context::flat_copy` is the old
   snapshot copy, used by `{% call %}` whose scope is kept in a closure.
3. `value.h` `value_object_t`: `insert` and `at` do one hash lookup instead of two or three; `insert_new` for keys known
   to be new.
4. `value.h` `is_val` / `cast_val`: an exact `typeid` comparison for the concrete value types (a tuple still counts as an
   array) instead of `dynamic_cast`.
5. `value.h` `value_bool_t` / `value_float_t` no longer allocate an inner integer value (only used for the hash).
6. `runtime.cpp` `for_statement`: the `loop` object is updated in place for the next iteration when the body kept no
   reference to it (otherwise a new one is made, as before); loop keys are shared values.
7. `runtime.cpp` / `runtime.h` output gathering: `if`, `for` and macro bodies append each statement's text into one
   string, moving the parts of values nobody else holds (`gather_string_parts_take`), instead of building an array and
   copying it at every nesting level; `gather_string_parts_recursive` no longer copies the array it walks.
   `string::append` reserves geometrically and has an rvalue overload.
8. `runtime.cpp` `+` / `~` on strings appends to a left side nobody else holds instead of copying it; a string or
   integer literal is made once and shared; `obj.name` keeps its key value; a missing property makes the undefined
   value only when it is returned; built-in filters are called without making a bound function value;
   `get_kwarg_or_pos` makes its default only when needed.

Not from llama.cpp: JSON text is turned into values by `src/text/template_capi.cpp` itself (a strict reader, same values
as nlohmann plus `from_json`, differential-tested against it), not by this directory.
