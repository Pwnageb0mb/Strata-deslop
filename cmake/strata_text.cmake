# cmake/strata_text.cmake - libstrata_text: the tokenizer as a shared library with a C ABI, loaded by serve/native_text.py.
#
# CPU only: no CUDA, no HIP, no llama.cpp.  Only the tokenizer sources, so it builds on any PC with a C++17
# compiler.  Included from the top-level CMakeLists.txt unconditionally (not behind STRATA_BUILD_TESTS).
#   cmake --build <dir> --target strata_text
get_filename_component(_strata_text_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
add_library(strata_text SHARED
  ${_strata_text_root}/src/text/strata_text_capi.cpp
  ${_strata_text_root}/src/tokenizer/tokenizer.cpp
  ${_strata_text_root}/src/tokenizer/pretokenizer.cpp
  ${_strata_text_root}/src/tokenizer/utf8.cpp
  # output side of a generation: detokenizer, stop strings, tool-call parser, loop check (strata_stream.h)
  ${_strata_text_root}/src/text/stream.cpp
  ${_strata_text_root}/src/text/output_parser.cpp
  ${_strata_text_root}/src/text/pytext.cpp
  # chat templates: llama.cpp's Jinja engine (third_party/llama_jinja) behind strata_template.h
  ${_strata_text_root}/src/text/template_capi.cpp
  ${_strata_text_root}/third_party/llama_jinja/jinja/lexer.cpp
  ${_strata_text_root}/third_party/llama_jinja/jinja/parser.cpp
  ${_strata_text_root}/third_party/llama_jinja/jinja/runtime.cpp
  ${_strata_text_root}/third_party/llama_jinja/jinja/string.cpp
  ${_strata_text_root}/third_party/llama_jinja/jinja/value.cpp
  ${_strata_text_root}/third_party/llama_jinja/shim/unicode.cpp)
target_include_directories(strata_text PRIVATE ${_strata_text_root}/include ${_strata_text_root}/third_party
  ${_strata_text_root}/third_party/llama_jinja ${_strata_text_root}/third_party/llama_jinja/shim)
target_compile_features(strata_text PRIVATE cxx_std_17)
set_target_properties(strata_text PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  WINDOWS_EXPORT_ALL_SYMBOLS OFF)
if(MSVC)
  target_compile_options(strata_text PRIVATE /O2 /utf-8)
else()
  target_compile_options(strata_text PRIVATE -O2)
endif()
