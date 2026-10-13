// Strata shim: llama.cpp's common/json.h wraps nlohmann::json as common_json; the Jinja engine only needs it to
// read request JSON (value.cpp from_json / global_from_json).  Object keys keep insertion order, as in Python.
#pragma once
#include "nlohmann/json.hpp"
using common_json = nlohmann::ordered_json;
