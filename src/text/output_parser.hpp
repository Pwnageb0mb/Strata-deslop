// src/text/output_parser.hpp - the model's output text -> reasoning / content / tool-call events (C++ port of
// serve/frontend.py's OutputParser; that class stays the reference and the test oracle).  Internal to libstrata_text.
#pragma once

#include "pytext.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace strata::text {

enum EvKind : uint8_t { kEvReasoning = 1, kEvContent = 2, kEvToolCall = 3, kEvToolStart = 4, kEvToolArgs = 5 };

struct Event {
    EvKind kind;
    std::string text;      // reasoning / content text, or a tool_args piece
    int cid = -1;          // the call a tool_start / tool_args / tool_call belongs to (-1: a call with no start)
    bool same = false;     // tool_call: the very object announced by tool_start (name as streamed, arguments {})
    std::string name;      // tool_start / tool_call
    std::string args;      // tool_call: the arguments as compact JSON
};

struct Call {
    std::string name;
    JVal args;  // an Obj
};

struct ToolInfo {
    std::unordered_set<std::string> string_params;  // the parameters the tool's schema declares as type "string"
};

enum class PState : int32_t { Reasoning = 0, Content = 1, Call = 2, Json = 3, RCall = 4 };

class OutputParser {
public:
    OutputParser(bool thinking, bool stream_tools, bool recover, bool any_tools,
                 std::unordered_map<std::string, ToolInfo> tools);

    void feed(std::string_view delta, std::vector<Event>& out);
    void finish(bool natural_stop, std::vector<Event>& out);

    PState state = PState::Reasoning;
    std::string buf;
    int rescued = 0, refused = 0;
    size_t pending_count() const { return pending.size(); }

private:
    struct Pending {
        std::string raw;
        std::optional<Call> call;
    };
    enum class SS { Name, Between, Str, Raw, Done };

    bool lead = false;
    std::unordered_map<std::string, ToolInfo> schemas;
    bool has_schemas;
    bool stream_tools, recover;
    std::vector<Pending> pending;
    std::string fence, line;
    int64_t ticks = 0;
    bool bare = false, batch = false;
    // the streaming scanner
    size_t sp = 0;
    SS ss = SS::Name;
    int scall = -1;
    std::string scall_name;
    bool sfirst = true, sval_started = false;
    const ToolInfo* sdeclared = nullptr;
    int next_cid = 0;

    void track(std::string_view text);
    bool ok_at(size_t p);
    bool opener_ok() const;
    bool in_code() const;
    std::optional<bool> quoted_think_end(size_t i) const;
    void release(bool deliver, std::vector<Event>& out);
    void reset_scan();
    void scan(std::string_view b, std::vector<Event>& out);
    void close_scan(std::string_view b, std::vector<Event>& out);
    enum class Follow { Call, Param, Json, Wait, Text };
    Follow follower(std::string_view after) const;
    std::pair<long, bool> bare_opener();
    void finish_call(std::string_view body, std::vector<Event>& out);
    Event text_event(EvKind k, std::string_view t);
    const ToolInfo* schema_of(std::string_view name) const;
};

}  // namespace strata::text
