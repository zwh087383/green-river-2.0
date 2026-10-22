#pragma once

#include <algorithm>
#include <chrono>
#include <deque>
#include <iomanip>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <stdexcept>
#include <unordered_map>

namespace greenriver {
inline std::string json_quote(const std::string& value) {
    std::string output = "\"";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') { output += '\\'; output += c; }
        else if (c >= 32 && c < 127) output += c;
        else output += '?';
    }
    return output + '"';
}
inline std::string new_instance_id() {
    std::random_device source;
    std::ostringstream output;
    for (int i = 0; i < 4; ++i) output << std::hex << std::setw(8) << std::setfill('0') << source();
    return output.str();
}
struct RequestResult {
    std::string token, purpose, fingerprint, canonical;
    std::string status = "accepted", detail, error_stage;
    bool sdk_returned = false, target_verified = false, process_verified = false, home_verified = false;
    std::optional<int> sdk_code;
    unsigned long observer_frames = 0;
    size_t reached_steps = 0, expected_steps = 0;
    long accepted_ms = 0, started_ms = -1, finished_ms = -1;
    bool terminal() const { return status == "cancelled_home" || status == "verified_completed" || status == "failed" || status == "unknown"; }
};

// Access is serialized by State::lock. No persistence or cross-restart dedupe.
class RequestLedger {
    std::unordered_map<std::string, RequestResult> records;
    std::deque<std::string> order;
    std::chrono::steady_clock::time_point origin = std::chrono::steady_clock::now();
public:
    const std::string instance_id = new_instance_id();
    long now_ms() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - origin).count();
    }
    bool contains(const std::string& token) const { return records.count(token); }
    bool same(const std::string& token, const std::string& fingerprint) const {
        auto it = records.find(token);
        return it != records.end() && it->second.fingerprint == fingerprint;
    }
    void accept(const std::string& token, const std::string& purpose, const std::string& fingerprint,
                const std::string& canonical = "") {
        if (contains(token)) return;
        if (records.size() >= 512) {
            auto candidate = std::find_if(order.begin(), order.end(), [&](const auto& id) {
                const auto& record = records.at(id);
                return !record.canonical.empty() || record.terminal();
            });
            if (candidate == order.end()) throw std::runtime_error("Request ledger capacity exhausted");
            records.erase(*candidate); order.erase(candidate);
        }
        RequestResult record;
        record.token = token; record.purpose = purpose; record.fingerprint = fingerprint;
        record.canonical = canonical; record.accepted_ms = now_ms();
        records.emplace(token, std::move(record)); order.push_back(token);
    }
    RequestResult* get(const std::string& token) {
        auto it = records.find(token);
        return it == records.end() ? nullptr : &it->second;
    }
    std::optional<RequestResult> result(const std::string& token) const {
        auto it = records.find(token);
        if (it == records.end()) return std::nullopt;
        auto result = it->second;
        if (!result.canonical.empty()) {
            auto canonical = records.find(result.canonical);
            if (canonical == records.end()) {
                result.status = "unknown"; result.detail = "Canonical result no longer retained";
            } else {
                auto original = result;
                result = canonical->second;
                result.token = original.token; result.canonical = original.canonical;
                result.accepted_ms = original.accepted_ms;
            }
        }
        return result;
    }
    void running(const std::string& token) {
        if (auto* record = get(token); record && !record->terminal()) {
            record->status = "running"; record->started_ms = now_ms();
        }
    }
    void fail_unfinished(const std::string& detail, const std::string& stage, bool ambiguous) {
        for (auto& [id, record] : records) {
            (void)id;
            if (!record.canonical.empty() || record.terminal()) continue;
            record.status = ambiguous && record.started_ms >= 0 ? "unknown" : "failed";
            record.detail = detail.substr(0, 384); if (record.error_stage.empty()) record.error_stage = stage;
            record.finished_ms = now_ms();
        }
    }
    std::string json(const std::string& token) const {
        auto value = result(token);
        if (!value) return "null";
        const auto& r = *value;
        return "{\"request_id\":" + json_quote(r.token) + ",\"canonical_request_id\":" +
            json_quote(r.canonical.empty() ? r.token : r.canonical) + ",\"purpose\":" + json_quote(r.purpose) +
            ",\"result_state\":" + json_quote(r.status) + ",\"detail\":" + json_quote(r.detail) +
            ",\"error_stage\":" + json_quote(r.error_stage) + ",\"sdk_returned\":" + (r.sdk_returned ? "true" : "false") +
            ",\"sdk_code\":" + (r.sdk_code ? std::to_string(*r.sdk_code) : "null") +
            ",\"target_verified\":" + (r.target_verified ? "true" : "false") +
            ",\"home_verified\":" + (r.home_verified ? "true" : "false") +
            ",\"process_verified\":" + (r.process_verified ? "true" : "false") +
            ",\"observer_frames\":" + std::to_string(r.observer_frames) +
            ",\"reached_steps\":" + std::to_string(r.reached_steps) + ",\"expected_steps\":" + std::to_string(r.expected_steps) +
            ",\"verification_scope\":\"arm_joints_only\",\"accepted_ms\":" + std::to_string(r.accepted_ms) +
            ",\"started_ms\":" + std::to_string(r.started_ms) + ",\"finished_ms\":" + std::to_string(r.finished_ms) +
            ",\"elapsed_ms\":" + std::to_string((r.finished_ms >= 0 ? r.finished_ms : now_ms()) - r.accepted_ms) + "}";
    }
};
} // namespace greenriver
