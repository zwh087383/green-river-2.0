#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

struct PresetFrame {
    std::array<double, 7> left{}, right{}; // radians at the SDK boundary
    bool home = false;
    double scale = .5;
    int wait_ms = 0;
};

struct MotionPreset {
    std::string id;
    int builtin = -1;
    double timeout = 30;
    std::vector<PresetFrame> frames;
};

inline bool preset_id_valid(const std::string& id) {
    return !id.empty() && id.size() <= 40 && id[0] >= 'a' && id[0] <= 'z'
        && std::all_of(id.begin(), id.end(), [](char c) {
            return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
        });
}

inline double preset_number(const nlohmann::json& value, double low, double high) {
    if (!value.is_number()) throw std::runtime_error("Preset number required");
    double result = value.get<double>();
    if (!std::isfinite(result) || result < low || result > high)
        throw std::runtime_error("Preset number out of range");
    return result;
}

inline MotionPreset read_motion_preset(const std::string& id) {
    const char* configured = std::getenv("GREENRIVER_MOTION_PRESETS");
    if (!configured || !preset_id_valid(id)) throw std::runtime_error("Preset catalog unavailable");
    if (std::filesystem::file_size(configured) > 262144) throw std::runtime_error("Preset catalog too large");
    std::ifstream input(configured);
    auto data = nlohmann::json::parse(input);
    if (data.at("version") != 1 || !data.at("presets").is_array()
            || data.at("presets").empty() || data.at("presets").size() > 64)
        throw std::runtime_error("Invalid preset catalog");
    std::set<std::string> seen;
    std::optional<MotionPreset> selected;
    for (const auto& item : data.at("presets")) {
        auto key = item.at("id").get<std::string>();
        if (!preset_id_valid(key) || !seen.insert(key).second) throw std::runtime_error("Invalid preset id");
        if (key != id) continue;
        MotionPreset preset;
        preset.id = key;
        preset.timeout = preset_number(item.value("timeout_s", nlohmann::json(30)), 5, 120);
        auto type = item.at("type").get<std::string>();
        if (type == "builtin") {
            if (!item.at("sdk_code").is_number_integer()) throw std::runtime_error("Invalid SDK preset");
            preset.builtin = item.at("sdk_code").get<int>();
            if (preset.builtin < 0 || preset.builtin > 6) throw std::runtime_error("Invalid SDK preset");
        } else if (type == "sequence") {
            auto frames = item.at("frames");
            if (!frames.is_array() || frames.size() < 2 || frames.size() > 32)
                throw std::runtime_error("Invalid preset frames");
            for (const auto& item_frame : frames) {
                PresetFrame frame;
                frame.scale = preset_number(item_frame.at("scale"), .05, 1);
                auto wait = item_frame.value("wait_ms", nlohmann::json(0));
                if (!wait.is_number_integer()) throw std::runtime_error("Invalid frame wait");
                frame.wait_ms = wait.get<int>();
                if (frame.wait_ms < 0 || frame.wait_ms > 5000) throw std::runtime_error("Invalid frame wait");
                frame.home = item_frame.value("home", false);
                if (frame.home) {
                    if (item_frame.contains("left_arm") || item_frame.contains("right_arm"))
                        throw std::runtime_error("Mixed home and joint frame");
                } else {
                    for (int side = 0; side < 2; ++side) {
                        auto angles = item_frame.at(side == 0 ? "left_arm" : "right_arm");
                        if (!angles.is_array() || angles.size() != 7) throw std::runtime_error("Expected seven joint angles");
                        auto& target = side == 0 ? frame.left : frame.right;
                        for (int joint = 0; joint < 7; ++joint)
                            target[joint] = preset_number(angles[joint], -180, 180) * std::acos(-1) / 180;
                    }
                }
                preset.frames.push_back(frame);
            }
            if (!preset.frames.back().home) throw std::runtime_error("Preset must return home");
        } else throw std::runtime_error("Invalid preset type");
        selected = preset;
    }
    if (!selected) throw std::runtime_error("Unknown preset");
    return *selected;
}
