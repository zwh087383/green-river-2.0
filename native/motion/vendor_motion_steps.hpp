#pragma once
#include "motion_presets.hpp"
#include <cstdint>

struct VendorMotionStep {
    std::optional<std::array<double,7>> left, right;
    std::optional<std::vector<uint16_t>> left_hand, right_hand;
    double scale=.5;
    int wait_ms=0;
};

inline std::vector<VendorMotionStep> read_vendor_motion_steps(int code) {
    static const std::array<std::string,7> keys{"M2","A","B","X","Y","LB","RB"};
    if (code<0 || code>=static_cast<int>(keys.size())) throw std::runtime_error("Invalid vendor motion code");
    const char* configured=std::getenv("GREENRIVER_VENDOR_MOTION_FILE");
    std::string path=configured?configured:"install/product/config/arm_motion.json";
    if (std::filesystem::file_size(path)>262144) throw std::runtime_error("Vendor motion file too large");
    std::ifstream input(path);auto data=nlohmann::json::parse(input);
    const auto& steps=data.at(keys[code]);
    if (!steps.is_array() || steps.empty() || steps.size()>64) throw std::runtime_error("Invalid vendor motion steps");
    std::vector<VendorMotionStep> result;
    for (const auto& raw:steps) {
        VendorMotionStep step;
        step.scale=preset_number(raw.value("scale",nlohmann::json(.5)),.05,1);
        auto wait=raw.value("wait_ms",nlohmann::json(0));
        if (!wait.is_number_integer()) throw std::runtime_error("Invalid vendor wait");
        step.wait_ms=wait.get<int>();
        if (step.wait_ms<0 || step.wait_ms>5000) throw std::runtime_error("Invalid vendor wait");
        for (int side=0;side<2;++side) {
            const char* arm_key=side==0?"left_arm":"right_arm";
            const char* hand_key=side==0?"left_end_effector":"right_end_effector";
            if (raw.contains(arm_key)) {
                const auto& values=raw.at(arm_key);
                if (!values.is_array() || values.size()!=7) throw std::runtime_error("Invalid vendor arm target");
                std::array<double,7> arm{};
                for (int j=0;j<7;++j) arm[j]=preset_number(values[j],-180,180)*std::acos(-1)/180;
                (side==0?step.left:step.right)=arm;
            }
            if (raw.contains(hand_key)) {
                const auto& values=raw.at(hand_key);
                if (!values.is_array() || values.size()!=6) throw std::runtime_error("Invalid vendor hand target");
                std::vector<uint16_t> hand;
                for (const auto& value:values) {
                    if (!value.is_number_integer()) throw std::runtime_error("Invalid vendor finger target");
                    hand.push_back(static_cast<uint16_t>(preset_number(value,0,1000)));
                }
                (side==0?step.left_hand:step.right_hand)=hand;
            }
        }
        if (!step.left && !step.right && !step.left_hand && !step.right_hand && !step.wait_ms)
            throw std::runtime_error("Empty vendor step");
        result.push_back(step);
    }
    return result;
}
