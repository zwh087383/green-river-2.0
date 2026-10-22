#pragma once
#include "trunk_control.hpp"
#include <functional>
#include <cstdint>

// Read-only access to the SAME vendor bus instance opened by TrunkControl.
extern "C" int harmonic_getStatusWord(uint8_t, uint8_t, uint16_t*, unsigned int);

class TrunkPower {
    wheeled_robot::TrunkControl control;
    std::array<float,6> home_target{};
    int adapter = -1;
    bool want_enabled = false;
    bool stop_latched = false;
    std::chrono::steady_clock::time_point last_connect{};
    std::array<int,6> power_states() {
        if (adapter < 0) throw std::runtime_error("Trunk telemetry not initialized");
        std::array<int,6> result{};
        for (int joint=0; joint<6; ++joint) {
            uint16_t word=0;
            int rc=harmonic_getStatusWord(static_cast<uint8_t>(adapter), static_cast<uint8_t>(joint+1), &word, 100);
            if (rc || !word || word==65535) throw std::runtime_error("Trunk status feedback unavailable");
            if (word & 8) throw std::runtime_error("Trunk motor fault on axis " + std::to_string(joint+1));
            result[joint]=(word & 0x6f)==0x27 ? 0 : 1;
        }
        return result;
    }
    std::array<float,6> positions() {
        std::array<float,6> result{};
        int rc=control.getJointPosition(result);
        if (rc || !std::all_of(result.begin(),result.end(),[](float x){return std::isfinite(x);}))
            throw std::runtime_error("Trunk position telemetry failed");
        return result;
    }
public:
    TrunkPower() {
        const char* configured=std::getenv("GREENRIVER_TRUNK_HOME_TARGETS");
        if (!configured) throw std::runtime_error("Trunk default position not configured");
        std::istringstream input(configured);
        for (float& value:home_target)
            if (!(input>>value) || !std::isfinite(value)) throw std::runtime_error("Invalid trunk home target");
    }
    void enable() {
        if (want_enabled) { healthy(); return; }
        int rc=control.connectDev(0);
        if (rc) throw std::runtime_error("Trunk connectDev failed: "+std::to_string(rc));
        last_connect=std::chrono::steady_clock::now();
        adapter=-1;
        for (int candidate=0;candidate<5;++candidate) {
            uint16_t word=0;
            if (harmonic_getStatusWord(static_cast<uint8_t>(candidate),1,&word,100)==0 && word && word!=65535) {
                if (adapter!=-1) throw std::runtime_error("Ambiguous active trunk adapter");
                adapter=candidate;
            }
        }
        if (adapter<0) throw std::runtime_error("Connected trunk bus not found");
        auto values=power_states();
        if (!std::all_of(values.begin(),values.end(),[](int x){return x==0;}))
            throw std::runtime_error("Trunk six-axis enable not confirmed");
        want_enabled=true;
    }
    bool enabled() const { return want_enabled; }
    void healthy() {
        auto values=power_states();
        if (!std::all_of(values.begin(),values.end(),[&](int x){return x==(want_enabled?0:1);}))
            throw std::runtime_error("Unexpected trunk enable state");
    }
    nlohmann::json status() {
        auto powers=power_states();
        auto actual=positions();
        std::array<uint8_t,6> brakes{};brakes.fill(255);
        int rc=control.getMotorBrakeState(brakes);
        if (rc) brakes.fill(255);
        bool at_home=true;
        for (size_t i=0;i<6;++i) at_home=at_home && std::abs(actual[i]-home_target[i])<=.5f;
        return {{"power",powers},{"positions_deg",actual},{"brakes",brakes},
                {"at_home",at_home},{"default_deg",home_target},{"feedback_age_ms",0},
                {"joint_names",{"脚踝","膝关节","大腿","腰部","转头","俯仰"}}};
    }
    void home(const std::function<void()>& active) {
        if (!want_enabled) return; // Already disabled axes are never re-enabled by power-off.
        bool goal_sent=false;
        try {
            while (std::chrono::steady_clock::now()-last_connect<std::chrono::milliseconds(1600)) {
                active();healthy();std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            active();healthy();
            int rc=control.setJointTargetPosition(home_target,.2f);
            if (rc) throw std::runtime_error("Trunk default position request failed: "+std::to_string(rc));
            goal_sent=true;
            auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(35);
            int stable=0;
            std::array<float,6> previous{};previous.fill(100000);
            while (stable<3) {
                active();healthy();
                auto actual=positions();
                std::array<bool,6> moving{};
                if (control.getMotorMovingState(moving)) throw std::runtime_error("Trunk motion feedback failed");
                bool reached=true;
                for (size_t i=0;i<6;++i)
                    reached=reached && std::abs(actual[i]-home_target[i])<=.5f
                        && std::abs(actual[i]-previous[i])<=.05f && !moving[i];
                stable=reached ? stable+1 : 0;previous=actual;
                if (std::chrono::steady_clock::now()>=deadline) throw std::runtime_error("Trunk home timeout; manual recovery required");
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        } catch (...) {
            if (goal_sent && !stop_latched) { stop_latched=true;control.emergencyStop(); }
            throw;
        }
    }
    void disable() {
        if (!want_enabled) { healthy();return; }
        if (!status().at("at_home").get<bool>()) throw std::runtime_error("Trunk must reach default before disable");
        // Official subset-disable API preserves the bus for real feedback.
        int rc=control.switchPowerOn({0,1,2,3,4,5});
        if (rc) throw std::runtime_error("Trunk disable failed: "+std::to_string(rc));
        want_enabled=false;
        auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
        while (true) {
            healthy();auto data=status();bool locked=true;
            for (size_t i=0;i<4;++i) locked=locked && data.at("brakes").at(i).get<int>()==0;
            if (locked) return;
            if (std::chrono::steady_clock::now()>=deadline) throw std::runtime_error("Trunk disabled but waist/leg brake lock not confirmed");
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
};
