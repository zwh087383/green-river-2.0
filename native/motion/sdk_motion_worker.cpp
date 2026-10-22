// Native SDK socket service, hosted in the vendor arm process by a controller
// plugin. A standalone main is compiled only for fake-SDK integration tests.
// No per-motion ROS command publisher and no shell command execution.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <unordered_set>
#include <fcntl.h>
#include <poll.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include "motion_presets.hpp"
#include "vendor_motion_steps.hpp"
#include "request_ledger.hpp"
#ifndef GREENRIVER_MOTION_FAKE
#include "arm_motion.hpp"
#include "trunk_power.hpp"
#include "sdk_compat.hpp"
#endif

using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;
int64_t wall_nanoseconds() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}
std::atomic<bool> stopping{false};
void signal_stop(int) { stopping = 1; }

std::string quote(const std::string& value) {
    std::string out = "\"";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (c >= 32 && c < 127) out += c;
        else out += '?';
    }
    return out + '"';
}

struct SdkCallFailure : std::runtime_error {
    int code;
    std::string call;
    SdkCallFailure(std::string message, std::string method, int rc)
        : std::runtime_error(std::move(message)), code(rc), call(std::move(method)) {}
};

struct Job {
    std::string token, purpose;
    int code;
    std::shared_ptr<const MotionPreset> preset = nullptr;
    std::string scope = "";
};
struct State {
    std::mutex lock;
    std::condition_variable wake;
    std::string phase = "initializing", detail, token, purpose;
    std::optional<Job> pending;
    std::optional<Job> pending_home;
    std::optional<Job> fault_restart;
    Clock::time_point deadline = Clock::now() + 30s;
    std::deque<std::string> order;
    std::unordered_set<std::string> seen;
    greenriver::RequestLedger ledger;
    unsigned long long next_scope = 0;
    std::unordered_map<std::string, Clock::time_point> scopes;
    unsigned long completed = 0;
    std::string power_source = "unknown";
    bool at_preset = false;
    unsigned int step_index = 0, step_count = 0;
    int left_power = -1, right_power = -1;
    std::string trunk_json = "null";
    bool trunk_on = false;
    Clock::time_point power_updated = Clock::now();
    bool driver_watch=false;
    int64_t driver_watch_since=0, fault_time=0, driver_error_time=0;
    std::string driver_error;
    unsigned long arm_log_count=0;

    State() { ledger.accept("startup", "startup_home", "HOME "); }
    void record_sdk_failure(const SdkCallFailure& error) {
        std::lock_guard<std::mutex> guard(lock);
        if (auto* result=ledger.get(token); result && !result->terminal()) {
            result->sdk_code=error.code; result->sdk_returned=true; result->error_stage=error.call;
        }
    }
    void fault_locked(const std::string& why) {
        bool timeout=why.find("timeout")!=std::string::npos || why.find("timed out")!=std::string::npos;
        ledger.fail_unfinished(why, timeout?"watchdog":"sdk_or_feedback", timeout);
        if (phase != "fault") std::cerr << "[motion] FAULT " << why << std::endl;
        if (phase!="fault" || detail.empty()) {detail=why;fault_time=wall_nanoseconds();}
        phase = "fault"; at_preset = false;driver_watch=false;
        pending.reset(); pending_home.reset();
    }
    void fault(const std::string& why) {
        std::lock_guard<std::mutex> guard(lock);fault_locked(why);
    }
    void ready() {
        std::lock_guard<std::mutex> guard(lock);
        // A timed-out call returning later never clears a latched fault.
        if (phase != "fault") { phase = "ready"; detail.clear(); }
    }
};

std::mutex observed_state_lock;
std::weak_ptr<State> observed_state;

bool driver_error_unlocked(State& state, const std::string& name, const std::string& message,
                           int level, int64_t timestamp) {
    bool arm_logger=name=="left_arm.hardware_interface" || name=="right_arm.hardware_interface"
        || name=="left_arm.arm_controller" || name=="right_arm.arm_controller";
    if (arm_logger) ++state.arm_log_count;
    if (!arm_logger || level<40 || timestamp<state.driver_watch_since || !state.driver_watch_since) return false;
    if (!state.driver_watch && !(state.phase=="fault" && timestamp<=state.fault_time)) return false;
    if (state.driver_error_time && timestamp>=state.driver_error_time) return false;
    state.driver_error_time=timestamp;state.driver_error="SDK driver ["+name+"]: "+message;
    state.ledger.fail_unfinished(state.driver_error,"vendor_driver",false);
    state.phase="fault";state.detail=state.driver_error;state.fault_time=timestamp;
    state.at_preset=false;state.driver_watch=false;state.pending.reset();state.pending_home.reset();
    std::cerr<<"[motion] FIRST_DRIVER_ERROR "<<name<<" "<<message<<std::endl;
    state.wake.notify_one();return true;
}

void observe_driver_error(const std::string& name,const std::string& message,int level,int64_t timestamp) {
    std::shared_ptr<State> state;
    {std::lock_guard<std::mutex> guard(observed_state_lock);state=observed_state.lock();}
    if (!state) return;
    std::lock_guard<std::mutex> guard(state->lock);
    driver_error_unlocked(*state,name,message,level,timestamp);
}

void guard_dispatch(State& state) {
    std::lock_guard<std::mutex> guard(state.lock);
    if (stopping || state.phase=="fault" || Clock::now()>state.deadline)
        throw std::runtime_error(state.detail.empty()?"Motion dispatch stopped or timed out":state.detail);
    if (!state.driver_watch) {state.driver_watch=true;state.driver_watch_since=wall_nanoseconds();}
}

#ifndef GREENRIVER_MOTION_FAKE
template<class T> void check_health(T& control) {
    greenriver::check_health(control);
    for (auto side : {T::Left, T::Right}) {
        std::array<double,7> actual{};
        int rc=control.getArmPosition(side,actual);
        if (rc || !std::all_of(actual.begin(),actual.end(),[](double x){return std::isfinite(x);}))
            throw std::runtime_error("Arm position telemetry failed: "+std::to_string(rc));
    }
}

class Backend {
    wheeled_robot::ArmControl control;
    wheeled_robot::ArmMotion motion;
    TrunkPower trunk;
    std::array<double, 14> preset{};
    std::array<double, 28> joint_limits{};
    bool have_joint_limits = false;
    bool configured = false;
    std::function<void()> active;
public:
    void set_active(std::function<void()> callback) {active=std::move(callback);}
    void connect() {
        const char* targets = std::getenv("GREENRIVER_PRESET_TARGETS");
        if (!targets) throw std::runtime_error("SDK preset target configuration missing");
        std::istringstream input(targets);
        for (double& value : preset) {
            if (!(input >> value) || !std::isfinite(value)) throw std::runtime_error("Invalid SDK preset target configuration");
        }
        if (const char* configured = std::getenv("GREENRIVER_JOINT_LIMITS")) {
            std::istringstream limits(configured);
            for (double& value : joint_limits)
                if (!(limits >> value) || !std::isfinite(value)) throw std::runtime_error("Invalid joint limits");
            have_joint_limits = true;
        }
        if (!std::filesystem::exists("install/product/config/arm_motion.json"))
            throw std::runtime_error("Vendor product motion config missing in SDK working directory");
        configured=true;
        trunk.enable();
        int rc = control.connectDev();
        if (rc) throw SdkCallFailure("connectDev failed: " + std::to_string(rc),"connectDev",rc);
        healthy();
        rc = motion.initialize(&control);
        if (rc) throw SdkCallFailure("ArmMotion initialize failed: " + std::to_string(rc),"initialize",rc);
    }
    void healthy() {
        check_health(control);
    }
    void verify_restart_safe(const std::function<void()>& active) {
        if (!configured) throw std::runtime_error("SDK configuration incomplete; restart blocked");
        std::array<std::array<double,7>,2> previous{};
        std::array<bool,2> observed{};
        for (int sample=0;sample<3;++sample) {
            active();
            auto states=power_states();
            for (int side=0;side<2;++side) {
                if (states[side]==2 || states[side]==3)
                    throw std::runtime_error("Emergency stop or safety door active; restart blocked");
                if (states[side]!=0 && states[side]!=1)
                    throw std::runtime_error("Arm power feedback unknown; restart blocked");
                if (states[side]==1) continue;
                auto arm=side==0?wheeled_robot::ArmControl::Left:wheeled_robot::ArmControl::Right;
                std::array<double,7> actual{};
                if (control.isArmMoving(arm)) throw std::runtime_error("Arm still moving; restart blocked");
                if (control.getArmPosition(arm,actual) ||
                    !std::all_of(actual.begin(),actual.end(),[](double x){return std::isfinite(x);}))
                    throw std::runtime_error("Arm position feedback unavailable; restart blocked");
                for (size_t joint=0;joint<7;++joint)
                    if (observed[side] && std::abs(actual[joint]-previous[side][joint])>.001)
                        throw std::runtime_error("Arm has not stopped; restart blocked");
                previous[side]=actual;observed[side]=true;
            }
            std::this_thread::sleep_for(100ms);
        }
    }
    std::array<int, 2> power_states() {
        return greenriver::read_power(control);
    }
    const char* power_source() const { return greenriver::power_source<wheeled_robot::ArmControl>(); }
    nlohmann::json trunk_status() { return trunk.status(); }
    bool trunk_enabled() const { return trunk.enabled(); }
    void trunk_healthy() { trunk.healthy(); }
    void home_trunk(const std::function<void()>& active) { trunk.home(active); }
    void power_all(bool on) {
        if (on) {
            trunk.enable();
            auto states=power_states();
            if (states[0]!=0 || states[1]!=0) power(true);
        } else {
            trunk.disable();
            auto states=power_states();
            if (states[0]==0 || states[1]==0) power(false);
        }
    }
    void power(bool on) {
        if (on) {
            auto states = power_states();
            for (int value : states)
                if (value == 2 || value == 3)
                    throw std::runtime_error("Emergency stop or safety door active; manual recovery required");
            int rc = control.connectDev();
            if (rc) throw SdkCallFailure("connectDev failed: " + std::to_string(rc),"connectDev",rc);
            healthy();
            rc = motion.initialize(&control);
            if (rc) throw SdkCallFailure("ArmMotion initialize failed: " + std::to_string(rc),"initialize",rc);
        } else {
            // Matches the SDK web UI's false /arm_control/connect operation.
            control.disconnectDev();
        }
    }
    bool moving() {
        // Evaluate both sides; do not short-circuit SDK telemetry refreshes.
        bool left = control.isArmMoving(wheeled_robot::ArmControl::Left);
        bool right = control.isArmMoving(wheeled_robot::ArmControl::Right);
        auto velocity=greenriver::read_velocity(control);
        bool velocity_moving=velocity && std::any_of(velocity->begin(),velocity->end(),
            [](double value){return std::abs(value)>.02;});
        return left || right || velocity_moving;
    }
    PresetFrame snapshot(double scale) {
        PresetFrame frame; frame.scale=scale;
        active(); healthy();
        if (control.getArmPosition(wheeled_robot::ArmControl::Left,frame.left)
                || control.getArmPosition(wheeled_robot::ArmControl::Right,frame.right))
            throw std::runtime_error("Retreat waypoint feedback unavailable");
        validate_frame(frame);
        return frame;
    }
    bool play(int code, const std::function<bool()>& admit = {}, std::vector<PresetFrame>* retreat = nullptr) {
        auto steps=read_vendor_motion_steps(code);
        // Validate every target before the first side effect.
        for (const auto& step:steps) for (int side=0;side<2;++side) {
            const auto& target=side==0?step.left:step.right;
            if (!target) continue;
            for (int j=0;j<7;++j) {
                auto i=(side*7+j)*2;
                if (!have_joint_limits || (*target)[j]<joint_limits[i] || (*target)[j]>joint_limits[i+1])
                    throw std::runtime_error("Vendor step exceeds joint limits");
            }
        }
        for (const auto& step:steps) {
            if (admit && !admit()) return false;
            // Capture both arms before submitting this indivisible two-arm segment.
            if (retreat) retreat->push_back(snapshot(step.scale));
            active();healthy();
            for (int side=0;side<2;++side) {
                const auto& target=side==0?step.left:step.right;
                if (!target) continue;
                auto arm=side==0?wheeled_robot::ArmControl::Left:wheeled_robot::ArmControl::Right;
                std::array<double,7> actual{};
                active();
                if (control.getArmPosition(arm,actual)) throw std::runtime_error("Vendor step feedback unavailable");
                bool unchanged=true;
                for (int j=0;j<7;++j) unchanged=unchanged && std::isfinite(actual[j]) && std::abs(actual[j]-(*target)[j])<.001;
                if (unchanged) continue;
                active();int rc=control.moveArm(arm,{*target},step.scale);
                if (rc) throw SdkCallFailure("Vendor step rejected: "+std::to_string(rc),"moveArm",rc);
            }
            for (int side=0;side<2;++side) {
                const auto& hand=side==0?step.left_hand:step.right_hand;
                if (!hand) continue;
                active();healthy();
                // Same IDs and register used by the supplied ArmMotion implementation.
                int rc=control.updateHandPosture(side==0?wheeled_robot::ArmControl::Left:wheeled_robot::ArmControl::Right,
                                                side==0?40:39,0,*hand);
                if (rc) throw SdkCallFailure("Vendor hand step rejected: "+std::to_string(rc),"updateHandPosture",rc);
            }
            int stable=0;
            while (stable<3) {
                active();healthy();bool reached=!moving();
                for (int side=0;side<2;++side) {
                    const auto& target=side==0?step.left:step.right;
                    if (!target) continue;
                    std::array<double,7> actual{};
                    if (control.getArmPosition(side==0?wheeled_robot::ArmControl::Left:wheeled_robot::ArmControl::Right,actual))
                        throw std::runtime_error("Vendor step feedback unavailable");
                    for (int j=0;j<7;++j) reached=reached && std::isfinite(actual[j]) && std::abs(actual[j]-(*target)[j])<=.02;
                }
                stable=reached?stable+1:0;std::this_thread::sleep_for(100ms);
            }
            auto until=Clock::now()+std::chrono::milliseconds(step.wait_ms);
            while (Clock::now()<until) {
                active();healthy();
                if (admit && !admit()) return false;
                std::this_thread::sleep_for(50ms);
            }
        }
        active(); return true;
    }
    PresetFrame resolve_frame(PresetFrame frame) {
        if (frame.home) {
            std::copy_n(preset.begin(), 7, frame.left.begin());
            std::copy_n(preset.begin() + 7, 7, frame.right.begin());
        }
        return frame;
    }
    void validate_frame(const PresetFrame& frame) {
        if (!have_joint_limits) throw std::runtime_error("SDK joint limits missing");
        for (int side = 0; side < 2; ++side) {
            const auto& target = side == 0 ? frame.left : frame.right;
            for (int joint = 0; joint < 7; ++joint) {
                auto index = (side * 7 + joint) * 2;
                if (!std::isfinite(target[joint]) || target[joint] < joint_limits[index] || target[joint] > joint_limits[index + 1])
                    throw std::runtime_error("Preset joint target exceeds vendor limits");
            }
        }
    }
    void move_frame(const PresetFrame& frame) {
        // moveArm queues one task per side. Submit both once, then the caller
        // waits for stationary feedback AND both target positions before advancing.
        active();int left = control.moveArm(wheeled_robot::ArmControl::Left, {frame.left}, frame.scale);
        if (left) throw SdkCallFailure("Left SDK frame rejected: " + std::to_string(left),"moveArm:left",left);
        active();int right = control.moveArm(wheeled_robot::ArmControl::Right, {frame.right}, frame.scale);
        if (right) throw SdkCallFailure("Right SDK frame rejected: " + std::to_string(right),"moveArm:right",right);
    }
    bool frame_reached(const PresetFrame& frame) {
        bool reached = true;
        for (int side = 0; side < 2; ++side) {
            std::array<double, 7> actual{};
            if (control.getArmPosition(static_cast<wheeled_robot::ArmControl::Side>(side), actual))
                throw std::runtime_error("Preset frame feedback unavailable");
            const auto& target = side == 0 ? frame.left : frame.right;
            for (int joint = 0; joint < 7; ++joint) {
                if (!std::isfinite(actual[joint])) throw std::runtime_error("Invalid joint feedback");
                reached = reached && std::abs(actual[joint] - target[joint]) <= .05;
            }
        }
        return reached;
    }
    bool at_home() { PresetFrame home; home.home=true; return frame_reached(resolve_frame(home)); }
    void verify_home() {
        for (int side = 0; side < 2; ++side) {
            std::array<double, 7> actual{};
            int rc = control.getArmPosition(static_cast<wheeled_robot::ArmControl::Side>(side), actual);
            if (rc) throw std::runtime_error("Preset position readback failed: " + std::to_string(rc));
            for (int joint = 0; joint < 7; ++joint) {
                if (!std::isfinite(actual[joint]) || std::abs(actual[joint] - preset[side * 7 + joint]) > 0.05)
                    throw std::runtime_error("Preset joint position not reached; no automatic retry");
            }
        }
    }
    // The host owns the actual device connection. Do not disconnect a shared
    // vendor driver when this high-level facade is destroyed.
    ~Backend() = default;
};
#else
// Built only by tests. The production binary has no simulation switch.
class Backend {
    bool test_at_home=true;
    bool powered = true;
    bool reconnected = false;
    bool trunk_on = true;
    bool trunk_home = true;
    unsigned int play_calls = 0;
    std::function<void()> active;
public:
    void set_active(std::function<void()> callback) {active=std::move(callback);}
    PresetFrame resolve_frame(PresetFrame frame) { return frame; }
    void validate_frame(const PresetFrame&) {
        if (std::getenv("MOTION_TEST_FRAME_LIMIT_FAIL")) throw std::runtime_error("Test target exceeds limits");
    }
    void move_frame(const PresetFrame& frame) {
        active(); test_at_home=frame.home;
        if (const char* log = std::getenv("MOTION_TEST_LOG")) {
            std::ofstream f(log, std::ios::app);
            f << nlohmann::json({{"scale", frame.scale}, {"home", frame.home},
                                 {"left", frame.left}, {"right", frame.right}}).dump() << std::endl;
        }
        if (std::getenv("MOTION_TEST_FRAME_FAIL")) throw std::runtime_error("Test frame failed");
    }
    bool frame_reached(const PresetFrame&) { return std::getenv("MOTION_TEST_FRAME_NOT_REACHED") == nullptr; }
    void connect() {
        if (std::getenv("MOTION_TEST_CONNECT_FAIL")) throw std::runtime_error("Test connect failure");
    }
    void healthy() {
        if (!powered) throw std::runtime_error("Test arm intentionally disconnected");
        if (std::getenv("MOTION_TEST_HEALTH_FAIL")) throw std::runtime_error("Test disconnected arm");
    }
    void verify_restart_safe(const std::function<void()>& active) {
        active();
        if (std::getenv("MOTION_TEST_UNSAFE_RESTART") || std::getenv("MOTION_TEST_MOVING"))
            throw std::runtime_error("Emergency stop, safety door or moving arm; restart blocked");
    }
    bool moving() { return std::getenv("MOTION_TEST_MOVING") != nullptr; }
    std::array<int, 2> power_states() {
        if (!powered && std::getenv("MOTION_TEST_OFF_UNKNOWN")) return {-1, -1};
        return {powered ? 0 : 1, powered ? 0 : 1};
    }
    const char* power_source() const { return "test"; }
    nlohmann::json trunk_status() {
        return {{"power",std::vector<int>(6,trunk_on?0:1)}, {"positions_deg",{0,0,0,0,0,0}},
                {"brakes",std::vector<int>(6,trunk_on?1:0)}, {"at_home",trunk_home},
                {"default_deg",{0,0,0,0,0,0}}, {"feedback_age_ms",0}};
    }
    bool trunk_enabled() const { return trunk_on; }
    void trunk_healthy() { if (std::getenv("MOTION_TEST_TRUNK_HEALTH_FAIL")) throw std::runtime_error("Test trunk fault"); }
    void home_trunk(const std::function<void()>& active) {
        if (!trunk_on) return;
        active();
        if (std::getenv("MOTION_TEST_TRUNK_HOME_FAIL")) throw std::runtime_error("Test trunk home failed");
        trunk_home=true;
        if (const char* log=std::getenv("MOTION_TEST_TRUNK_LOG")) { std::ofstream f(log,std::ios::app);f<<"HOME"<<std::endl; }
    }
    void power_all(bool on) {
        if (on) {
            if (std::getenv("MOTION_TEST_TRUNK_POWER_FAIL")) throw std::runtime_error("Test trunk enable failed");
            trunk_on=true;
            if (const char* log=std::getenv("MOTION_TEST_TRUNK_LOG")) { std::ofstream f(log,std::ios::app);f<<"ENABLE"<<std::endl; }
            if (!powered) power(true);
        } else {
            if (std::getenv("MOTION_TEST_TRUNK_DISABLE_FAIL")) throw std::runtime_error("Test trunk disable failed");
            trunk_on=false;
            if (const char* log=std::getenv("MOTION_TEST_TRUNK_LOG")) { std::ofstream f(log,std::ios::app);f<<"DISABLE"<<std::endl; }
            if (powered) power(false);
        }
    }
    void power(bool on) {
        if (std::getenv("MOTION_TEST_POWER_FAIL")) throw std::runtime_error("Test SDK power failure");
        const char* log = std::getenv("MOTION_TEST_LOG");
        if (log) { std::ofstream f(log, std::ios::app); f << (on ? "POWER_ON" : "POWER_OFF") << std::endl; }
        powered = on;
        if (on) reconnected = true;
    }
    void verify_home() {
        if (std::getenv("MOTION_TEST_HOME_MISMATCH")) throw std::runtime_error("Test preset not reached");
        if (reconnected && std::getenv("MOTION_TEST_POWER_ON_HOME_FAIL")) throw std::runtime_error("Test powered-on preset not reached");
    }
    bool at_home() { return test_at_home; }
    PresetFrame snapshot(double scale) { PresetFrame frame; frame.scale=scale; frame.home=test_at_home; return frame; }
    bool play(int code, const std::function<bool()>& admit = {}, std::vector<PresetFrame>* retreat = nullptr) {
        int segments=1;
        if (admit) if (const char* value=std::getenv("MOTION_TEST_VENDOR_SEGMENTS")) segments=std::stoi(value);
        const char* duration=std::getenv("MOTION_TEST_DURATION_MS");
        for (int index=0;index<segments;++index) {
            if (admit && !admit()) return false;
            if (retreat) retreat->push_back(snapshot(.5));
            active();
            if (index==0) {
                if (const char* log=std::getenv("MOTION_TEST_LOG")) {
                    std::ofstream f(log,std::ios::app); f<<code<<std::endl;
                }
                ++play_calls;
            }
            if (admit) if (const char* log=std::getenv("MOTION_TEST_SEGMENT_LOG")) {
                std::ofstream f(log,std::ios::app); f<<index<<std::endl;
            }
            test_at_home=code==1;
            std::this_thread::sleep_for(std::chrono::milliseconds(duration?std::stoi(duration):150));
            if (std::getenv("MOTION_TEST_FIRST_PLAY_POWER_OFF") && play_calls==1) {
                powered=false;throw std::runtime_error("Arm power state is not ready: 1");
            }
            if (std::getenv("MOTION_TEST_PLAY_FAIL")) throw std::runtime_error("Test SDK failure");
            if (std::getenv("MOTION_TEST_FIRST_PLAY_FAIL") && play_calls==1) throw std::runtime_error("Test transient SDK failure");
            active();
        }
        return !admit || admit();
    }
};
#endif

void refresh_power_feedback(State& state, Backend& backend) {
    auto values=backend.power_states();
    {
        std::lock_guard<std::mutex> guard(state.lock);
        state.left_power=values[0];state.right_power=values[1];state.power_updated=Clock::now();
        state.power_source=backend.power_source();
    }
    auto trunk_values=backend.trunk_status();
    std::lock_guard<std::mutex> guard(state.lock);
    state.left_power=values[0];state.right_power=values[1];
    state.trunk_json=trunk_values.dump();state.trunk_on=backend.trunk_enabled();
    state.power_updated=Clock::now();
}

void serve_jobs(State& state, Backend& backend, double timeout) {
    auto refresh_power=[&] { refresh_power_feedback(state,backend); };
    while (!stopping) {
        std::optional<Job> job;
        bool powered_off;
        {
            std::unique_lock<std::mutex> guard(state.lock);
            state.wake.wait_for(guard, 100ms, [&] { return state.pending.has_value() || stopping; });
            if (state.phase == "fault") break;
            if (state.pending) { job = state.pending; state.pending.reset();state.ledger.running(job->token); }
            else state.deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(timeout));
            powered_off = state.phase == "powered_off";
        }
        // An intentional disconnect must not trip health checks or execute
        // queued homes/gestures. Only an explicit POWER on can leave it.
        if (!job && powered_off) { backend.trunk_healthy();refresh_power(); continue; }
        bool all_on=job && job->code==301;
        bool all_off=job && job->code==300;
        auto active_check=[&] {
            std::lock_guard<std::mutex> guard(state.lock);
            if (state.phase=="fault" || stopping || Clock::now()>state.deadline)
                throw std::runtime_error("Power operation stopped or timed out");
        };
        if (all_on) { active_check();backend.power_all(true); }
        bool skip_arm=all_off && backend.power_states()==std::array<int,2>{1,1};
        bool power_on = job && job->code == 101;
        bool power_off = job && job->code == 100;
        bool restart_arm = job && job->code == 102;
        bool restart_from_off = restart_arm && backend.power_states()==std::array<int,2>{1,1};
        if (power_on || restart_from_off) {
            {
                std::lock_guard<std::mutex> guard(state.lock);
                if (state.phase == "fault" || stopping || Clock::now() > state.deadline) break;
            }
            backend.power(true);
        }
        if (!skip_arm) backend.healthy();
        backend.trunk_healthy();
        refresh_power();
        if (!job) {
            if (backend.moving()) throw std::runtime_error("Unexpected external arm movement");
            continue;
        }
        if (!skip_arm && backend.moving()) throw std::runtime_error("Arm busy before SDK gesture");
        {
            std::lock_guard<std::mutex> guard(state.lock);
            if (state.phase == "fault" || stopping) break;
        }
        std::cout << "[motion] START " << job->token << " " << job->purpose << std::endl;
        int motion_code = (power_on || power_off || restart_arm || all_on || all_off) ? 1 : job->code;
        bool cancelled=false;
        auto admit = [&] {
            std::lock_guard<std::mutex> guard(state.lock);
            if (job->scope.empty()) return true;
            auto scope=state.scopes.find(job->scope);
            bool live=scope!=state.scopes.end() && Clock::now()<scope->second;
            if (!live) {
                if (!cancelled) {
                    state.deadline=std::max(state.deadline,Clock::now()+
                        std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(job->preset->timeout)));
                    std::cout<<"[motion] FINISH_REQUESTED "<<job->token<<std::endl;
                }
                cancelled=true;
                if (state.phase!="fault") state.phase="finishing";
            }
            return live;
        };
        if (job->preset) {
            auto active = [&] {
                std::lock_guard<std::mutex> guard(state.lock);
                if (state.phase == "fault") throw std::runtime_error(state.detail);
                if (stopping || Clock::now() > state.deadline)
                    throw std::runtime_error("Preset stopped or timed out; no next frame");
            };
            auto progress = [&](unsigned int index) {
                std::lock_guard<std::mutex> guard(state.lock);
                state.step_index = index;
            };
            auto settle = [&](const PresetFrame* frame) {
                int stable = 0;
                while (stable < 3) {
                    active();
                    backend.healthy(); refresh_power();
                    bool reached = frame == nullptr || backend.frame_reached(*frame);
                    stable = (!backend.moving() && reached) ? stable + 1 : 0;
                    std::this_thread::sleep_for(100ms);
                }
                active();
            };
            active();
            std::vector<PresetFrame> frames;
            for (auto frame : job->preset->frames) {
                frame = backend.resolve_frame(frame);
                backend.validate_frame(frame); // Validate ALL frames before any movement.
                frames.push_back(frame);
            }
            backend.verify_home();
            std::vector<PresetFrame> retreat;
            if (job->preset->builtin >= 0) {
                progress(1); active();
                if (job->scope.empty()) backend.play(job->preset->builtin);
                else backend.play(job->preset->builtin,admit,&retreat);
                settle(nullptr);
            } else {
                for (unsigned int index = 0; index < frames.size(); ++index) {
                    auto& frame = frames[index];
                    active();
                    if (!admit()) break;
                    if (!job->scope.empty()) retreat.push_back(backend.snapshot(frame.scale));
                    progress(index + 1); backend.move_frame(frame); settle(&frame);
                    auto until = Clock::now() + std::chrono::milliseconds(frame.wait_ms);
                    while (Clock::now() < until) {
                        active(); backend.healthy(); refresh_power();
                        if (!admit()) break;
                        std::this_thread::sleep_for(50ms);
                    }
                    if (cancelled) break;
                }
            }
            if (!job->scope.empty()) admit();
            if (cancelled && !backend.at_home()) {
                // Retrace the already traversed joint waypoints, with no new poses or holds.
                // Cancellation never bypasses health, feedback or driver-fault guards.
                for (auto it=retreat.rbegin();it!=retreat.rend();++it) {
                    active(); backend.validate_frame(*it); backend.move_frame(*it); settle(&*it);
                }
            }
            if (cancelled || (job->preset->builtin>=0 && job->preset->builtin!=1)) {
                progress(job->preset->builtin>=0?2:frames.size()); active(); backend.play(1); settle(nullptr);
            }
            active(); backend.verify_home();
            motion_code = 1; // Every debug preset finishes at the SDK home pose.
        } else if (!skip_arm) backend.play(motion_code); // Vendor call stays off the voice thread.
        int settled = 0;
        while (!skip_arm && !stopping && settled < 3) {
            backend.healthy();
            settled = backend.moving() ? 0 : settled + 1;
            {
                std::lock_guard<std::mutex> guard(state.lock);
                if (state.phase == "fault") break;
            }
            std::this_thread::sleep_for(100ms);
        }
        bool can_finish;
        {
            std::lock_guard<std::mutex> guard(state.lock);
            can_finish = state.phase != "fault" && !stopping;
        }
        if (!can_finish) break;
        if (motion_code == 1 && !skip_arm) backend.verify_home();
        {std::lock_guard<std::mutex> guard(state.lock);state.driver_watch=false;}
        if (restart_arm && !restart_from_off) {
            active_check();backend.power(false);refresh_power();
            if (backend.power_states()!=std::array<int,2>{1,1})
                throw std::runtime_error("Arm restart stopped: both arms must confirm power off");
            auto until=Clock::now()+500ms;
            while (Clock::now()<until) {
                active_check();backend.trunk_healthy();std::this_thread::sleep_for(50ms);
            }
            active_check();backend.power(true);active_check();
            backend.healthy();backend.play(1);
            int stable=0;
            while (stable<3) {
                active_check();backend.healthy();backend.trunk_healthy();refresh_power();
                stable=backend.moving()?0:stable+1;
                std::this_thread::sleep_for(100ms);
            }
            active_check();backend.verify_home();
            {std::lock_guard<std::mutex> guard(state.lock);state.driver_watch=false;}
        }
        if (all_on || all_off || job->purpose=="startup_home") {
            active_check();backend.home_trunk(active_check);active_check();
        }
        // Never power down unless the home call and joint-feedback check
        // both succeeded. Check the watchdog again before the side effect.
        if (power_off || all_off) {
            {
                std::lock_guard<std::mutex> guard(state.lock);
                if (state.phase == "fault" || stopping || Clock::now() > state.deadline) break;
            }
            if (all_off) backend.power_all(false);else backend.power(false);
        }
        refresh_power();
        {
            std::lock_guard<std::mutex> guard(state.lock);
            if (state.phase == "fault" || stopping) break;
            ++state.completed;
            if (auto* record=state.ledger.get(job->token); record && !record->terminal()) {
                record->sdk_returned=true;record->sdk_code=0;record->target_verified=true;
                record->process_verified=!cancelled && (bool(job->preset) || motion_code!=1);
                bool confirmed=(!power_off && !all_off) || (state.left_power==1 && state.right_power==1);
                record->status=confirmed?(cancelled?"cancelled_home":"verified_completed"):"unknown";
                record->home_verified=motion_code==1 && !skip_arm;
                record->detail=cancelled?"Playback ended; remaining gesture segments omitted; retreat and home verified":
                    confirmed?"SDK steps and final joint readback verified":"Home verified; power-off feedback unconfirmed";
                record->finished_ms=state.ledger.now_ms();
            }
            state.at_preset = motion_code == 1 && !power_off && !all_off;
            state.detail = all_off ? "Arms and trunk home verified; disable and brake feedback confirmed" : power_off ? "Preset verified; SDK disconnect returned; check power feedback" :
                motion_code == 1 ? "Preset verified from joint feedback" : "SDK returned; both arms stationary";
            if (power_off || all_off) {
                state.phase = "powered_off";
                state.pending_home.reset();
            } else if (state.pending_home) {
                state.pending = state.pending_home; state.pending_home.reset();
                state.token = state.pending->token; state.purpose = state.pending->purpose;
                state.phase = "homing";
                state.deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(timeout));
            } else state.phase = "ready";
        }
        std::cout << "[motion] FINISHED " << job->token << std::endl;
    }
}

void sdk_loop(State& state, double timeout) {
    // Retain the facade even after a fault: its SDK destructor disconnects the
    // devices. A failed home must not implicitly become a power-off operation.
    std::unique_ptr<Backend> ownership;
    try {
        ownership = std::make_unique<Backend>();
        Backend& backend = *ownership;
        backend.set_active([&state]{guard_dispatch(state);});
        backend.connect();
        backend.healthy();
        auto refresh_power=[&] { refresh_power_feedback(state,backend); };
        refresh_power();
        if (backend.moving()) throw std::runtime_error("Arm already moving at startup");
        {
            std::lock_guard<std::mutex> guard(state.lock);
            if (state.phase != "fault" && !stopping) {
#ifdef GREENRIVER_MOTION_FAKE
                bool home = !std::getenv("MOTION_TEST_SKIP_STARTUP_HOME");
#else
                bool home = true;
#endif
                state.phase = home ? "homing" : "ready";
                if (home) {
                    state.token = "startup"; state.purpose = "startup_home";
                    state.pending = Job{"startup", "startup_home", 1};
                }
                state.deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(home ? std::max(timeout,60.) : timeout));
            }
        }
        serve_jobs(state,backend,timeout);
    } catch (const SdkCallFailure& exc) {
        state.record_sdk_failure(exc);state.fault(exc.what());
    } catch (const std::exception& exc) {
        state.fault(exc.what());
    } catch (...) { state.fault("Unknown SDK exception"); }
    // The same SDK thread consumes an explicit recovery only after the previous
    // SDK call has unwound. A pending click never unlatches the old instruction.
    while (!stopping) {
        std::optional<Job> recovery;
        {
            std::unique_lock<std::mutex> guard(state.lock);
            state.wake.wait_for(guard,100ms,[&]{return state.fault_restart.has_value() || stopping;});
            if (stopping) break;
            if (state.fault_restart) {
                if (Clock::now()>state.deadline) {
                    if(auto* record=state.ledger.get(state.fault_restart->token)) {
                        record->status="failed";record->detail="Previous SDK call did not exit; restart not executed";
                        record->finished_ms=state.ledger.now_ms();
                    }
                    state.fault_restart.reset();state.detail="Previous SDK call did not exit; restart not executed";
                } else {
                    recovery=state.fault_restart;state.fault_restart.reset();
                    state.ledger.running(recovery->token);
                    state.phase="restarting_arms";state.detail.clear();state.at_preset=false;
                    state.driver_watch=false;state.driver_watch_since=0;state.driver_error.clear();state.driver_error_time=0;state.fault_time=0;
                    state.deadline=Clock::now()+60s;
                }
            }
        }
        if (!recovery) {
            // Read-only refresh after a fault; never hide a stale "on" snapshot
            // behind a genuine power-off fault. Preserve the first fault cause.
            if (ownership) {try {refresh_power_feedback(state,*ownership);} catch (...) {}}
            continue;
        }
        try {
            if (!ownership) throw std::runtime_error("SDK not initialized; restart blocked");
            Backend& backend=*ownership;
            auto active=[&] {
                std::lock_guard<std::mutex> guard(state.lock);
                if (stopping || state.phase=="fault" || Clock::now()>state.deadline)
                    throw std::runtime_error("Arm fault recovery stopped or timed out");
            };
            active();backend.trunk_healthy();backend.verify_restart_safe(active);
            // Homing cannot run against a faulted arm connection. Verify that
            // it has stopped, disconnect at the held pose, then reconnect/home.
            active();backend.power(false);refresh_power_feedback(state,backend);
            if (backend.power_states()!=std::array<int,2>{1,1})
                throw std::runtime_error("Arm fault recovery stopped: power off not confirmed");
            auto until=Clock::now()+500ms;
            while (Clock::now()<until) {active();backend.trunk_healthy();std::this_thread::sleep_for(50ms);}
            active();backend.power(true);active();backend.healthy();backend.play(1);
            int stable=0;
            while (stable<3) {
                active();backend.healthy();backend.trunk_healthy();refresh_power_feedback(state,backend);
                stable=backend.moving()?0:stable+1;std::this_thread::sleep_for(100ms);
            }
            active();backend.verify_home();refresh_power_feedback(state,backend);
            {
                std::lock_guard<std::mutex> guard(state.lock);
                if (state.phase=="fault" || stopping) throw std::runtime_error("Arm recovery interrupted");
                state.phase="ready";state.at_preset=true;state.detail="Arm fault recovery and preset feedback verified";
                state.driver_watch=false;
                state.step_index=state.step_count=0;++state.completed;
                if(auto* record=state.ledger.get(recovery->token)) {
                    record->status="verified_completed";record->target_verified=true;
                    record->sdk_returned=true;record->sdk_code=0;record->detail="Explicit arm recovery and home verified";
                    record->finished_ms=state.ledger.now_ms();
                }
                state.deadline=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(timeout));
            }
            std::cout<<"[motion] RECOVERED "<<recovery->token<<std::endl;
            serve_jobs(state,backend,timeout);
        } catch (const SdkCallFailure& exc) {state.record_sdk_failure(exc);state.fault(exc.what());}
          catch (const std::exception& exc) {state.fault(exc.what());}
          catch (...) {state.fault("Unknown arm recovery exception");}
    }
}

bool valid_token(const std::string& token) {
    return !token.empty() && token.size() <= 128 && std::all_of(token.begin(), token.end(), [](unsigned char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '-' || c == '_';
    });
}

std::string handle(State& state, const std::string& line, double timeout) {
    std::istringstream in(line);
    std::string cmd, token, purpose, extra, query, expected_instance;
    in >> cmd;
    std::lock_guard<std::mutex> guard(state.lock);
    if (Clock::now() > state.deadline && state.phase != "fault") {
        state.fault_locked("SDK watchdog timeout; manual recovery required");
    }
    if (state.phase=="fault" && state.fault_restart && Clock::now()>state.deadline) {
        if(auto* record=state.ledger.get(state.fault_restart->token)) {
                        record->status="failed";record->detail="Previous SDK call did not exit; restart not executed";
                        record->finished_ms=state.ledger.now_ms();
                    }
                    state.fault_restart.reset();state.detail="Previous SDK call did not exit; restart not executed";
    }
    std::string signature_cmd,signature_token,signature_purpose;
    std::istringstream signature(line);signature>>signature_cmd>>signature_token>>signature_purpose;
    std::string fingerprint=signature_cmd+" "+signature_purpose;
    bool mutation=signature_cmd=="PLAY" || signature_cmd=="HOME" || signature_cmd=="PRESET" || signature_cmd=="GESTURE" || signature_cmd=="POWER" || signature_cmd=="ALLPOWER";
    if (signature_cmd=="GESTURE") {
        std::string scope,instance; signature>>scope>>instance; fingerprint+=" "+scope+" "+instance;
    }
    bool conflict=mutation && state.ledger.contains(signature_token) && !state.ledger.same(signature_token,fingerprint);
    std::string result, response_scope;
    if (conflict) result="conflict";
    else if (cmd=="RESULT" && (in>>token) && valid_token(token)) {
        bool has_instance=bool(in>>expected_instance);
        if ((!has_instance || valid_token(expected_instance)) && !(in>>extra)) {
            query=token;
            result=has_instance && expected_instance!=state.ledger.instance_id?"instance_changed":
                state.ledger.contains(token)?"ok":"not_found";
        } else result="invalid";
    }
    else if (cmd=="OPEN" && (in>>expected_instance) && !(in>>extra)) {
        if (expected_instance!=state.ledger.instance_id) result="instance_changed";
        else {
            for (auto it=state.scopes.begin();it!=state.scopes.end();) {
                if (Clock::now()>=it->second) it=state.scopes.erase(it); else ++it;
            }
            if (state.scopes.size()>=64 || state.next_scope==~0ULL) result="scope_capacity";
            else {
                response_scope=std::to_string(++state.next_scope);
                state.scopes[response_scope]=Clock::now()+10s; result="accepted";
            }
        }
    }
    else if ((cmd=="KEEP" || cmd=="END") && (in>>response_scope>>expected_instance)
             && !(in>>extra) && valid_token(response_scope)) {
        if (expected_instance!=state.ledger.instance_id) result="instance_changed";
        else {
            auto scope=state.scopes.find(response_scope);
            if (cmd=="END") {state.scopes.erase(response_scope);result="closed";}
            else if (scope==state.scopes.end() || Clock::now()>=scope->second) result="closed";
            else {scope->second=Clock::now()+10s;result="ok";}
        }
    }
    else if (cmd == "STATUS" && !(in >> extra)) result = "ok";
    else if (cmd=="DRIVER_LOG") {
        int64_t stamp=0;int level=0;std::string name,hex;
        if (!(in>>stamp>>level>>name>>hex) || (in>>extra) || stamp<=0 || level<20 || level>50 || hex.empty() || hex.size()>224 || hex.size()%2)
            result="invalid";
        else {
            auto digit=[](char c)->int {return c>='0'&&c<='9'?c-'0':c>='a'&&c<='f'?c-'a'+10:-1;};
            std::string message;bool valid=true;
            for (size_t i=0;i<hex.size();i+=2) {
                int high=digit(hex[i]),low=digit(hex[i+1]);
                if (high<0 || low<0 || high*16+low<32) {valid=false;break;}
                message.push_back(static_cast<char>(high*16+low));
            }
            if (!valid) result="invalid";
            else {driver_error_unlocked(state,name,message,level,stamp);result="ok";}
        }
    }
#ifdef GREENRIVER_MOTION_FAKE
    else if (cmd=="TEST_DRIVER_ERROR" && (in>>token) && !(in>>extra)) {
        driver_error_unlocked(state,"right_arm.hardware_interface",token,40,wall_nanoseconds());result="ok";
    }
#endif
    else if (cmd == "ALLPOWER" && (in >> token >> purpose) && !(in >> extra) && valid_token(token) &&
             (purpose=="on" || purpose=="off")) {
        if (state.seen.count(token)) result="duplicate";
        else if (state.phase=="fault") result="unavailable";
        else if (state.phase!="ready" && state.phase!="powered_off") result="busy";
        else if (purpose=="on" && state.trunk_on && state.left_power==0 && state.right_power==0) result="already_on";
        else if (purpose=="off" && !state.trunk_on && state.left_power==1 && state.right_power==1) result="already_off";
        else {
            state.seen.insert(token);state.order.push_back(token);
            if (state.order.size()>512) {state.seen.erase(state.order.front());state.order.pop_front();}
            state.token=token;state.purpose="all_power_"+purpose+"_home";
            state.phase=purpose=="on" ? "powering_on_all" : "powering_off_all";
            state.detail.clear();state.at_preset=false;state.pending_home.reset();
            state.deadline=Clock::now()+std::chrono::seconds(60);
            state.pending=Job{token,state.purpose,purpose=="on"?301:300};state.wake.notify_one();result="accepted";
        }
    }
    else if (cmd == "POWER" && (in >> token >> purpose) && !(in >> extra) && valid_token(token) &&
             (purpose == "on" || purpose == "off" || purpose == "restart")) {
        if (state.seen.count(token)) result = "duplicate";
        else if (state.phase=="fault" && purpose=="restart") {
            if (state.fault_restart) result="busy";
            else {
                state.seen.insert(token);state.order.push_back(token);
                if (state.order.size()>512) {state.seen.erase(state.order.front());state.order.pop_front();}
                state.token=token;state.purpose="power_restart_home";state.pending.reset();state.pending_home.reset();
                state.fault_restart=Job{token,state.purpose,103};state.deadline=Clock::now()+60s;
                state.wake.notify_one();result="accepted";
            }
        }
        else if (state.phase == "fault") result = "unavailable";
        else if (state.phase != "ready" && state.phase != "powered_off") result = "busy";
        else if (purpose == "off" && state.phase == "powered_off") result = "already_off";
        else if (purpose == "on" && state.phase == "ready") result = "already_on";
        else {
            state.seen.insert(token); state.order.push_back(token);
            if (state.order.size() > 512) { state.seen.erase(state.order.front()); state.order.pop_front(); }
            state.token = token; state.purpose = "power_" + purpose + "_home";
            state.phase = purpose == "restart" ? "restarting_arms" : purpose == "on" ? "powering_on" : "powering_off";
            state.detail.clear(); state.at_preset = false; state.pending_home.reset();
            state.deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(purpose=="restart" ? std::max(timeout,60.) : timeout));
            state.pending = Job{token, state.purpose, purpose == "restart" ? 102 : purpose == "on" ? 101 : 100};
            state.wake.notify_one(); result = "accepted";
        }
    }
    else if (cmd == "HOME" && (in >> token) && !(in >> extra) && valid_token(token)) {
        if (state.seen.count(token)) result = "duplicate";
        else if (state.phase == "fault" || state.phase == "powered_off" ||
                 state.phase == "powering_on" || state.phase == "powering_off" ||
                  state.phase == "powering_on_all" || state.phase == "powering_off_all" ||
                  state.phase == "restarting_arms") result = "unavailable";
        else {
            state.seen.insert(token); state.order.push_back(token);
            if (state.order.size() > 512) { state.seen.erase(state.order.front()); state.order.pop_front(); }
            if (state.phase == "initializing" || state.phase == "homing" || state.pending_home) result = "coalesced";
            else if (state.phase == "running" || state.phase == "finishing") {
                state.pending_home = Job{token, "refresh_home", 1}; result = "queued";
            } else {
                state.token = token; state.purpose = "refresh_home"; state.phase = "homing"; state.detail.clear(); state.at_preset = false;
                state.deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(timeout));
                state.pending = Job{token, "refresh_home", 1}; state.wake.notify_one(); result = "accepted";
            }
        }
    }
    else if ((cmd=="PRESET" || cmd=="GESTURE") && (in>>token>>purpose)
             && valid_token(token) && preset_id_valid(purpose)) {
        std::string scope, instance;
        bool scoped=cmd=="GESTURE";
        bool parsed=!scoped || bool(in>>scope>>instance);
        bool live=!scoped || (state.scopes.count(scope) && Clock::now()<state.scopes.at(scope));
        if (!parsed || (in>>extra)) result="invalid";
        else if (scoped && instance!=state.ledger.instance_id) result="instance_changed";
        else if (!live) result="closed";
        else if (state.seen.count(token)) result = "duplicate";
        else if (state.phase == "fault" || state.phase == "powered_off") result = "unavailable";
        else if (state.phase != "ready") result = "busy";
        else if (!state.at_preset) result = "not_at_home";
        else {
            try {
                auto preset = std::make_shared<const MotionPreset>(read_motion_preset(purpose));
                state.seen.insert(token); state.order.push_back(token);
                if (state.order.size() > 512) { state.seen.erase(state.order.front()); state.order.pop_front(); }
                state.token = token; state.purpose = "preset:" + purpose;
                state.phase = "running"; state.detail.clear(); state.at_preset = false;
                state.step_index = 0;
                state.step_count = preset->builtin >= 0 ? (preset->builtin == 1 ? 1 : 2) : preset->frames.size();
                state.deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(preset->timeout));
                state.pending = Job{token, state.purpose, 200, preset, scope};
                state.wake.notify_one(); result = "accepted";
            } catch (const std::exception&) { result = "invalid_preset"; }
        }
    }
    else if (cmd == "PLAY" && (in >> token >> purpose) && !(in >> extra) && valid_token(token) &&
             (purpose == "wake_wave" || purpose == "answer_gesture" || purpose == "closing_wave")) {
        if (state.seen.count(token)) result = "duplicate";
        else if (state.phase != "ready") result = (state.phase == "running" || state.phase == "homing") ? "busy" : "unavailable";
        else {
            state.seen.insert(token); state.order.push_back(token);
            if (state.order.size() > 512) { state.seen.erase(state.order.front()); state.order.pop_front(); }
            state.token = token; state.purpose = purpose; state.phase = "running"; state.detail.clear(); state.at_preset = false;
            state.deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(timeout));
            state.pending = Job{token, purpose, purpose == "answer_gesture" ? 5 : 4};
            state.wake.notify_one(); result = "accepted";
        }
    } else result = "invalid";
    if (mutation && valid_token(signature_token) && (result=="accepted" || result=="queued" || result=="coalesced")) {
        std::string canonical;
        if(result=="coalesced") canonical=state.pending_home?state.pending_home->token:state.token.empty()?"startup":state.token;
        state.ledger.accept(signature_token,result=="queued" || result=="coalesced"?"refresh_home":state.purpose,fingerprint,canonical);
    }
    if(query.empty()) query=mutation?signature_token:state.token.empty()?"startup":state.token;
    return "{\"status\":" + quote(result) + ",\"state\":" + quote(state.phase) +
           ",\"detail\":" + quote(state.detail) + ",\"request_id\":" + quote(state.token) +
           ",\"instance_id\":" + quote(state.ledger.instance_id) +
           ",\"scope_id\":" + quote(response_scope) +
           ",\"result\":" + (result=="instance_changed"?"null":state.ledger.json(query)) +
           ",\"hardware_timestamp_available\":false,\"power_source\":" + quote(state.power_source) +
           ",\"purpose\":" + quote(state.purpose) + ",\"home_pending\":" + (state.pending_home ? "true" : "false") + ",\"at_preset\":" + (state.at_preset ? "true" : "false") + ",\"completed\":" + std::to_string(state.completed) +
           ",\"restart_pending\":" + (state.fault_restart ? "true" : "false") +
           ",\"driver_error\":" + quote(state.driver_error) +
           ",\"arm_log_count\":" + std::to_string(state.arm_log_count) +
           ",\"step_index\":" + std::to_string(state.step_index) + ",\"step_count\":" + std::to_string(state.step_count) +
           ",\"power\":{\"left\":" + std::to_string(state.left_power) + ",\"right\":" + std::to_string(state.right_power) + "}" +
           ",\"trunk\":" + state.trunk_json +
           ",\"power_feedback_age_ms\":" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - state.power_updated).count()) +
#ifdef GREENRIVER_MOTION_FAKE
           ",\"backend\":\"test\"}\n";
#else
           ",\"backend\":\"native_sdk\"}\n";
#endif
}

#ifndef GREENRIVER_MOTION_FAKE
void check_other_owner() {
    for (const auto& entry : std::filesystem::directory_iterator("/proc")) {
        std::ifstream file(entry.path() / "cmdline");
        std::string command;
        std::getline(file, command, '\0');
        std::string name = std::filesystem::path(command).filename();
        if (name == "arm_control_node" || name == "arm_test")
            throw std::runtime_error("Existing vendor arm controller owns hardware: " + entry.path().string());
    }
}
#endif

int motion_server(int argc, char** argv) {
    try {
        std::string path;
        double timeout = 30;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--socket" && i + 1 < argc) path = argv[++i];
            else if (arg == "--timeout" && i + 1 < argc) timeout = std::stod(argv[++i]);
            else throw std::runtime_error("Usage: sdk_motion_worker --socket PATH [--timeout SECONDS]");
        }
        if (path.empty() || path.size() >= sizeof(sockaddr_un::sun_path) || !std::isfinite(timeout) || timeout <= 0 || timeout > 300)
            throw std::runtime_error("Invalid socket path or timeout");
        umask(0077);
#ifdef GREENRIVER_MOTION_FAKE
        std::string lock_path = path + ".lock";
#else
        std::string lock_path = "/run/user/" + std::to_string(getuid()) + "/greenriver-arm-sdk.lock";
#endif
        int owner = open(lock_path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
        if (owner < 0 || flock(owner, LOCK_EX | LOCK_NB)) throw std::runtime_error("Native SDK owner already running or lock inaccessible");
#if !defined(GREENRIVER_MOTION_FAKE) && !defined(GREENRIVER_MOTION_EMBEDDED)
        check_other_owner();
#endif
        int server = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
        if (server < 0) throw std::runtime_error("socket failed");
        sockaddr_un address{}; address.sun_family = AF_UNIX;
        std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
        if (std::filesystem::exists(path)) {
            struct stat info{};
            if (lstat(path.c_str(), &info) || !S_ISSOCK(info.st_mode) || info.st_uid != getuid())
                throw std::runtime_error("Refusing to replace non-owned socket path");
            unlink(path.c_str());
        }
        if (bind(server, reinterpret_cast<sockaddr*>(&address), sizeof(address)) || listen(server, 8))
            throw std::runtime_error("bind/listen failed");
#ifndef GREENRIVER_MOTION_EMBEDDED
        signal(SIGINT, signal_stop); signal(SIGTERM, signal_stop); signal(SIGPIPE, SIG_IGN);
#endif
        auto state_owner=std::make_shared<State>();State& state=*state_owner;
        state.deadline=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(timeout));
        {std::lock_guard<std::mutex> guard(observed_state_lock);observed_state=state_owner;}
        std::atomic<bool> sdk_done{false};
        std::thread sdk([&] { sdk_loop(state, timeout); sdk_done = true; });
        std::cout << "[motion] Socket listening: " << path << std::endl;
        while (!stopping) {
            // Watchdog also covers initialization and telemetry calls, without issuing resets.
            {
                std::lock_guard<std::mutex> guard(state.lock);
                if (state.phase != "fault" && Clock::now() > state.deadline) {
                    state.fault_locked("SDK watchdog timeout; manual recovery required");
                }
            }
            pollfd wait{server, POLLIN, 0};
            if (poll(&wait, 1, 100) <= 0) continue;
            int client = accept4(server, nullptr, nullptr, SOCK_CLOEXEC);
            if (client < 0) continue;
            // Bound idle/malformed clients; they cannot stall the SDK or voice playback.
            auto deadline = Clock::now() + 100ms;
            std::string line;
            bool complete = false;
            while (line.size() <= 300 && Clock::now() < deadline) {
                pollfd input{client, POLLIN, 0};
                auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
                if (poll(&input, 1, std::max(1L, left)) <= 0) break;
                char c;
                if (recv(client, &c, 1, 0) != 1) break;
                if (c == '\n') { complete = true; break; }
                line += c;
            }
            auto reply = complete ? handle(state, line, timeout) : std::string("{\"status\":\"invalid\"}\n");
            // Reply is <1KB, fits the socket buffer; don't let an abandoned peer block us.
            send(client, reply.data(), reply.size(), MSG_NOSIGNAL | MSG_DONTWAIT);
            close(client);
        }
        state.wake.notify_one();
        close(server); unlink(path.c_str());
        // Give normal SDK destruction time to disconnect. Never join a blocked
        // vendor call forever and never issue an extra physical reset on exit.
        auto exit_deadline = Clock::now() + 2s;
        while (!sdk_done && Clock::now() < exit_deadline) std::this_thread::sleep_for(10ms);
        if (sdk_done) {
            sdk.join();
            std::cout << "[motion] Stopped; no reset command sent" << std::endl;
            return 0;
        }
        sdk.detach();
        std::cerr << "[motion] SDK did not stop; exiting without reset or automatic restart" << std::endl;
        std::_Exit(0);
    } catch (const std::exception& exc) {
        std::cerr << "[motion] " << exc.what() << std::endl;
        return 1;
    }
}

#ifdef GREENRIVER_MOTION_FAKE
int main(int argc, char** argv) { return motion_server(argc, argv); }
#endif
