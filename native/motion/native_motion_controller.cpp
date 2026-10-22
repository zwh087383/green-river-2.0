// SDK facade must share the vendor's ArmRegistry and ControllerManagerRegistry.
// A controller plugin supplies that context without hardware topic commands.
#define GREENRIVER_MOTION_EMBEDDED
#include "sdk_motion_worker.cpp"
#include "controller_interface/controller_interface.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace greenriver {
class NativePresetController : public controller_interface::ControllerInterface {
    std::thread server;
    std::atomic<bool> done{false};
    bool started = false;
public:
    controller_interface::CallbackReturn on_init() override {
        return controller_interface::CallbackReturn::SUCCESS;
    }
    controller_interface::InterfaceConfiguration command_interface_configuration() const override {
        return {controller_interface::interface_configuration_type::NONE, {}};
    }
    controller_interface::InterfaceConfiguration state_interface_configuration() const override {
        return {controller_interface::interface_configuration_type::NONE, {}};
    }
    controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State&) override {
        if (started) return controller_interface::CallbackReturn::ERROR;
        const char* configured = std::getenv("GREENRIVER_MOTION_SOCKET");
        if (!configured) return controller_interface::CallbackReturn::ERROR;
        std::string path(configured);
        std::string timeout = std::getenv("GREENRIVER_MOTION_TIMEOUT") ? std::getenv("GREENRIVER_MOTION_TIMEOUT") : "30";
        stopping = false; done = false; started = true;
        server = std::thread([this, path, timeout] {
            std::string name = "native_motion", option = "--socket", limit = "--timeout";
            std::string socket_path(path), seconds(timeout);
            char* args[] = {name.data(), option.data(), socket_path.data(), limit.data(), seconds.data()};
            motion_server(5, args);
            done = true;
        });
        return controller_interface::CallbackReturn::SUCCESS;
    }
    controller_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State&) override {
        stopping = true;
        auto deadline = Clock::now() + 2500ms;
        while (!done && Clock::now() < deadline) std::this_thread::sleep_for(10ms);
        if (!done) return controller_interface::CallbackReturn::ERROR;
        if (server.joinable()) server.join();
        return controller_interface::CallbackReturn::SUCCESS;
    }
    controller_interface::return_type update(const rclcpp::Time&, const rclcpp::Duration&) override {
        return controller_interface::return_type::OK;
    }
    ~NativePresetController() override {
        stopping = true;
        // This plugin runs only in the dedicated arm host. Never unload while a
        // vendor SDK call is stuck; terminate that host instead of reconnecting.
        if (server.joinable()) server.join();
    }
};
}
PLUGINLIB_EXPORT_CLASS(greenriver::NativePresetController, controller_interface::ControllerInterface)
