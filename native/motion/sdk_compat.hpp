#pragma once

#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace greenriver {
template<class T, class = void> struct HasPowerState : std::false_type {};
template<class T> struct HasPowerState<T, std::void_t<decltype(std::declval<T&>().getPowerState(T::Left))>> : std::true_type {};
template<class T, class = void> struct HasVelocity : std::false_type {};
template<class T> struct HasVelocity<T, std::void_t<decltype(std::declval<T&>().getJointsVelocity(
    T::Left, std::declval<std::array<double, 7>&>()))>> : std::true_type {};

template<class T> const char* power_source() {
    return HasPowerState<T>::value ? "sdk_enum" : "sdk_boolean";
}

template<class T> std::array<int, 2> read_power(T& control) {
    std::array<int, 2> values{-1, -1};
    for (int i = 0; i < 2; ++i) {
        auto side = static_cast<typename T::Side>(i);
        if constexpr (HasPowerState<T>::value) {
            int value = control.getPowerState(side);
            values[i] = value >= 0 && value <= 3 ? value : -1;
        } else {
            // An unavailable connection cannot prove power-off. Boolean feedback
            // does not expose the safety-door state; never manufacture it.
            if (control.isArmConnected(side)) {
                values[i] = control.isArmEmergencyStopped(side) ? 2 :
                    control.isArmPowerOn(side) ? 0 : 1;
            }
        }
    }
    return values;
}

template<class T> void check_health(T& control) {
    auto values = read_power(control);
    for (int i = 0; i < 2; ++i) {
        if (values[i] != 0) throw std::runtime_error("Arm power/connection is not ready");
    }
}

template<class T> std::optional<std::array<double, 14>> read_velocity(T& control) {
    if constexpr (HasVelocity<T>::value) {
        std::array<double, 14> values{};
        for (int i = 0; i < 2; ++i) {
            std::array<double, 7> side{};
            int rc = control.getJointsVelocity(static_cast<typename T::Side>(i), side);
            if (rc) throw std::runtime_error("Joint velocity readback failed: " + std::to_string(rc));
            for (int j = 0; j < 7; ++j) {
                if (!std::isfinite(side[j])) throw std::runtime_error("Invalid joint velocity");
                values[i * 7 + j] = side[j];
            }
        }
        return values;
    }
    return std::nullopt;
}
} // namespace greenriver
