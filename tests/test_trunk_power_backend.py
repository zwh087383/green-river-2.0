from pathlib import Path
import os
import shutil
import subprocess
import pytest

ROOT=Path(__file__).parents[1]


def test_real_trunk_backend_uses_sdk_all_six_disable_and_verified_default(tmp_path):
    if not shutil.which('g++'):pytest.skip('g++ required')
    (tmp_path/'trunk_control.hpp').write_text(r'''
#pragma once
#include <array>
#include <vector>
inline bool motor_enabled=false;
inline bool requested_home=false;
inline std::vector<uint8_t> disabled_axes;
namespace wheeled_robot {
class TrunkControl {
public:
 int connectDev(unsigned mode) { if(mode!=0)return 1;motor_enabled=true;return 0; }
 int getJointPosition(std::array<float,6>& values) { values.fill(0);return 0; }
 int getMotorBrakeState(std::array<uint8_t,6>& values) { values.fill(motor_enabled?1:0);return 0; }
 int getMotorMovingState(std::array<bool,6>& values) { values.fill(false);return 0; }
 int setJointTargetPosition(const std::array<float,6>& target,float scale) {
  if(scale>.3f || target!=std::array<float,6>{0,0,0,0,0,0})return 1;
  requested_home=true;return 0;
 }
 int switchPowerOn(const std::vector<uint8_t>& axes) { disabled_axes=axes;motor_enabled=false;return 0; }
 int emergencyStop(){return 0;}
};
}
''')
    source=tmp_path/'probe.cpp'
    source.write_text(r'''
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <nlohmann/json.hpp>
#include "trunk_power.hpp"
extern "C" int harmonic_getStatusWord(uint8_t adapter,uint8_t motor,uint16_t* word,unsigned) {
 if(adapter!=2 || motor<1 || motor>6)return -1;
 *word=motor_enabled?0x27:0x23;return 0;
}
int main(){
 TrunkPower trunk;trunk.enable();trunk.home([]{});
 if(!requested_home || !trunk.status().at("at_home").get<bool>())return 1;
 trunk.disable();
 if(disabled_axes!=std::vector<uint8_t>{0,1,2,3,4,5})return 2;
 if(trunk.status().at("power")!=nlohmann::json::array({1,1,1,1,1,1}))return 3;
 trunk.enable();if(!trunk.enabled())return 4;
 return 0;
}
''')
    binary=tmp_path/'probe'
    subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-pthread','-I',str(tmp_path),'-I',str(ROOT/'native/motion'),str(source),'-o',str(binary)],check=True,capture_output=True,text=True)
    subprocess.run([str(binary)],env=dict(os.environ,GREENRIVER_TRUNK_HOME_TARGETS='0 0 0 0 0 0'),check=True,timeout=8)
