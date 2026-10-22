from pathlib import Path
import json
import os
import shutil
import subprocess

import pytest

ROOT=Path(__file__).parents[1]


def test_production_backend_stops_after_first_driver_error_before_other_side_or_next_step(tmp_path):
    if not shutil.which('g++'):pytest.skip('g++ required')
    (tmp_path/'arm_control.hpp').write_text(r'''
#pragma once
#include <array>
#include <vector>
#include <string>
#include <cstdint>
void observe_driver_error(const std::string&,const std::string&,int,int64_t);
int64_t wall_nanoseconds();
inline int arm_moves=0,hand_moves=0;
inline bool inject_error=false;
namespace wheeled_robot {
class ArmControl {
 std::array<std::array<double,7>,2> position{};
public:
 enum Side {Left=0,Right=1};
 int connectDev(){return 0;}
 void disconnectDev(){}
 int getPowerState(Side){return 0;}
 bool isArmMoving(Side){return false;}
 int getArmPosition(Side s,std::array<double,7>& out){out=position[s];return 0;}
 int moveArm(Side s,const std::vector<std::array<double,7>>& targets,double){
  ++arm_moves;position[s]=targets.back();
  if(inject_error)observe_driver_error("right_arm.hardware_interface","first-trajectory-error",40,wall_nanoseconds());
  return 0;
 }
 int updateHandPosture(Side s,uint8_t id,uint16_t addr,const std::vector<uint16_t>&){
  if(id!=(s==Left?40:39)||addr!=0)return 1;
  ++hand_moves;return 0;
 }
};
}
''')
    (tmp_path/'arm_motion.hpp').write_text(r'''
#pragma once
#include "arm_control.hpp"
namespace wheeled_robot { class ArmMotion {public:int initialize(ArmControl*){return 0;}}; }
''')
    (tmp_path/'trunk_control.hpp').write_text(r'''
#pragma once
#include <array>
#include <vector>
namespace wheeled_robot { class TrunkControl {public:
 int connectDev(unsigned){return 0;}
 int getJointPosition(std::array<float,6>& x){x.fill(0);return 0;}
 int getMotorBrakeState(std::array<uint8_t,6>& x){x.fill(1);return 0;}
 int getMotorMovingState(std::array<bool,6>& x){x.fill(false);return 0;}
 int setJointTargetPosition(const std::array<float,6>&,float){return 0;}
 int switchPowerOn(const std::vector<uint8_t>&){return 0;}
 int emergencyStop(){return 0;}
}; }
''')
    source=tmp_path/'probe.cpp'
    source.write_text(r'''
#define GREENRIVER_MOTION_EMBEDDED
#include "sdk_motion_worker.cpp"
extern "C" int harmonic_getStatusWord(uint8_t adapter,uint8_t,uint16_t* out,unsigned){if(adapter)return -1;*out=0x27;return 0;}
int main(){
 auto state=std::make_shared<State>();state->phase="running";
 {std::lock_guard<std::mutex> guard(observed_state_lock);observed_state=state;}
 Backend backend;backend.set_active([&]{guard_dispatch(*state);});backend.connect();
 backend.play(4);if(arm_moves!=3||hand_moves!=2)return 1;
 arm_moves=hand_moves=0;inject_error=true;
 try{backend.play(4);return 2;}catch(const std::exception&){}
 if(arm_moves!=1||hand_moves||state->phase!="fault")return 3;
 auto first=state->detail;state->fault("Arm power state is not ready: 1");
 observe_driver_error("right_arm.hardware_interface","secondary-timeout",40,wall_nanoseconds());
 if(state->detail!=first||first.find("first-trajectory-error")==std::string::npos)return 4;
 State idle;idle.phase="ready";
 if(driver_error_unlocked(idle,"web","unrelated",40,wall_nanoseconds()))return 5;
 return 0;
}
''')
    config=tmp_path/'install/product/config/arm_motion.json';config.parent.mkdir(parents=True)
    config.write_text(json.dumps({'Y':[
        {'left_arm':[10]*7,'right_arm':[20]*7,'left_end_effector':[100]*6,'right_end_effector':[100]*6,'scale':.4},
        {'right_arm':[30]*7,'scale':.5}]}))
    binary=tmp_path/'probe'
    compiled=subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-pthread','-I',str(tmp_path),'-I',str(ROOT/'native/motion'),str(source),'-o',str(binary)],capture_output=True,text=True)
    assert compiled.returncode==0,compiled.stderr
    env=dict(os.environ,GREENRIVER_PRESET_TARGETS=' '.join(['0']*14),GREENRIVER_JOINT_LIMITS=' '.join(['-3.2','3.2']*14),GREENRIVER_TRUNK_HOME_TARGETS='0 0 0 0 0 0',GREENRIVER_VENDOR_MOTION_FILE=str(config))
    subprocess.run([str(binary)],cwd=tmp_path,env=env,check=True,capture_output=True,text=True,timeout=10)
