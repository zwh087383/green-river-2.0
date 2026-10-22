from pathlib import Path
from types import SimpleNamespace
import importlib.util
import subprocess
import xml.etree.ElementTree as ET

import pytest
import yaml


@pytest.fixture
def host():
    path = Path(__file__).parents[1] / 'scripts/sdk_motion_host.py'
    spec = importlib.util.spec_from_file_location('motion_host_description_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('side', ['left', 'right'])
def test_host_supplies_only_requested_arm_description_without_external_publisher(host, tmp_path, monkeypatch, side):
    sdk = tmp_path / 'sdk'
    source = sdk / f'install/config/controller/{side}_arm_controller.yaml'
    source.parent.mkdir(parents=True)
    joints = [f'{side}_joint_{i}' for i in range(7)]
    config = {
        f'/{side}_arm/info': {'ros__parameters': {'arm_joints': joints}},
        f'/{side}_arm/controller_manager': {'ros__parameters': {'update_rate': 1000}},
    }
    source.write_text(yaml.safe_dump(config))
    output = tmp_path / 'controller.yaml'
    def generate(command, **kwargs):
        assert command == ['xacro', str(sdk / 'install/share/robot_description/urdf/wheel_robot_control.xacro'),
                           f'{side}_arm_controller:={output}', f'{side}_arm_plugin:=nerv/arm_hw/ArmRokaeAr5']
        assert kwargs['check'] and kwargs['timeout'] == 20
        assert yaml.safe_load(output.read_text()) == config
        control = f'<ros2_control name="{side}_arm" type="system"><hardware><plugin>nerv/arm_hw/ArmRokaeAr5</plugin></hardware>'
        return SimpleNamespace(stdout='<robot>' + control + ''.join(f'<joint name="{joint}"/>' for joint in joints) + '</ros2_control></robot>')
    monkeypatch.setattr(host.subprocess, 'run', generate)
    description_file = host.prepare_controller_config(sdk, side, output)
    saved = yaml.safe_load(output.read_text())
    assert 'robot_description' not in saved[f'/{side}_arm/controller_manager']['ros__parameters']
    description = yaml.safe_load(description_file.read_text())['/**']['ros__parameters']['robot_description']
    assert ET.fromstring(description).find('ros2_control').get('name') == f'{side}_arm'
    assert saved == config
    assert yaml.safe_load(source.read_text()) == config


@pytest.mark.parametrize('description', ['<robot/>', '<robot><ros2_control name="right_arm"/></robot>', '<invalid'])
def test_invalid_description_aborts_configuration(host, tmp_path, monkeypatch, description):
    sdk = tmp_path / 'sdk'
    source = sdk / 'install/config/controller/left_arm_controller.yaml'
    source.parent.mkdir(parents=True)
    source.write_text(yaml.safe_dump({'/left_arm/info': {'ros__parameters': {'arm_joints': ['j1']}},
                                    '/left_arm/controller_manager': {'ros__parameters': {}}}))
    monkeypatch.setattr(host.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=description))
    with pytest.raises((ValueError, ET.ParseError)):
        host.prepare_controller_config(sdk, 'left', tmp_path / 'output.yaml')


def test_startup_publishers_are_cleaned_up_including_stuck_helpers(host):
    from unittest.mock import Mock
    first = Mock()
    first.poll.return_value = None
    second = Mock()
    second.poll.return_value = None
    second.wait.side_effect = [subprocess.TimeoutExpired('publisher', 3), 0]
    publishers = [first, second]
    host.stop_description_publishers(publishers)
    first.terminate.assert_called_once()
    first.kill.assert_not_called()
    second.terminate.assert_called_once()
    second.kill.assert_called_once()
    assert publishers == []


@pytest.fixture
def initial_pose_sdk(tmp_path):
    sdk=tmp_path/'sdk'
    config=sdk/'teach_script/teach_arm_config_init_pose.yaml';config.parent.mkdir(parents=True)
    config.write_text(yaml.safe_dump({'arm_sequence': [{'name': '初始姿势', 'trunk_position': [-45,-81,-36,0,0,-22]}]}))
    model=sdk/'install/share/robot_description/urdf/wheel_robot.urdf';model.parent.mkdir(parents=True)
    model.write_text('<robot>'+''.join(f'<joint name="{name}"><limit lower="{low}" upper="{high}"/></joint>' for name,low,high in (
        ('trunk1_joint',-2,1),('trunk2_joint',-3,1),('trunk3_joint',-2,1),('trunk4_joint',-1,1),('head1_joint',-1.57,1.57),('head2_joint',-.5,.5)))+'</robot>')
    return sdk


def test_trunk_default_uses_teaching_body_with_upright_head_for_neck_padding(host,initial_pose_sdk):
    assert host.trunk_target_environment(initial_pose_sdk)=='-45 -81 -36 0 0 0'


@pytest.mark.parametrize('targets', [[0]*5,[0]*7,[True,0,0,0,0,0],[float('nan'),0,0,0,0,0],
                                     [float('inf'),0,0,0,0,0],[-45,-81,-36,0,0,-30],'knee_bend_2',None])
def test_invalid_initial_pose_fails_before_hardware_start(host,initial_pose_sdk,targets):
    config=initial_pose_sdk/'teach_script/teach_arm_config_init_pose.yaml'
    config.write_text(yaml.safe_dump({'arm_sequence':[{'trunk_position':targets}]}))
    with pytest.raises(ValueError):host.trunk_target_environment(initial_pose_sdk)


@pytest.mark.parametrize('config', [None,{}, {'arm_sequence':[]}, {'arm_sequence':[None]},
                                  {'arm_sequence':[{'trunk_position':[0]*6}]*2}])
def test_missing_or_ambiguous_initial_pose_is_rejected(host,initial_pose_sdk,config):
    (initial_pose_sdk/'teach_script/teach_arm_config_init_pose.yaml').write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):host.trunk_target_environment(initial_pose_sdk)
