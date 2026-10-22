import importlib.util
import io
from pathlib import Path
import socket
import threading


def host_module():
    path=Path(__file__).parents[1]/'scripts/sdk_motion_host.py'
    spec=importlib.util.spec_from_file_location('host_log_forwarding',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_driver_error_is_forwarded_with_exact_time_and_bounded_wire_length(tmp_path):
    module=host_module();path=tmp_path/'driver.sock';commands=[]
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
        server.bind(str(path));server.listen(1)
        def receive():
            stream,_=server.accept()
            with stream:
                commands.append(stream.recv(4096).decode());stream.sendall(b'{"status":"ok"}\n')
        thread=threading.Thread(target=receive);thread.start()
        line='[ERROR] [1792589894.321603398] [right_arm.hardware_interface]: '+('trajectory failure '*20)+'(10-kCommandJointVelocityLimitsViolation)'
        assert module.forward_arm_log(line,str(path))
        thread.join(1)
    command=commands[0];assert len(command)<=301
    parts=command.split();assert parts[:4]==['DRIVER_LOG','1792589894321603398','40','right_arm.hardware_interface']
    assert '10-kCommandJointVelocityLimitsViolation' in bytes.fromhex(parts[4]).decode()


def test_pipeline_preserves_output_and_only_forwards_one_info_then_errors(monkeypatch,capsys):
    module=host_module();calls=[]
    monkeypatch.setattr(module,'forward_arm_log',lambda line,path:calls.append(line) or True)
    lines='[INFO] connected\n[INFO] normal\n[ERROR] failed\n[INFO] end\n'
    module.forward_sdk_output(io.StringIO(lines),'unused')
    assert calls==['[INFO] connected\n','[ERROR] failed\n']
    assert capsys.readouterr().out==lines


def test_unrelated_or_malformed_logs_never_use_control_socket():
    module=host_module()
    for line in ('[motion] FAULT old','[ERROR] [1.0] [web]: error','not a log'):
        assert module.forward_arm_log(line,'/does-not-exist') is False
