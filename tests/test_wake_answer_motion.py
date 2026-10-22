import threading
from rag_system import robot_motion as motion


def test_first_answer_waits_for_wake_and_dispatches_once(monkeypatch):
    commands=[];polls=0
    def exchange(command,**kwargs):
        nonlocal polls
        commands.append(command)
        if command=='STATUS':
            polls+=1
            return {'status':'ok','state':'running' if polls<3 else 'ready','request_id':'wake-token'}
        return {'status':'accepted','request_id':'wake-token' if len(commands)==1 else 'answer-token'}
    monkeypatch.setattr(motion,'exchange',exchange)
    monkeypatch.setattr(motion,'select_motion_preset',lambda purpose:'wave_goodbye' if purpose=='wake_wave' else 'speech')
    client=motion.MotionClient()
    assert client.request('wake',1,'wake_wave');client._thread.join(1)
    client.advance(2)
    assert client.request('answer',2,'answer_gesture');client._thread.join(2);client.close()
    assert polls==3
    # Background result queries are read-only; dispatch order still has exactly two PRESETs.
    assert [c.split()[0] for c in commands if not c.startswith('RESULT ')]==['PRESET','STATUS','STATUS','STATUS','PRESET']
    assert commands[-1].endswith(' speech')


def test_interruption_discards_waiting_first_answer(monkeypatch):
    commands=[];polling=threading.Event();release=threading.Event()
    def exchange(command,**kwargs):
        commands.append(command)
        if command=='STATUS':
            polling.set();release.wait(2)
            return {'status':'ok','state':'running','request_id':'wake-token'}
        return {'status':'accepted','request_id':'wake-token'}
    monkeypatch.setattr(motion,'exchange',exchange)
    monkeypatch.setattr(motion,'select_motion_preset',lambda purpose:'wave_goodbye' if purpose=='wake_wave' else 'speech')
    client=motion.MotionClient()
    client.request('wake',1,'wake_wave');client._thread.join(1)
    client.request('answer',2,'answer_gesture');assert polling.wait(1)
    client.advance(3);release.set();client._thread.join(1);client.close()
    assert len([c for c in commands if c.startswith('PRESET')])==1


def test_wake_fault_does_not_send_answer_or_power_on(monkeypatch):
    commands=[]
    def exchange(command,**kwargs):
        commands.append(command)
        return {'status':'ok','state':'fault','request_id':'wake-token','detail':'driver failure'} if command=='STATUS' else {'status':'accepted','request_id':'wake-token'}
    monkeypatch.setattr(motion,'exchange',exchange)
    monkeypatch.setattr(motion,'select_motion_preset',lambda purpose:'wave_goodbye' if purpose=='wake_wave' else 'speech')
    client=motion.MotionClient()
    client.request('wake',1,'wake_wave');client._thread.join(1)
    client.request('answer',2,'answer_gesture');client._thread.join(1);client.close()
    assert len(commands)==2 and commands[-1]=='STATUS'
