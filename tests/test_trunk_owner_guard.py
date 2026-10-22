from pathlib import Path
import importlib.util
import pytest

ROOT=Path(__file__).parents[1]


def test_vendor_guard_is_reversible_idempotent_and_required_before_hardware(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    from trunk_owner_guard import install_guard,verify_guard,paths
    root=tmp_path/'project';sdk=tmp_path/'sdk'
    (root/'scripts').mkdir(parents=True)
    (root/'scripts/trunk_owner_idle.py').write_bytes((ROOT/'scripts/trunk_owner_idle.py').read_bytes())
    executable,original,guard=paths(sdk)
    executable.parent.mkdir(parents=True)
    vendor_bytes=b'\x7fELFfake-vendor-executable'
    executable.write_bytes(vendor_bytes)
    with pytest.raises(RuntimeError):verify_guard(root,sdk)
    result=install_guard(root,sdk)
    assert original.read_bytes()==vendor_bytes
    assert str(original)==result['original_backup']
    assert b'GREENRIVER_TRUNK_OWNER_GUARD_V1' in executable.read_bytes()
    assert guard.read_bytes()==(root/'scripts/trunk_owner_idle.py').read_bytes()
    verify_guard(root,sdk)
    install_guard(root,sdk)
    assert original.read_bytes()==vendor_bytes
    guard.write_text('changed')
    with pytest.raises(RuntimeError):verify_guard(root,sdk)
