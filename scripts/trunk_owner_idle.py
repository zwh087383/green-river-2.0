"""Vendor-launch placeholder: project motion host owns the trunk SDK bus.

This process opens no hardware. It is spawned by the vendor launch, so its
respawn policy cannot create a second six-axis hardware controller.
"""
import signal
import threading
stop=threading.Event()
signal.signal(signal.SIGTERM,lambda *_:stop.set())
signal.signal(signal.SIGINT,lambda *_:stop.set())
print('[trunk owner guard] Trunk hardware is owned by the project motion service.',flush=True)
stop.wait()
