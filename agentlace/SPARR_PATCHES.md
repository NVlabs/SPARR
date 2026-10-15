# Local patches to the vendored agentlace

This tree is a checkout of upstream `youliangtan/agentlace` at
`cf2c337c5e3694cdbfc14831b239bd657bc4894d`, with the changes below applied.
Every patched line is marked with a `# SPARR patch:` comment, so
`grep -rn "SPARR patch" .` lists them all.

Re-syncing against upstream means re-applying these. They are small and
independent.

## Shutdown deadlock in `zmq_wrapper/pipeline.py`

**Symptom.** Any process using a `TrainerServer` hangs forever on shutdown. For
us that meant the learner could not be stopped with Ctrl-C and the transport
tests never terminated, despite passing.

**Cause.** `Consumer.start()` calls `self.results_receiver.recv()` on a PULL
socket with no receive timeout, so the worker thread blocks in `recv()`
indefinitely. `Consumer.stop()` sets `is_kill = True` and then calls
`self.thread.join()` with no timeout — but the thread is parked inside `recv()`
and never gets to re-check the flag. `stop()` therefore deadlocks. The thread was
also created without `daemon=True`, so even abandoning `stop()` would leave the
interpreter unable to exit.

The sibling `zmq_wrapper/broadcast.py` already does this correctly: it sets
`RCVTIMEO = 1500` and catches `zmq.Again` to re-check the flag. The patch brings
`pipeline.py` in line with it.

**Changes:**

1. `Consumer.__init__` sets `RCVTIMEO = 1500` on the PULL socket.
2. `Consumer.start()` catches `zmq.Again` and continues, re-checking `is_kill`.
3. `Consumer.async_start()` marks the thread `daemon=True`.
4. `Consumer.stop()` joins with `timeout=5.0` rather than unbounded.

## Non-daemon listener thread in `zmq_wrapper/broadcast.py`

`BroadcastClient.async_start()` created its listener without `daemon=True`, so a
thread that is slow to notice `is_kill` delays interpreter exit. Its `stop()`
also joined without a timeout. Marked daemon and bounded the join at 5 s.
`RCVTIMEO` was already set upstream here, so this is a robustness fix rather
than a deadlock fix.

## Not patched, but worth knowing

`setup.py` declares `opencv-python`, the dead `zmq` placeholder, and the
`typing` stdlib backport. Installing this package with dependencies would pull
OpenCV 5 in ahead of the apt OpenCV that `cv_bridge` links against, breaking
image conversion with `KeyError: 16`. The Phase 2 image therefore puts this tree
on `PYTHONPATH` instead of pip-installing it; see `docker/phase2/Dockerfile.phase2`.

## Trimmed to what SPARR uses

Only the trainer, the ZeroMQ transport (`zmq_wrapper/` except `pair.py`),
`data/data_store.py` and `internal/` are kept. The gym action/inference
servers, the TFDS/RLDS/TF-Agents data writers, the samplers, and upstream's
`examples/`, `tests/`, `docs/` and CI workflow were removed. The upstream
README's examples refer to those files; see the upstream repository for them.
