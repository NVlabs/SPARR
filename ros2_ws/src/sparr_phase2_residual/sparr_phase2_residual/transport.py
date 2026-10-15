"""Actor/learner transport over agentlace (ZeroMQ).

Two channels:

* **5588** — request/data. The actor pushes transitions and stats to the learner.
* **5589** — broadcast. The learner publishes updated actor parameters.

``actor_env`` feeds the online buffer and ``actor_env_ideal`` the demonstration
buffer; a promoted episode is sent on both.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

DEFAULT_PORT = 5588
DEFAULT_BROADCAST_PORT = 5589

#: Wire names for the two data streams. Both ends must agree.
ONLINE_STORE = "actor_env"
DEMO_STORE = "actor_env_ideal"


class TransportError(RuntimeError):
    """Raised when the actor/learner link cannot be established."""


def trainer_config(
    port: int = DEFAULT_PORT, broadcast_port: int = DEFAULT_BROADCAST_PORT
) -> Any:
    """The shared agentlace config; the client refuses a server whose config differs."""
    from agentlace.trainer import TrainerConfig

    return TrainerConfig(
        port_number=port,
        broadcast_port=broadcast_port,
        request_types=["send-stats"],
    )


def start_learner_server(
    online_store: Any,
    demo_store: Any,
    *,
    port: int = DEFAULT_PORT,
    broadcast_port: int = DEFAULT_BROADCAST_PORT,
    on_stats: Optional[Callable[[dict], None]] = None,
) -> Any:
    """Serve the two buffers to a remote actor.

    Inserts from a server thread, so the stores must be thread-safe.
    """
    from agentlace.trainer import TrainerServer

    def request_callback(request_type: str, payload: dict) -> dict:
        if request_type != "send-stats":
            raise TransportError(f"unexpected request type {request_type!r}")
        if on_stats is not None:
            on_stats(payload)
        return {}

    server = TrainerServer(
        trainer_config(port, broadcast_port), request_callback=request_callback
    )
    server.register_data_store(ONLINE_STORE, online_store)
    server.register_data_store(DEMO_STORE, demo_store)
    server.start(threaded=True)
    return server


def stop_quietly(*endpoints: Any, timeout_s: float = 10.0) -> list[str]:
    """Shut down transport endpoints without letting one hang the process.

    agentlace's ``stop()`` can block forever in ``recv()``, so each stop runs in
    a daemon thread with a timeout. Returns failure descriptions (empty if clean).
    """
    import threading

    problems: list[str] = []
    for endpoint in endpoints:
        if endpoint is None:
            continue
        name = type(endpoint).__name__
        outcome: list[str] = []

        def shutdown(target: Any = endpoint, sink: list = outcome) -> None:
            try:
                target.stop()
            except Exception as exc:  # noqa: BLE001
                sink.append(f"{type(exc).__name__}: {exc}")

        worker = threading.Thread(target=shutdown, daemon=True)
        worker.start()
        worker.join(timeout=timeout_s)
        if worker.is_alive():
            problems.append(f"{name}.stop() did not return within {timeout_s:.0f}s")
        elif outcome:
            problems.append(f"{name}.stop() raised {outcome[0]}")
    return problems


def connect_actor_client(
    online_store: Any,
    demo_store: Any,
    *,
    host: str = "localhost",
    port: int = DEFAULT_PORT,
    broadcast_port: int = DEFAULT_BROADCAST_PORT,
    on_params: Optional[Callable[[Any], None]] = None,
    wait_for_server: bool = True,
    timeout_ms: int = 3000,
) -> Any:
    """Connect the robot-side actor to a learner."""
    from agentlace.trainer import TrainerClient

    client = TrainerClient(
        ONLINE_STORE,
        host,
        trainer_config(port, broadcast_port),
        data_stores={ONLINE_STORE: online_store, DEMO_STORE: demo_store},
        wait_for_server=wait_for_server,
        timeout_ms=timeout_ms,
    )
    if on_params is not None:
        client.recv_network_callback(on_params)
    return client
