#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the NVIDIA OneWay Noncommercial License (March 22, 2022).
# See the LICENSE file in the repository root for the full license text:
# https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf

"""Interactive multi-trial driver for Phase 1 collection.

Runs trials back to back by following the trial state on /sparr/phase1/diagnostics,
and asks the operator to confirm the collector's suggested label for each.

Usage: docker/phase1/session.sh run [count] [--yes]
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import select
import sys
import termios
import threading
import time
import tty

import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from rclpy.node import Node
from std_srvs.srv import Trigger

TERMINAL = {"succeeded", "timed_out", "rejected", "aborted"}
#: Pressed during a rollout: stop the trial at once and label it a failure.
STOP_KEY = "x"

SERVICES = (
    "start_trial",
    "begin_rollout",
    "stop_trial",
    "confirm_success",
    "confirm_failure",
    "discard_trial",
    "abort_session",
)


class TrialRunner(Node):
    def __init__(self) -> None:
        super().__init__("sparr_phase1_trial_runner")
        self._lock = threading.Lock()
        self._values: dict[str, str] = {}
        self._message = ""
        self.create_subscription(
            DiagnosticArray, "/sparr/phase1/diagnostics", self._on_diagnostics, 10
        )
        # Not self._clients: rclpy.Node already uses that attribute.
        self._service_clients = {
            name: self.create_client(Trigger, f"/sparr_phase1/{name}")
            for name in SERVICES
        }

    def _on_diagnostics(self, message: DiagnosticArray) -> None:
        for status in message.status:
            if status.name != "sparr_phase1_collection":
                continue
            with self._lock:
                self._values = {item.key: item.value for item in status.values}
                self._message = status.message

    def snapshot(self) -> tuple[dict[str, str], str]:
        with self._lock:
            return dict(self._values), self._message

    def wait_for_collector(self, timeout_s: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.snapshot()[0]:
                return True
            time.sleep(0.2)
        return False

    def call(self, name: str, timeout_s: float = 30.0) -> tuple[bool, str]:
        client = self._service_clients[name]
        if not client.wait_for_service(timeout_sec=5.0):
            return False, f"service /sparr_phase1/{name} unavailable"
        future = client.call_async(Trigger.Request())
        deadline = time.monotonic() + timeout_s
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not future.done():
            return False, "service call timed out"
        response = future.result()
        return bool(response.success), str(response.message)

    def wait_for_state(self, states: set[str], timeout_s: float) -> str | None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            state = self.snapshot()[0].get("trial_state")
            if state in states:
                return state
            time.sleep(0.1)
        return None


@contextlib.contextmanager
def raw_key_reader():
    """Read single keypresses without Enter, restoring the terminal after.

    Yields a reader that always returns None when stdin is not a TTY.
    """
    if not sys.stdin.isatty():
        yield lambda: None
        return
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)

        def poll():
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1).lower()
            return None

        yield poll
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def prompt(text: str) -> str:
    try:
        return input(text).strip().lower()
    except EOFError:
        return "q"


class Tally:
    """Successes and completed trials (those labelled success or failure)."""

    def __init__(self, requested: int, window: int = 20) -> None:
        self.requested = requested
        self.completed = 0
        self.successes = 0
        self.window = window
        self.recent: collections.deque[bool] = collections.deque(maxlen=max(1, window))

    def record(self, label: str) -> None:
        if label in {"success", "failure"}:
            self.completed += 1
            self.successes += label == "success"
            self.recent.append(label == "success")

    @staticmethod
    def _rate(hits: int, total: int) -> str:
        return f"{hits}/{total} ({100.0 * hits / total if total else 0.0:.1f}%)"

    def line(self) -> str:
        requested = str(self.requested) if self.requested else "-"
        recent = (f"   |   last {self.window}: "
                  f"{self._rate(sum(self.recent), len(self.recent))}"
                  if self.window > 0 else "")
        return (f"successes {self._rate(self.successes, self.completed)}"
                + recent
                + f"   |   completed {self.completed}/{requested}")


SERVICE_LABEL = {
    "confirm_success": "success",
    "confirm_failure": "failure",
    "discard_trial": "discard",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "count", nargs="?", type=int, default=0,
        help="completed trials to run (success or failure; discarded attempts "
             "and reruns after a failed reset do not count); 0 means keep "
             "going until the session reaches target_successes",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="accept the collector's own outcome without prompting between "
             "trials, and carry on through rejected or aborted attempts",
    )
    parser.add_argument(
        "--max-consecutive-rejects", type=int, default=10,
        help="with --yes, give up after this many rejected/aborted attempts in "
             "a row; 0 means never give up. A run of rejects usually means the "
             "rig needs attention (a reposition tolerance that cannot be met, "
             "a stale camera), not that the next attempt will fare better.",
    )
    parser.add_argument(
        "--window", type=int, default=20,
        help="also show the success rate over the last N completed trials "
             "(default 20; 0 hides it)",
    )
    parser.add_argument(
        "--trial-timeout-s", type=float, default=180.0,
        help="give up waiting for one attempt to finish (default 180)",
    )
    args = parser.parse_args()

    rclpy.init()
    node = TrialRunner()
    spin = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True
    )
    spin.start()

    try:
        if not node.wait_for_collector():
            print("ERROR: no diagnostics from the collector. Is it running?", file=sys.stderr)
            return 2

        tally = Tally(args.count, args.window)
        consecutive_rejects = 0
        while True:
            values, _ = node.snapshot()
            if values.get("session_complete") == "True":
                print("\nSession complete.")
                break
            accepted = int(values.get("accepted_successes", 0))
            target = int(values.get("target_successes", 0))
            if target and accepted >= target:
                print(f"\nReached target_successes ({accepted}/{target}).")
                break
            if args.count and tally.completed >= args.count:
                print(f"\nCompleted {tally.completed} trials as requested.")
                print(tally.line())
                break

            state = node.wait_for_state({"idle"}, timeout_s=60.0)
            if state is None:
                print(f"ERROR: collector is not idle (state="
                      f"{node.snapshot()[0].get('trial_state')}); cannot start.",
                      file=sys.stderr)
                return 2

            print(f"\n--- trial {tally.completed + 1}"
                  + (f" of {args.count}" if args.count else "")
                  + f" ---   {tally.line()}"
                  + f"   ['{STOP_KEY}' during a rollout stops it as a failure]")
            if not args.yes:
                answer = prompt("Enter to RESET to the start pose, 'q' to quit: ")
                if answer == "q":
                    print("Stopping. The session is still open; "
                          "run 'session.sh abort' to close it cleanly.")
                    print(tally.line())
                    return 0

            ok, detail = node.call("start_trial")
            if not ok:
                print(f"start_trial refused: {detail}", file=sys.stderr)
                if "teach_reset_pose" in detail:
                    # Not retried: teaching captures the current pose, so the robot must be placed first.
                    print(
                        "\n  The reset pose has not been taught yet.\n"
                        "  1. Jog the tool to the pose every trial should start "
                        "from, a few cm from the socket.\n"
                        "  2. In another window:  session.sh teach\n"
                        "  3. Come back here and press 'r'.\n",
                        file=sys.stderr,
                    )
                answer = prompt("'r' to retry, anything else to quit: ")
                if answer == "r":
                    continue
                return 2

            print("repositioning...", flush=True)
            before_rollout = True

            # The collector holds in READY only with runtime.require_start_confirmation.
            state = node.wait_for_state({"ready", "running"} | TERMINAL,
                                        timeout_s=args.trial_timeout_s)
            if state is None:
                print("ERROR: collector did not reach the start pose in time.",
                      file=sys.stderr)
                return 2
            if state in TERMINAL:
                # Reposition was rejected; handled by the labelling block below.
                print(f"reposition ended early ({state}).", flush=True)
            elif state == "ready":
                if not args.yes:
                    answer = prompt("At start pose. Enter to RUN the policy, "
                                    "'s' to skip this attempt, 'q' to quit: ")
                    if answer == "q":
                        print("Stopping. The robot is holding the start pose and the "
                              "session is still open; run 'session.sh abort' to close it.")
                        return 0
                    if answer == "s":
                        ok, detail = node.call("discard_trial")
                        if not ok:
                            print(f"discard_trial refused: {detail}", file=sys.stderr)
                        continue
                ok, detail = node.call("begin_rollout")
                if not ok:
                    print(f"begin_rollout refused: {detail}", file=sys.stderr)
                    answer = prompt("'r' to retry this attempt, anything else to quit: ")
                    if answer == "r":
                        continue
                    return 2

            seen_running = False
            final = None
            stopped_by_operator = False
            deadline = time.monotonic() + args.trial_timeout_s
            with raw_key_reader() as read_key:
                while time.monotonic() < deadline:
                    key = read_key()
                    if key == STOP_KEY and not stopped_by_operator:
                        ok, detail = node.call("stop_trial")
                        stopped_by_operator = True
                        print(f"\n'{STOP_KEY}' pressed -- "
                              + ("stopping the trial, will be marked a failure"
                                 if ok else f"stop refused: {detail}"),
                              flush=True)
                    state = node.snapshot()[0].get("trial_state")
                    if state in TERMINAL:
                        final = state
                        break
                    if state == "running" and not seen_running:
                        seen_running = True
                        before_rollout = False
                        print("policy running...", flush=True)
                    time.sleep(0.1)
            if final is None:
                print("ERROR: attempt did not finish in time.", file=sys.stderr)
                return 2

            values, message = node.snapshot()
            print(f"outcome: {final}   ({message})")
            if final not in {"rejected", "aborted"}:
                consecutive_rejects = 0

            if final in {"rejected", "aborted"}:
                # Rejects and aborts are never auto-labelled.
                if before_rollout:
                    print("The policy never ran; the same trial will be run again.")
                else:
                    print("This attempt was not a normal completion; it stays in "
                          "the file as a non-accepted attempt.")
                # Only an operator service returns a terminal state to IDLE.
                ok, detail = node.call("discard_trial")
                if not ok:
                    print(f"could not clear the attempt: {detail}", file=sys.stderr)
                    return 2
                if args.yes:
                    consecutive_rejects += 1
                    limit = args.max_consecutive_rejects
                    if limit and consecutive_rejects >= limit:
                        print(f"ERROR: {consecutive_rejects} rejected/aborted "
                              "attempts in a row; stopping so the cause can be "
                              "inspected. Raise --max-consecutive-rejects, or 0 "
                              "to disable, if this is expected.",
                              file=sys.stderr)
                        return 1
                    print(f"Continuing ({consecutive_rejects} reject(s) in a row).")
                    continue
                answer = prompt("Enter to continue, 'q' to quit: ")
                if answer == "q":
                    return 1
                continue

            # The geometric success check can miss a seated plug, so a timed-out trial may be
            # labelled a success by the operator. Operator stops and runaways are always failures.
            forced_failure = stopped_by_operator or (
                values.get("trial_state") == "timed_out"
                and "runaway" in message.lower()
            )
            suggested = "success" if final == "succeeded" else "failure"
            if forced_failure:
                print("recording this attempt as a failure.")
                ok, detail = node.call("confirm_failure")
                print(f"failure: {'ok' if ok else 'REFUSED'} ({detail})")
                if ok:
                    tally.record("failure")
                    print(tally.line())
                continue
            if args.yes:
                choice = suggested
            else:
                answer = prompt(
                    f"label [{suggested}]  -- Enter to accept, "
                    f"'f' failure, 's' success, 'd' discard, 'q' quit: "
                )
                if answer == "q":
                    print(tally.line())
                    return 0
                choice = {
                    "": suggested, "s": "success", "f": "failure", "d": "discard"
                }.get(answer, suggested)

            service = {
                "success": "confirm_success",
                "failure": "confirm_failure",
                "discard": "discard_trial",
            }[choice]
            ok, detail = node.call(service)
            print(f"{choice}: {'ok' if ok else 'REFUSED'} ({detail})")
            while not ok:
                if "cannot be relabelled a success" in detail:
                    print(
                        "\n  This trial stopped on a guard or an abort, so its final\n"
                        "  observation is marked invalid. Labelling it a success would\n"
                        "  put knowingly broken data into the training set. Discard it.\n",
                        file=sys.stderr,
                    )
                elif "not export-eligible" in detail:
                    print(
                        "\n  This trial is debug-only, so a success label would not be\n"
                        "  exported anyway. The usual cause is policy.stochastic=false\n"
                        "  or shadow mode. Confirm failure or discard.\n",
                        file=sys.stderr,
                    )
                if args.yes:
                    # Unattended: discard rather than guess a label.
                    ok, detail = node.call("discard_trial")
                    print(f"--yes: label refused ({detail.strip()}); "
                          f"discarded instead of guessing")
                    service, ok = "discard_trial", False
                    break
                answer = prompt(
                    "label again -- 's' success, 'f' failure, 'd' discard, 'q' quit: "
                )
                if answer == "q":
                    print("Leaving the trial unfinalised; run 'session.sh failure' "
                          "or 'session.sh discard' then 'session.sh abort'.")
                    return 2
                service = {
                    "s": "confirm_success", "f": "confirm_failure", "d": "discard_trial"
                }.get(answer, "confirm_failure")
                ok, detail = node.call(service)
                print(f"{service}: {'ok' if ok else 'REFUSED'} ({detail})")
            if ok:
                tally.record(SERVICE_LABEL[service])
            print(tally.line())

        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
