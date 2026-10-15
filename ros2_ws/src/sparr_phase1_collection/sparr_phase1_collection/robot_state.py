"""ROS-independent adapter for a canonical Flexiv ``RobotStates`` snapshot.

Every source field is preserved; nothing is clamped or filled in. Invalid
quaternions are rejected at construction; other validation is exposed separately
so the runtime can make an explicit reject/hold decision.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .pose_math import Pose


NANOSECONDS_PER_SECOND = 1_000_000_000
_MISSING = object()
_NO_DEFAULT = object()


class RobotStateAdapterError(ValueError):
    """Raised when a source object cannot be represented faithfully."""


@dataclass(frozen=True, slots=True)
class Vector3:
    x: float
    y: float
    z: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def is_finite(self) -> bool:
        return all(math.isfinite(value) for value in self.as_tuple())


@dataclass(frozen=True, slots=True)
class SpatialVector:
    """A six-dimensional vector represented as linear and angular triples."""

    linear: Vector3
    angular: Vector3

    def is_finite(self) -> bool:
        return self.linear.is_finite() and self.angular.is_finite()


@dataclass(frozen=True, slots=True)
class Wrench:
    force: Vector3
    torque: Vector3

    def is_finite(self) -> bool:
        return self.force.is_finite() and self.torque.is_finite()


@dataclass(frozen=True, slots=True)
class RobotStateSnapshot:
    """One canonical, immutable robot sample.

    ``source_fields`` is a recursively frozen, plain-Python copy of every source
    field; typed attributes are conveniences, not a filtered replacement.

    The flange pose is the task anchor; the TCP pose is recorded but never
    returned by ``anchor_pose`` or ``success_pose``.
    """

    ros_timestamp_ns: int
    robot_timestamp_ns: int | None
    frame_id: str

    q: tuple[float, ...]
    theta: tuple[float, ...]
    dq: tuple[float, ...]
    dtheta: tuple[float, ...]
    tau: tuple[float, ...]
    tau_des: tuple[float, ...]
    tau_dot: tuple[float, ...]
    tau_ext: tuple[float, ...]
    tau_interact: tuple[float, ...]
    temperature: tuple[float, ...]

    tcp_pose: Pose | None
    tcp_velocity: SpatialVector | None
    flange_pose: Pose | None
    ft_sensor_raw: Wrench | None
    ext_wrench_in_tcp: Wrench | None
    ext_wrench_in_world: Wrench | None
    ext_wrench_in_tcp_raw: Wrench | None
    ext_wrench_in_world_raw: Wrench | None

    source_fields: Mapping[str, Any]


    @property
    def anchor_pose(self) -> Pose:
        """Return the flange pose used to anchor a policy action."""

        if self.flange_pose is None:
            raise RobotStateAdapterError("flange_pose is unavailable")
        return self.flange_pose

    @property
    def success_pose(self) -> Pose:
        """Return the flange pose used for the task-success calculation."""

        return self.anchor_pose


    def as_record(self) -> dict[str, Any]:
        """Return a mutable plain-Python copy suitable for serialization."""

        return _thaw(self.source_fields)

    def source_frame_id(self, field_name: str) -> str | None:
        """Return a nested stamped field's source frame without inventing one."""

        value = self.source_fields.get(str(field_name))
        if not isinstance(value, Mapping):
            return None
        header = value.get("header")
        if not isinstance(header, Mapping) or "frame_id" not in header:
            return None
        frame_id = str(header["frame_id"])
        return frame_id or None

    def validation_errors(
        self,
        *,
        expected_joint_count: int | None = 7,
        quaternion_norm_tolerance: float = 0.01,
        require_robot_timestamp: bool = True,
        expected_pose_frame_id: str | None = None,
    ) -> tuple[str, ...]:
        """Describe invalid state without altering it.

        The list is deterministic and intended to be stored with a rejected
        sample.  Freshness is evaluated by :mod:`safety`, because it requires a
        caller-provided current time.
        """

        errors: list[str] = []
        if self.ros_timestamp_ns < 0:
            errors.append("header.stamp is negative")
        if require_robot_timestamp and self.robot_timestamp_ns is None:
            errors.append("robot_timestamp is missing")
        elif self.robot_timestamp_ns is not None and self.robot_timestamp_ns < 0:
            errors.append("robot_timestamp is negative")

        joint_fields = {
            "q": self.q,
            "theta": self.theta,
            "dq": self.dq,
            "dtheta": self.dtheta,
            "tau": self.tau,
            "tau_des": self.tau_des,
            "tau_dot": self.tau_dot,
            "tau_ext": self.tau_ext,
            "tau_interact": self.tau_interact,
            "temperature": self.temperature,
        }
        for name, values in joint_fields.items():
            if expected_joint_count is not None and len(values) != expected_joint_count:
                errors.append(
                    f"{name} has {len(values)} values; expected {expected_joint_count}"
                )
            if any(not math.isfinite(value) for value in values):
                errors.append(f"{name} contains a non-finite value")

        if self.flange_pose is None:
            errors.append("flange_pose is missing")

        poses = {"tcp_pose": self.tcp_pose, "flange_pose": self.flange_pose}
        for name, pose in poses.items():
            if pose is None:
                continue
            if expected_pose_frame_id is not None:
                source_frame = self.source_frame_id(name)
                if source_frame is None:
                    errors.append(f"{name}.header.frame_id is missing")
                elif source_frame != expected_pose_frame_id:
                    errors.append(
                        f"{name}.header.frame_id {source_frame!r} does not match "
                        f"{expected_pose_frame_id!r}"
                    )
            pose_values = tuple(float(value) for value in pose.position) + tuple(
                float(value) for value in pose.quaternion_xyzw
            )
            if any(not math.isfinite(value) for value in pose_values):
                errors.append(f"{name} contains a non-finite value")
                continue
            quaternion_norm = math.sqrt(sum(value * value for value in pose_values[3:]))
            norm_error = abs(quaternion_norm - 1.0)
            if norm_error > quaternion_norm_tolerance:
                errors.append(
                    f"{name}.quaternion norm error {norm_error:.9g} exceeds "
                    f"{quaternion_norm_tolerance:.9g}"
                )

        spatial_fields = {
            "tcp_velocity": self.tcp_velocity,
            "ft_sensor_raw": self.ft_sensor_raw,
            "ext_wrench_in_tcp": self.ext_wrench_in_tcp,
            "ext_wrench_in_world": self.ext_wrench_in_world,
            "ext_wrench_in_tcp_raw": self.ext_wrench_in_tcp_raw,
            "ext_wrench_in_world_raw": self.ext_wrench_in_world_raw,
        }
        for name, value in spatial_fields.items():
            if value is not None and not value.is_finite():
                errors.append(f"{name} contains a non-finite value")

        return tuple(errors)


def timestamp_to_ns(value: Any) -> int:
    """Convert common ROS/plain timestamp representations to nanoseconds."""

    if isinstance(value, bool):
        raise RobotStateAdapterError("a boolean is not a timestamp")
    if isinstance(value, int):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) == 2:
            return _seconds_nanoseconds_to_ns(value[0], value[1])

    nanoseconds = _get(value, "nanoseconds", _MISSING)
    if nanoseconds is not _MISSING:
        return _coerce_int(nanoseconds, "nanoseconds")

    seconds = _get(value, "sec", _MISSING)
    if seconds is _MISSING:
        seconds = _get(value, "secs", _MISSING)
    remainder = _get(value, "nanosec", _MISSING)
    if remainder is _MISSING:
        remainder = _get(value, "nsec", _MISSING)
    if seconds is not _MISSING and remainder is not _MISSING:
        return _seconds_nanoseconds_to_ns(seconds, remainder)

    stamp = _get(value, "stamp", _MISSING)
    if stamp is not _MISSING:
        return timestamp_to_ns(stamp)
    header = _get(value, "header", _MISSING)
    if header is not _MISSING:
        return timestamp_to_ns(header)

    raise RobotStateAdapterError(f"unsupported timestamp value: {type(value).__name__}")


def adapt_robot_states(message: Any) -> RobotStateSnapshot:
    """Create a canonical snapshot from a ROS message or message-like object."""

    if message is None:
        raise RobotStateAdapterError("RobotStates message is None")

    header = _get(message, "header", _MISSING)
    if header is _MISSING:
        raise RobotStateAdapterError("RobotStates.header is missing")
    stamp = _get(header, "stamp", _MISSING)
    if stamp is _MISSING:
        raise RobotStateAdapterError("RobotStates.header.stamp is missing")
    ros_timestamp_ns = timestamp_to_ns(stamp)
    frame_id = str(_get(header, "frame_id", ""))

    robot_stamp = _get(message, "robot_timestamp", _MISSING)
    robot_timestamp_ns = (
        None if robot_stamp is _MISSING else timestamp_to_ns(robot_stamp)
    )

    all_fields = to_plain_data(message)
    if not isinstance(all_fields, Mapping):
        raise RobotStateAdapterError("RobotStates did not expose named fields")

    return RobotStateSnapshot(
        ros_timestamp_ns=ros_timestamp_ns,
        robot_timestamp_ns=robot_timestamp_ns,
        frame_id=frame_id,
        q=_float_tuple(_get(message, "q", ()), "q"),
        theta=_float_tuple(_get(message, "theta", ()), "theta"),
        dq=_float_tuple(_get(message, "dq", ()), "dq"),
        dtheta=_float_tuple(_get(message, "dtheta", ()), "dtheta"),
        tau=_float_tuple(_get(message, "tau", ()), "tau"),
        tau_des=_float_tuple(_get(message, "tau_des", ()), "tau_des"),
        tau_dot=_float_tuple(_get(message, "tau_dot", ()), "tau_dot"),
        tau_ext=_float_tuple(_get(message, "tau_ext", ()), "tau_ext"),
        tau_interact=_float_tuple(_get(message, "tau_interact", ()), "tau_interact"),
        temperature=_float_tuple(_get(message, "temperature", ()), "temperature"),
        tcp_pose=_optional_pose(_get(message, "tcp_pose", None), "tcp_pose"),
        tcp_velocity=_optional_spatial_vector(
            _get(message, "tcp_vel", None), "tcp_vel"
        ),
        flange_pose=_optional_pose(
            _get(message, "flange_pose", None), "flange_pose"
        ),
        ft_sensor_raw=_optional_wrench(
            _get(message, "ft_sensor_raw", None), "ft_sensor_raw"
        ),
        ext_wrench_in_tcp=_optional_wrench(
            _get(message, "ext_wrench_in_tcp", None), "ext_wrench_in_tcp"
        ),
        ext_wrench_in_world=_optional_wrench(
            _get(message, "ext_wrench_in_world", None), "ext_wrench_in_world"
        ),
        ext_wrench_in_tcp_raw=_optional_wrench(
            _get(message, "ext_wrench_in_tcp_raw", None), "ext_wrench_in_tcp_raw"
        ),
        ext_wrench_in_world_raw=_optional_wrench(
            _get(message, "ext_wrench_in_world_raw", None), "ext_wrench_in_world_raw"
        ),
        source_fields=_freeze(all_fields),
    )


def to_plain_data(value: Any) -> Any:
    """Recursively copy a ROS/dataclass/message-like value to plain Python."""

    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, Mapping):
        return {str(key): to_plain_data(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_plain_data(item) for item in value]

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return to_plain_data(tolist())

    names = _field_names(value)
    if names:
        return {name: to_plain_data(_get(value, name)) for name in names}

    raise RobotStateAdapterError(
        f"cannot preserve unsupported field type {type(value).__name__}"
    )


def _field_names(value: Any) -> tuple[str, ...]:
    get_types = getattr(value, "get_fields_and_field_types", None)
    if callable(get_types):
        result = get_types()
        if isinstance(result, Mapping):
            return tuple(str(name) for name in result)
    if is_dataclass(value):
        return tuple(field.name for field in fields(value))
    if hasattr(value, "__dict__"):
        return tuple(
            name for name in vars(value) if not name.startswith("_")
        )
    slots = getattr(type(value), "__slots__", ())
    names: list[str] = []
    for raw_name in slots:
        candidate = raw_name[1:] if raw_name.startswith("_") else raw_name
        if hasattr(value, candidate):
            names.append(candidate)
    return tuple(names)


def _get(value: Any, name: str, default: Any = _NO_DEFAULT) -> Any:
    if isinstance(value, Mapping):
        if name in value:
            return value[name]
    elif hasattr(value, name):
        return getattr(value, name)
    if default is not _NO_DEFAULT:
        return default
    raise RobotStateAdapterError(
        f"{type(value).__name__} does not provide field {name!r}"
    )


def _coerce_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise RobotStateAdapterError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RobotStateAdapterError(f"{label} must be an integer") from exc
    if result != value:
        raise RobotStateAdapterError(f"{label} must be an exact integer")
    return result


def _seconds_nanoseconds_to_ns(seconds: Any, nanoseconds: Any) -> int:
    seconds_int = _coerce_int(seconds, "seconds")
    nanoseconds_int = _coerce_int(nanoseconds, "nanoseconds")
    if not 0 <= nanoseconds_int < NANOSECONDS_PER_SECOND:
        raise RobotStateAdapterError("nanoseconds must be in [0, 1000000000)")
    return seconds_int * NANOSECONDS_PER_SECOND + nanoseconds_int


def _float_tuple(values: Iterable[Any], label: str) -> tuple[float, ...]:
    if values is None:
        return ()
    try:
        return tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RobotStateAdapterError(f"{label} is not a numeric sequence") from exc


def _vector3(value: Any, label: str) -> Vector3:
    try:
        return Vector3(
            x=float(_get(value, "x")),
            y=float(_get(value, "y")),
            z=float(_get(value, "z")),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise RobotStateAdapterError(f"{label} is not a Vector3") from exc


def _optional_pose(value: Any, label: str) -> Pose | None:
    if value is None:
        return None
    pose = _get(value, "pose", value)
    try:
        position = _get(pose, "position")
        orientation = _get(pose, "orientation")
        position_values = [
            float(_get(position, "x")),
            float(_get(position, "y")),
            float(_get(position, "z")),
        ]
        quaternion_values = [
            float(_get(orientation, "x")),
            float(_get(orientation, "y")),
            float(_get(orientation, "z")),
            float(_get(orientation, "w")),
        ]
        if any(not math.isfinite(item) for item in position_values + quaternion_values):
            raise RobotStateAdapterError(f"{label} contains a non-finite value")
        norm = math.sqrt(sum(item * item for item in quaternion_values))
        if abs(norm - 1.0) > 0.01:
            raise RobotStateAdapterError(f"{label} quaternion is not unit length")
        return Pose(position_values, quaternion_values)
    except RobotStateAdapterError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise RobotStateAdapterError(f"{label} is not a Pose/PoseStamped") from exc


def _optional_spatial_vector(value: Any, label: str) -> SpatialVector | None:
    if value is None:
        return None
    payload = _get(value, "accel", _MISSING)
    if payload is _MISSING:
        payload = _get(value, "twist", value)
    try:
        return SpatialVector(
            linear=_vector3(_get(payload, "linear"), f"{label}.linear"),
            angular=_vector3(_get(payload, "angular"), f"{label}.angular"),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise RobotStateAdapterError(f"{label} is not a six-dimensional vector") from exc


def _optional_wrench(value: Any, label: str) -> Wrench | None:
    if value is None:
        return None
    payload = _get(value, "wrench", value)
    try:
        return Wrench(
            force=_vector3(_get(payload, "force"), f"{label}.force"),
            torque=_vector3(_get(payload, "torque"), f"{label}.torque"),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise RobotStateAdapterError(f"{label} is not a Wrench/WrenchStamped") from exc


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


__all__ = [
    "NANOSECONDS_PER_SECOND",
    "Pose",
    "RobotStateAdapterError",
    "RobotStateSnapshot",
    "SpatialVector",
    "Vector3",
    "Wrench",
    "adapt_robot_states",
    "timestamp_to_ns",
    "to_plain_data",
]
